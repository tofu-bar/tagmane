using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using System.Linq;
using R3;
using System.Threading.Tasks.Dataflow;
using System.Diagnostics;

public class AsyncPipelineService
{
    public event Action<Counter[]>? ProgressUpdated;
    public event EventHandler<string>? LogUpdated;

    private readonly int _cpuConcurrencyLimit;
    private readonly int _initialGpuConcurrencyLimit;
    private readonly int _statusUpdateIntervalMs = 1000;
    private volatile int _currentGpuConcurrencyLimit;
    private readonly object _concurrencyLock = new object();
    private bool _isProcessing = false;
    private readonly Counter[] _processedItemsCounter;
    private readonly List<PipelineStage> _pipelineStages;

    // 並列度調整のために占有するGPUセマフォのカウント
    private int _gpuNoopSemaphoreCount = 0;

    // 前段のパイプラインの処理時間を追跡
    private readonly Dictionary<int, RingBuffer<double>> _stageProcessingTimes = new();
    private const int TIMING_WINDOW_SIZE = 10;

    private const int MIN_RECORDS_FOR_ADJUSTMENT = 10; // 調整開始までに必要な最小レコード数
    private const int ADJUSTMENT_INTERVAL_MS = 5000;   // 調整間隔（ミリ秒）
    private DateTime _lastAdjustmentTime = DateTime.MinValue;

    private void AddLogEntry(string message)
    {
        string logMessage = $"Pipeline: {DateTime.Now:HH:mm:ss} - {message}";
        LogUpdated?.Invoke(this, logMessage);
    }

    public AsyncPipelineService(int cpuConcurrencyLimit, int gpuConcurrencyLimit, List<PipelineStage> pipelineStages)
    {
        _cpuConcurrencyLimit = cpuConcurrencyLimit;
        _initialGpuConcurrencyLimit = gpuConcurrencyLimit;
        _currentGpuConcurrencyLimit = gpuConcurrencyLimit;
        
        _pipelineStages = pipelineStages;

        _processedItemsCounter = new Counter[_pipelineStages.Count];
        for (int i = 0; i < _pipelineStages.Count; i++) {
            _stageProcessingTimes[i] = new RingBuffer<double>(TIMING_WINDOW_SIZE);
            _processedItemsCounter[i] = new Counter();
        }
    }

    private double GetAveragePreviousStageProcessingTime(int stageIndex)
    {
        var previousStageIndex = stageIndex - 1;
        if (previousStageIndex < 0 || !_stageProcessingTimes.ContainsKey(previousStageIndex))
            return 0; // 無効を返り値0で表現

        var processingTimes = _stageProcessingTimes[previousStageIndex];
        return processingTimes.Count > 0 ? processingTimes.Average() : 0;
    }

    private void ReportStageProcessingTime(int stageIndex, double processingTimeMs)
    {
        _stageProcessingTimes[stageIndex].Enqueue(processingTimeMs);
    }

    private int AdjustGpuConcurrency(int stageIndex, double currentProcessingTimeMs)
    {
        var previousStageProcessingTime = GetAveragePreviousStageProcessingTime(stageIndex);
        if (previousStageProcessingTime <= 0) return 0;

        // 十分なレコードが蓄積されているか確認
        var previousStageIndex = stageIndex - 1;
        if (_stageProcessingTimes[previousStageIndex].Count < MIN_RECORDS_FOR_ADJUSTMENT) return 0;

        // 前回の調整から十分な時間が経過しているか確認
        if ((DateTime.Now - _lastAdjustmentTime).TotalMilliseconds < ADJUSTMENT_INTERVAL_MS) return 0;

        lock (_concurrencyLock)
        {
            _lastAdjustmentTime = DateTime.Now;

            var newLimit = _currentGpuConcurrencyLimit;
            
            // 調整ロジックをより慎重に
            var ratio = currentProcessingTimeMs / previousStageProcessingTime;
            
            if (ratio > 1.2)
            {
                // 処理時間が20%以上長い場合は減少
                newLimit = Math.Max(1, _currentGpuConcurrencyLimit - 1);
                AddLogEntry($"GPU並列度を下げます: {_currentGpuConcurrencyLimit} → {newLimit} (比率: {ratio:F2})");
            }
            else if (ratio < 0.9)
            {
                // 処理時間が10%以上短い場合は増加
                newLimit = Math.Min(_initialGpuConcurrencyLimit, _currentGpuConcurrencyLimit + 1);
                AddLogEntry($"GPU並列度を上げます: {_currentGpuConcurrencyLimit} → {newLimit} (比率: {ratio:F2})");
            }

            newLimit = Math.Max(1, Math.Min(_initialGpuConcurrencyLimit, newLimit));
            
            if (newLimit != _currentGpuConcurrencyLimit)
            {
                _currentGpuConcurrencyLimit = newLimit;
            }
            return newLimit;
        }
    }

    public async Task ProcessAsync<TInput, TOutput>(
        IAsyncEnumerable<TInput> inputs,
        CancellationTokenSource cts)
    {
        AddLogEntry("パイプラインの処理を開始します。");

        _isProcessing = true;
        _lastAdjustmentTime = DateTime.Now;
        var progressObservable = Observable.Interval(TimeSpan.FromMilliseconds(_statusUpdateIntervalMs))
            .ToAsyncEnumerable();
        var statusReportTask = Task.Run(async () =>
        {
            await foreach (var _ in progressObservable)
            {
                if (!_isProcessing) break;
                ProgressUpdated?.Invoke(_processedItemsCounter);
            }
        });

        var blocks = new List<TransformBlock<object?, object?>>();
        using var semaphoreCPU = new SemaphoreSlim(_cpuConcurrencyLimit);

        try
        {
            AddLogEntry("パイプラインの構築を開始します。");

            // パイプラインを定義する
            for (int i = 0; i < _pipelineStages.Count; i++) {
                var stage = _pipelineStages[i];
                var stageIndex = i; // ブロック内でインデックスを使用するためにキャプチャ
                var counter = _processedItemsCounter[i];
                var semaphore = stage.IsGpuStage
                    ? new SemaphoreSlim(Math.Max(1, _currentGpuConcurrencyLimit))
                    : semaphoreCPU;
                // Blockは構成した瞬間に中身の評価（依存関数の実行）が走るので、初期化処理はブロックの外で行う
                blocks.Add(
                    new TransformBlock<object?, object?>(
                        async input => {
                            await semaphore.WaitAsync(cts.Token);
                            counter.Increment();
                            try {
                                if (input == null) return null;
                                
                                var sw = Stopwatch.StartNew();
                                var result = await stage.ProcessFunc(input);
                                sw.Stop();

                                var processingTime = sw.ElapsedMilliseconds;
                                ReportStageProcessingTime(stageIndex, processingTime);

                                if (stage.IsGpuStage)
                                {
                                    var newLimit = AdjustGpuConcurrency(stageIndex, processingTime);
                                    if (newLimit == 0) return result;
                                    if (newLimit != _currentGpuConcurrencyLimit)
                                    {
                                        _currentGpuConcurrencyLimit = newLimit;
                                        var targetLocksToTake = _initialGpuConcurrencyLimit - _currentGpuConcurrencyLimit;
                                        while (_gpuNoopSemaphoreCount < targetLocksToTake)
                                        {
                                            await semaphore.WaitAsync(cts.Token);
                                            _gpuNoopSemaphoreCount++;
                                        }
                                        while (_gpuNoopSemaphoreCount > targetLocksToTake)
                                        {
                                            semaphore.Release();
                                            _gpuNoopSemaphoreCount--;
                                        }
                                    }
                                }

                                return result;
                            }
                            finally
                            {
                                semaphore.Release();
                            }
                        },
                        new ExecutionDataflowBlockOptions
                        {
                            MaxDegreeOfParallelism = stage.IsGpuStage ? 
                                Math.Max(1, _currentGpuConcurrencyLimit) : 
                                _cpuConcurrencyLimit,
                            CancellationToken = cts.Token,
                            SingleProducerConstrained = true,
                        }
                    )
                );
            }

            // 最後に完了検出用のActionBlockを追加
            var completionBlock = new ActionBlock<object?>(
                obj => { },
                new ExecutionDataflowBlockOptions
                {
                    CancellationToken = cts.Token
                }
            );

            // パイプラインの接続
            for (int i = 0; i < blocks.Count - 1; i++)
            {
                var targetBlock = blocks[i + 1] as ITargetBlock<object>;
                blocks[i].LinkTo(targetBlock!, new DataflowLinkOptions { PropagateCompletion = true });
            }
            // 最後のTransformBlockをActionBlockに接続
            blocks[blocks.Count - 1].LinkTo(completionBlock, new DataflowLinkOptions { PropagateCompletion = true });

            AddLogEntry("パイプラインの接続が完了しました。");

            // タスクを投入
            await foreach (var input in inputs)
            {
                // 全量を投入してもパイプラインは順次処理可能だが、ここではキャンセルの反応速度を上げるため、流量をセマフォで制御している（取得するロックは高々1)
                await semaphoreCPU.WaitAsync(cts.Token);
                if (input != null)
                {
                    try
                    {
                        await blocks[0].SendAsync(input, cts.Token);
                    }
                    catch (OperationCanceledException)
                    {
                        // キャンセルされた場合はロックを開放してループを抜ける
                        semaphoreCPU.Release();
                        break;
                    }
                    finally
                    {
                        semaphoreCPU.Release();
                    }
                }
                else
                {
                    semaphoreCPU.Release(); // nullの場合もリリースする
                }
            }

            // 完了処理
            AddLogEntry("入力処理が完了しました。パイプラインの完了処理を開始します。");
            
            // //全ブロックを完了させる
            // for (int i = 0; i < blocks.Count ; i++) 
            // {
            //     if (i == 0)
            //     {
            //         blocks[i].Complete();
            //     }
            //     await blocks[i].Completion;
            //     AddLogEntry($"ステージ {i + 1} の処理が完了しました。");
            // }

            // // 最後のダミーステージは完了させない
            // AddLogEntry("すべての実行ステージの処理が完了しました。");

            // 全ブロックの完了を待つ
            blocks[0].Complete();
            await blocks[blocks.Count - 1].Completion;
            var completionTasks = blocks.Take(blocks.Count - 1).Select(b => b.Completion).ToArray();
            await Task.WhenAll(completionTasks).WaitAsync(cts.Token);
            AddLogEntry("すべてのブロックの完了処理が終了しました。");
            
        }
        catch (OperationCanceledException)
        {
            AddLogEntry("パイプラインの処理がキャンセルされました。");
            foreach (var block in blocks)
            {
                block.Complete();
            }
            throw;
        }
        finally
        {
            _isProcessing = false;
            AddLogEntry("パイプラインの処理が完了しました。");
            await statusReportTask;
            foreach (var block in blocks)
            {
                block.Complete();
            }
        }
    }

    private TransformBlock<object?, object?> CreateTransformBlock(
        PipelineStage stage,
        int stageIndex,
        Counter counter,
        SemaphoreSlim semaphore,
        CancellationTokenSource cts)
    {
        return new TransformBlock<object?, object?>(
            async input =>
            {
                await semaphore.WaitAsync(cts.Token);
                counter.Increment();
                try
                {
                    if (input == null) return null;

                    var sw = Stopwatch.StartNew();
                    var result = await stage.ProcessFunc(input);
                    sw.Stop();

                    var processingTime = sw.ElapsedMilliseconds;
                    ReportStageProcessingTime(stageIndex, processingTime);

                    if (stage.IsGpuStage)
                    {
                        AdjustGpuConcurrency(stageIndex, processingTime);
                        await AdjustSemaphore(semaphore);
                    }

                    return result;
                }
                finally
                {
                    semaphore.Release();
                }
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = stage.IsGpuStage
                    ? Math.Max(1, _currentGpuConcurrencyLimit)
                    : _cpuConcurrencyLimit,
                CancellationToken = cts.Token,
                SingleProducerConstrained = true,
            });
    }

    private async Task AdjustSemaphore(SemaphoreSlim semaphore)
    {
        try
        {
            var currentCount = semaphore.CurrentCount;
            var targetCount = _currentGpuConcurrencyLimit;

            if (currentCount < targetCount)
            {
                for (int j = currentCount; j < targetCount; j++)
                {
                    semaphore.Release();
                }
            }
        }
        catch (SemaphoreFullException)
        {
            // セマフォが既に最大値に達している場合は無視
        }
    }
}
