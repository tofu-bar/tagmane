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
    
    // パイプライン実行時のblocks参照用
    private List<TransformBlock<object?, object?>>? _currentBlocks = null;
    
    private readonly Dictionary<int, SemaphoreSlim> _gpuSemaphores = new();
    private readonly Dictionary<int, Counter> _gpuSemaphoreNoopObtainedCounters = new();

    // 前段のパイプラインの処理時間を追跡
    private readonly Dictionary<int, RingBuffer<double>> _stageProcessingTimes = new();
    private const int TIMING_WINDOW_SIZE = 10;

    // スループット監視用
    private readonly Dictionary<int, RingBuffer<double>> _stageThroughput = new();
    private readonly Dictionary<int, DateTime> _stageLastProcessTime = new();
    private readonly Dictionary<int, int> _stageProcessedCount = new();
    
    // キュー長監視用
    private readonly Dictionary<int, int> _stageQueueLengths = new();
    private readonly object _queueMonitorLock = new object();
    
    private const int MIN_RECORDS_FOR_ADJUSTMENT = 2; // 調整開始までに必要な最小レコード数（さらに短縮）
    private const int ADJUSTMENT_INTERVAL_MS = 2000;   // 調整間隔（ミリ秒）（さらに短縮）
    private const int TARGET_QUEUE_LENGTH = 2; // 目標キュー長
    private DateTime _lastAdjustmentTime = DateTime.MinValue;

    private void AddLogEntry(string message)
    {
        string logMessage = $"Pipeline: {DateTime.Now:HH:mm:ss} - {message}";
        // デバッグ用：LogUpdatedイベントがnullでないかチェック
        if (LogUpdated == null)
        {
            Console.WriteLine($"[DEBUG] LogUpdatedイベントがnullです: {logMessage}");
        }
        else
        {
            var handlerCount = LogUpdated.GetInvocationList().Length;
            Console.WriteLine($"[DEBUG] LogUpdatedイベントを発火 (ハンドラー数: {handlerCount}): {logMessage}");
        }
        LogUpdated?.Invoke(this, logMessage);
    }

    public AsyncPipelineService(int cpuConcurrencyLimit, int gpuConcurrencyLimit, List<PipelineStage> pipelineStages)
    {
        Console.WriteLine($"[DEBUG] AsyncPipelineServiceコンストラクタ開始 - CPU:{cpuConcurrencyLimit}, GPU:{gpuConcurrencyLimit}");
        _cpuConcurrencyLimit = cpuConcurrencyLimit;
        _initialGpuConcurrencyLimit = gpuConcurrencyLimit;
        _currentGpuConcurrencyLimit = gpuConcurrencyLimit;        
        _pipelineStages = pipelineStages;

        _processedItemsCounter = new Counter[_pipelineStages.Count];
        for (int i = 0; i < _pipelineStages.Count; i++) {
            _stageProcessingTimes[i] = new RingBuffer<double>(TIMING_WINDOW_SIZE);
            _stageThroughput[i] = new RingBuffer<double>(TIMING_WINDOW_SIZE);
            _stageLastProcessTime[i] = DateTime.Now;
            _stageProcessedCount[i] = 0;
            _stageQueueLengths[i] = 0;
            _processedItemsCounter[i] = new Counter();
            if (_pipelineStages[i].IsGpuStage) {
                _gpuSemaphores[i] = new SemaphoreSlim(gpuConcurrencyLimit);
                _gpuSemaphoreNoopObtainedCounters[i] = new Counter();
            }
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
        
        // スループット計算（枚/秒）
        var throughput = processingTimeMs > 0 ? 1000.0 / processingTimeMs : 0;
        _stageThroughput[stageIndex].Enqueue(throughput);
        
        // 処理カウントと時間の更新
        _stageProcessedCount[stageIndex]++;
        _stageLastProcessTime[stageIndex] = DateTime.Now;
    }

    private double CalculateCurrentThroughput(int stageIndex)
    {
        var throughputBuffer = _stageThroughput[stageIndex];
        if (throughputBuffer.Count == 0) return 0;
        
        return throughputBuffer.Average(); // 枚/秒
    }

    private void UpdateQueueLength(int stageIndex, int queueLength)
    {
        lock (_queueMonitorLock)
        {
            _stageQueueLengths[stageIndex] = queueLength;
        }
    }

    private int GetQueueLength(int stageIndex)
    {
        lock (_queueMonitorLock)
        {
            return _stageQueueLengths.GetValueOrDefault(stageIndex, 0);
        }
    }

    private int AdjustGpuConcurrency(int stageIndex, double currentProcessingTimeMs, int currentQueueLength = 0)
    {
        AddLogEntry($"並列度調整チェック開始 - Stage:{stageIndex}, Queue:{currentQueueLength}, ProcessTime:{currentProcessingTimeMs:F1}ms, 現在並列度:{_currentGpuConcurrencyLimit}");
        
        // 前回の調整から十分な時間が経過しているか確認
        if ((DateTime.Now - _lastAdjustmentTime).TotalMilliseconds < ADJUSTMENT_INTERVAL_MS) 
        {
            AddLogEntry($"調整間隔未達 - 経過時間:{(DateTime.Now - _lastAdjustmentTime).TotalMilliseconds:F0}ms < {ADJUSTMENT_INTERVAL_MS}ms");
            return 0;
        }

        // 十分なレコードが蓄積されているか確認
        if (_stageThroughput[stageIndex].Count < MIN_RECORDS_FOR_ADJUSTMENT) 
        {
            AddLogEntry($"レコード不足 - 現在:{_stageThroughput[stageIndex].Count} < 必要:{MIN_RECORDS_FOR_ADJUSTMENT}");
            return 0;
        }

        lock (_concurrencyLock)
        {
            _lastAdjustmentTime = DateTime.Now;
            var newLimit = _currentGpuConcurrencyLimit;
            var adjustmentReason = "";

            // 1. キュー長ベースの調整（最優先）
            if (currentQueueLength > TARGET_QUEUE_LENGTH * 3)
            {
                // キューが著しく長い場合は並列度を大幅に上げる
                var increase = Math.Min(2, _initialGpuConcurrencyLimit - _currentGpuConcurrencyLimit);
                newLimit = Math.Min(_currentGpuConcurrencyLimit + increase, _initialGpuConcurrencyLimit);
                adjustmentReason = $"キュー長過大 (Q:{currentQueueLength})";
            }
            else if (currentQueueLength > TARGET_QUEUE_LENGTH * 1.5)
            {
                // キューがやや長い場合は並列度を上げる
                newLimit = Math.Min(_currentGpuConcurrencyLimit + 1, _initialGpuConcurrencyLimit);
                adjustmentReason = $"キュー長高 (Q:{currentQueueLength})";
            }
            else if (currentQueueLength == 0 && _currentGpuConcurrencyLimit > 1)
            {
                // 2. スループットベースの調整
                var previousStageIndex = stageIndex - 1;
                if (previousStageIndex >= 0)
                {
                    var currentThroughput = CalculateCurrentThroughput(stageIndex);
                    var previousThroughput = CalculateCurrentThroughput(previousStageIndex);
                    
                    if (previousThroughput > 0 && currentThroughput > 0)
                    {
                        var throughputRatio = previousThroughput / currentThroughput;
                        
                        if (throughputRatio > 1.5 && _currentGpuConcurrencyLimit < _initialGpuConcurrencyLimit)
                        {
                            // 前段のスループットが現段より50%以上高い場合は並列度を上げる
                            newLimit = Math.Min(_currentGpuConcurrencyLimit + 1, _initialGpuConcurrencyLimit);
                            adjustmentReason = $"スループット不均衡 (前段:{previousThroughput:F1}→現段:{currentThroughput:F1}枚/秒)";
                        }
                        else if (throughputRatio < 0.8)
                        {
                            // 前段より現段の方が20%以上速い場合は並列度を下げる
                            newLimit = Math.Max(_currentGpuConcurrencyLimit - 1, 1);
                            adjustmentReason = $"GPU過剰 (前段:{previousThroughput:F1}→現段:{currentThroughput:F1}枚/秒)";
                        }
                    }
                }
                
                // キューが空で他に調整要因がない場合は並列度を下げる
                if (adjustmentReason == "" && _currentGpuConcurrencyLimit > 1)
                {
                    newLimit = Math.Max(_currentGpuConcurrencyLimit - 1, 1);
                    adjustmentReason = "キュー空";
                }
            }
            
            // 3. 理論値による上限チェック（Little's Lawの適用）
            var previousStageProcessingTime = GetAveragePreviousStageProcessingTime(stageIndex);
            if (previousStageProcessingTime > 0)
            {
                var theoreticalOptimalParallelism = Math.Ceiling(currentProcessingTimeMs / previousStageProcessingTime);
                var maxTheoreticalLimit = Math.Min((int)theoreticalOptimalParallelism + 1, _initialGpuConcurrencyLimit);
                
                if (newLimit > maxTheoreticalLimit)
                {
                    newLimit = maxTheoreticalLimit;
                    adjustmentReason += $" [理論上限:{theoreticalOptimalParallelism:F1}]";
                }
            }

            // 段階的調整（急激な変化を抑制）
            var maxChange = Math.Max(1, _initialGpuConcurrencyLimit / 4); // 最大25%ずつ変更
            if (newLimit > _currentGpuConcurrencyLimit + maxChange)
            {
                newLimit = _currentGpuConcurrencyLimit + maxChange;
            }
            else if (newLimit < _currentGpuConcurrencyLimit - maxChange)
            {
                newLimit = _currentGpuConcurrencyLimit - maxChange;
            }

            newLimit = Math.Max(1, Math.Min(_initialGpuConcurrencyLimit, newLimit));
            
            if (newLimit != _currentGpuConcurrencyLimit)
            {
                AddLogEntry($"GPU並列度調整: {_currentGpuConcurrencyLimit} → {newLimit} ({adjustmentReason})");
                _currentGpuConcurrencyLimit = newLimit;
            }
            
            return newLimit;
        }
    }

    public async Task ProcessAsync<TInput, TOutput>(
        IAsyncEnumerable<TInput> inputs,
        CancellationTokenSource cts)
    {
        Console.WriteLine("[DEBUG] ProcessAsync開始");
        AddLogEntry("パイプラインの処理を開始します。");

        _isProcessing = true;
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
        _currentBlocks = blocks; // 参照を保存
        using var semaphoreCPU = new SemaphoreSlim(_cpuConcurrencyLimit);
        _currentGpuConcurrencyLimit = 1;

        try
        {
            AddLogEntry("パイプラインの構築を開始します。");

            // パイプラインを定義する
            for (int i = 0; i < _pipelineStages.Count; i++) {
                var stage = _pipelineStages[i];
                var stageIndex = i; // ブロック内でインデックスを使用するためにキャプチャ
                var counter = _processedItemsCounter[i];
                var semaphore = stage.IsGpuStage
                    ? _gpuSemaphores[i]
                    : semaphoreCPU;

                if (stage.IsGpuStage)
                {
                    var noopObtainedCounter = _gpuSemaphoreNoopObtainedCounters[i];
                    while (noopObtainedCounter.Count() < _initialGpuConcurrencyLimit - 1)
                    {
                        noopObtainedCounter.Increment();
                        await semaphore.WaitAsync(cts.Token);
                    }
                }
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
                                    // キュー長を取得（可能な場合）
                                    var currentQueueLength = 0;
                                    try 
                                    {
                                        // _currentBlocksを使用してキュー長を取得
                                        if (_currentBlocks != null && stageIndex < _currentBlocks.Count)
                                        {
                                            currentQueueLength = _currentBlocks[stageIndex].InputCount;
                                            UpdateQueueLength(stageIndex, currentQueueLength);
                                        }
                                    }
                                    catch 
                                    {
                                        // キュー長取得に失敗した場合は0とする
                                        currentQueueLength = 0;
                                    }

                                    var oldLimit = _currentGpuConcurrencyLimit;
                                    var newLimit = AdjustGpuConcurrency(stageIndex, processingTime, currentQueueLength);
                                    if (newLimit == 0) return result;
                                    // ロックを取得してないがスレッドセーフ（AdjustGpuConcurrencyの副作用）
                                    if (newLimit != oldLimit)
                                    {
                                        var targetLocksToTake = _initialGpuConcurrencyLimit - newLimit;
                                        var noopCounter = _gpuSemaphoreNoopObtainedCounters[stageIndex];
                                        while (noopCounter.Count() < targetLocksToTake)
                                        {
                                            noopCounter.Increment();
                                            await semaphore.WaitAsync(cts.Token);
                                        }
                                        while (noopCounter.Count() > targetLocksToTake)
                                        {
                                            noopCounter.Decrement();
                                            semaphore.Release();
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
                            MaxDegreeOfParallelism = _cpuConcurrencyLimit,
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
            _currentBlocks = null; // 参照をクリア
            AddLogEntry("パイプラインの処理が完了しました。");
            await statusReportTask;
            foreach (var block in blocks)
            {
                block.Complete();
            }
        }
    }
}
