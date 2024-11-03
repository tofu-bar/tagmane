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

    // 前段のパイプラインの処理時間を追跡
    private readonly Dictionary<int, Queue<double>> _previousStageTimings = new();
    private const int TIMING_WINDOW_SIZE = 10;

    private void AddLogEntry(string message)
    {
        string logMessage = $"Pipeline: {DateTime.Now:HH:mm:ss} - {message}";
        LogUpdated?.Invoke(this, logMessage);
    }

    public AsyncPipelineService(int cpuConcurrencyLimit = 14, int gpuConcurrencyLimit = 14)
    {
        _cpuConcurrencyLimit = cpuConcurrencyLimit;
        _initialGpuConcurrencyLimit = gpuConcurrencyLimit;
        _currentGpuConcurrencyLimit = gpuConcurrencyLimit;
    }

    private double GetAveragePreviousStageTime(int stageIndex)
    {
        if (stageIndex <= 0 || !_previousStageTimings.ContainsKey(stageIndex - 1))
            return 0;

        var timings = _previousStageTimings[stageIndex - 1];
        return timings.Count > 0 ? timings.Average() : 0;
    }

    private void UpdatePreviousStageTime(int stageIndex, double processingTimeMs)
    {
        if (!_previousStageTimings.ContainsKey(stageIndex))
            _previousStageTimings[stageIndex] = new Queue<double>();

        var timings = _previousStageTimings[stageIndex];
        timings.Enqueue(processingTimeMs);
        
        while (timings.Count > TIMING_WINDOW_SIZE)
            timings.Dequeue();
    }

    private void AdjustGpuConcurrency(int stageIndex, double currentProcessingTimeMs)
    {
        var previousStageTime = GetAveragePreviousStageTime(stageIndex);
        if (previousStageTime <= 0) return;

        lock (_concurrencyLock)
        {
            var newLimit = _currentGpuConcurrencyLimit;
            
            if (currentProcessingTimeMs > previousStageTime * 1.2)
            {
                newLimit = Math.Max(1, _currentGpuConcurrencyLimit - 1);
            }
            else if (currentProcessingTimeMs < previousStageTime)
            {
                newLimit = Math.Min(_initialGpuConcurrencyLimit, _currentGpuConcurrencyLimit + 1);
            }

            newLimit = Math.Max(1, Math.Min(_initialGpuConcurrencyLimit, newLimit));
            
            if (newLimit != _currentGpuConcurrencyLimit)
            {
                _currentGpuConcurrencyLimit = newLimit;
            }
        }
    }

    public async Task ProcessAsync<TInput, TOutput>(
        IAsyncEnumerable<TInput> inputs,
        List<PipelineStage> stages,
        CancellationTokenSource cts)
    {
        AddLogEntry("パイプラインの処理を開始します。");

        var blocks = new List<TransformBlock<object?, object?>>();
        var processedItemsCounter = new Counter[stages.Count];
        using var semaphoreCPU = new SemaphoreSlim(_cpuConcurrencyLimit);

        for (int i = 0; i < stages.Count; i++)
        {
            processedItemsCounter[i] = new Counter();
        }

        _isProcessing = true;
        var progressObservable = Observable.Interval(TimeSpan.FromMilliseconds(_statusUpdateIntervalMs))
            .ToAsyncEnumerable();
        var statusReportTask = Task.Run(async () =>
        {
            await foreach (var _ in progressObservable)
            {
                if (!_isProcessing) break;
                ProgressUpdated?.Invoke(processedItemsCounter);
            }
        });

        try
        {
            AddLogEntry("パイプラインの構築を開始します。");

            // パイプラインを定義する
            for (int i = 0; i < stages.Count; i++) {
                var stage = stages[i];
                var stageIndex = i; // ブロック内でインデックスを使用するためにキャプチャ
                var counter = processedItemsCounter[i];
                var semaphore = stage.IsGpuStage
                    ? new SemaphoreSlim(Math.Max(1, _currentGpuConcurrencyLimit))
                    : semaphoreCPU;
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
                                UpdatePreviousStageTime(stageIndex, processingTime);

                                if (stage.IsGpuStage)
                                {
                                    AdjustGpuConcurrency(stageIndex, processingTime);
                                    
                                    // セマフォの調整を安全に行う
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

            // パイプラインの接続
            for (int i = 0; i < blocks.Count - 1; i++)
            {
                var targetBlock = blocks[i + 1] as ITargetBlock<object>;
                blocks[i].LinkTo(targetBlock!, new DataflowLinkOptions { PropagateCompletion = true });
            }
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
            
            // // すべてのブロックを順番に完了させる
            // for (int i = 0; i < blocks.Count; i++)
            // {
            //     if (i == 0)
            //     {
            //         blocks[i].Complete();
            //     }
            //     await blocks[i].Completion;
            //     AddLogEntry($"ステージ {i + 1} の処理が完了しました。");
            // }
            
            // AddLogEntry("すべてのパイプライン処理が完了しました。");

            AddLogEntry("パイプラインの完了処理を開始します。");
            blocks[0].Complete();
            AddLogEntry("1");
            var completionTasks = blocks.Select(b => b.Completion).ToArray();
            AddLogEntry("2");
            await Task.WhenAll(completionTasks).WaitAsync(cts.Token);
            AddLogEntry("3");
            
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
                    UpdatePreviousStageTime(stageIndex, processingTime);

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
