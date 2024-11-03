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
    public event Action<string>? LogUpdated;

    private readonly int _cpuConcurrencyLimit;
    private readonly int _initialGpuConcurrencyLimit;
    private readonly int _statusUpdateIntervalMs = 1000;
    private volatile int _currentGpuConcurrencyLimit;
    private readonly object _concurrencyLock = new object();
    private bool _isProcessing = false;

    // 前段のパイプラインの処理時間を追跡
    private readonly Dictionary<int, Queue<double>> _previousStageTimings = new();
    private const int TIMING_WINDOW_SIZE = 10;

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
        var blocks = new List<TransformBlock<object?, object?>>();
        var semaphoreCPU = new SemaphoreSlim(_cpuConcurrencyLimit);
        var processedItemsCounter = new Counter[stages.Count];

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
            // パイプラインの構築
            for (int i = 0; i < stages.Count; i++)
            {
                var stage = stages[i];
                var stageIndex = i;
                var counter = processedItemsCounter[i];
                var semaphore = stage.IsGpuStage
                    ? new SemaphoreSlim(Math.Max(1, _currentGpuConcurrencyLimit))
                    : new SemaphoreSlim(_cpuConcurrencyLimit);

                blocks.Add(CreateTransformBlock(stage, stageIndex, counter, semaphore, cts));
            }

            // パイプラインの接続
            for (int i = 0; i < blocks.Count - 1; i++)
            {
                var targetBlock = blocks[i + 1] as ITargetBlock<object>;
                blocks[i].LinkTo(targetBlock!, new DataflowLinkOptions { PropagateCompletion = true });
            }

            // 入力処理
            await foreach (var input in inputs.WithCancellation(cts.Token))
            {
                await semaphoreCPU.WaitAsync(cts.Token);
                try
                {
                    if (input != null)
                    {
                        await blocks[0].SendAsync(input, cts.Token);
                    }
                }
                finally
                {
                    semaphoreCPU.Release();
                }
            }

            // 完了処理
            blocks[0].Complete();
            var completionTasks = blocks.Select(b => b.Completion).ToArray();
            await Task.WhenAll(completionTasks).WaitAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            foreach (var block in blocks)
            {
                block.Complete();
            }
            throw;
        }
        finally
        {
            _isProcessing = false;
            await statusReportTask;
            
            semaphoreCPU.Dispose();
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
