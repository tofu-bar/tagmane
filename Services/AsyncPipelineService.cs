using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using System.Linq; // この行を追加
using R3;
using System.Threading.Tasks.Dataflow;
using System.Diagnostics;

// GPU処理の待機を別タスク(スレッド)へ切り出してCPU時間を有効に使うパイプライン実装
public class AsyncPipelineService
{
    public event Action<Counter[]>? ProgressUpdated;
    public event Action<string>? LogUpdated;

    private readonly int _cpuConcurrencyLimit;
    private readonly int _gpuConcurrencyLimit;
    private readonly int _initialGpuConcurrencyLimit;
    private readonly int _statusUpdateIntervalMs = 1000;

    private bool _isProcessing = false;

    private volatile int _currentGpuConcurrencyLimit;
    private readonly object _concurrencyLock = new object();
    
    // 前段のパイプラインの処理時間を追跡
    private readonly Dictionary<int, Queue<double>> _previousStageTimings = new();
    private const int TIMING_WINDOW_SIZE = 10; // 直近の処理時間をいくつ保持するか

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
        if (previousStageTime <= 0) return; // 前段の処理時間が不明な場合は調整しない

        lock (_concurrencyLock)
        {
            var newLimit = _currentGpuConcurrencyLimit;
            
            // 現在の処理時間が前段の処理時間より長い場合は並列度を下げる
            if (currentProcessingTimeMs > previousStageTime * 1.2)
            {
                newLimit = Math.Max(1, _currentGpuConcurrencyLimit - 1);
            }
            // 現在の処理時間が前段の処理時間より十分短い場合は並列度を上げる
            else if (currentProcessingTimeMs < previousStageTime)
            {
                newLimit = Math.Min(_initialGpuConcurrencyLimit, _currentGpuConcurrencyLimit + 1);
            }

            // 値が1未満にならないよう保証
            newLimit = Math.Max(1, Math.Min(_initialGpuConcurrencyLimit, newLimit));
            
            if (newLimit != _currentGpuConcurrencyLimit)
            {
                _currentGpuConcurrencyLimit = newLimit;
            }
        }
    }

    // CPU並列度設定の基準は (Environment.ProcessorCount - 2) = 14 (VLMPredictionでGPU処理の軽いjoytag利用時の最速設定)
    public AsyncPipelineService(int cpuConcurrencyLimit=14, int gpuConcurrencyLimit=14)
    {
        _cpuConcurrencyLimit = cpuConcurrencyLimit;
        _gpuConcurrencyLimit = gpuConcurrencyLimit;
        _initialGpuConcurrencyLimit = gpuConcurrencyLimit;
    }

    public async Task ProcessAsync<TInput, TOutput>(
        IAsyncEnumerable<TInput> inputs,
        List<PipelineStage> stages,
        CancellationTokenSource cts)
    {
        var blocks = new List<TransformBlock<object?, object?>>();

        SemaphoreSlim semaphoreCPU = new SemaphoreSlim(_cpuConcurrencyLimit);
        SemaphoreSlim semaphoreGPU = new SemaphoreSlim(_gpuConcurrencyLimit);

        // パイプラインの数だけカウンターを定義する
        var processedItemsCounter = new Counter[stages.Count];
        for (int i = 0; i < stages.Count; i++) {
            processedItemsCounter[i] = new Counter();
        }

        // 一定間隔で処理状況を更新する。Rx(R3)でなくても実装できるが、便利でスレッドの処理も任せられて安心なのでR3を使う
        _isProcessing = true;
        var progressObservable = Observable.Interval(TimeSpan.FromMilliseconds(_statusUpdateIntervalMs)).ToAsyncEnumerable();
        var statusReportTask = Task.Run(async () => { 
            await foreach (var _ in progressObservable) {
                if (!_isProcessing) break;
                ProgressUpdated?.Invoke(processedItemsCounter);
            }
        });

        // パイプラインを定義する
        for (int i = 0; i < stages.Count; i++) {
            var stage = stages[i];
            var stageIndex = i; // ブロック内でインデックスを使用するためにキャプチャ
            var counter = processedItemsCounter[i];
            var semaphore = stage.IsGpuStage ? 
                new SemaphoreSlim(Math.Max(1, _currentGpuConcurrencyLimit)) : 
                new SemaphoreSlim(_cpuConcurrencyLimit);
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

        // 直列にパイプラインを接続
        for (int i = 0; i < blocks.Count - 1; i++) {
            var targetBlock = blocks[i+1] as ITargetBlock<object>;
            blocks[i].LinkTo(targetBlock!, new DataflowLinkOptions { PropagateCompletion = true });
        }

        try {
            // タスクを投入
            await foreach (var input in inputs) {
                // 全量を投入してもパイプラインは順次処理可能だが、ここではキャンセルの反応速度を上げるため、流量をセマフォで制御している（取得するロックは高々1)
                await semaphoreCPU.WaitAsync(cts.Token);
                if (input != null) {
                    try {
                        await blocks[0].SendAsync(input, cts.Token);
                    } catch (OperationCanceledException) {
                        // キャンセルされた場合はロックを開放してループを抜ける
                        semaphoreCPU.Release();
                        break;
                    } finally {
                        semaphoreCPU.Release();
                    }
                } else {
                    semaphoreCPU.Release(); // nullの場合もリリースする
                }
            }
            // 全量投入後、最初のブロックを完了させる
            blocks[0].Complete();
            // 最後のブロックの完了を待つ
            await blocks[blocks.Count - 1].Completion;
        } finally {
            // 処理スピード計測タスクをキャンセル
            _isProcessing = false;
            await statusReportTask;
        }
    }
}
