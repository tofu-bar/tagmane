using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using R3;
using System.Threading.Tasks.Dataflow;

// GPU処理の待機を別タスク(スレッド)へ切り出してCPU時間を有効に使うパイプライン実装
public class AsyncPipelineService
{
    public event Action<Counter[]>? ProgressUpdated;
    public event Action<string>? LogUpdated;

    private readonly int _cpuConcurrencyLimit;
    private readonly int _gpuConcurrencyLimit;
    private readonly int _statusUpdateIntervalMs = 1000;

    private bool _isProcessing = false;

    // CPU並列度設定の基準は (Environment.ProcessorCount - 2) = 14 (VLMPredictionでGPU処理の軽いjoytag利用時の最速設定)
    public AsyncPipelineService(int cpuConcurrencyLimit=14, int gpuConcurrencyLimit=1000)
    {
        _cpuConcurrencyLimit = cpuConcurrencyLimit;
        _gpuConcurrencyLimit = gpuConcurrencyLimit;
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
            var counter = processedItemsCounter[i]; // ブロック定義の外で参照しないと、ブロックの中で参照できない
            var semaphore = stage.IsGpuStage ? semaphoreGPU : semaphoreCPU;
            blocks.Add(
                new TransformBlock<object?, object?>(
                    async input => {
                        // 処理開始前にロックを取得 or 並列度の上限に達している間は処理を止める
                        // 比較的軽い処理でも、CPUで処理される場合は、並列度を制限しておくとUIスレッドのハングを防げる
                        await semaphore.WaitAsync(cts.Token);
                        counter.Increment();
                        try {
                            if (input == null) return null;
                            return await stage.ProcessFunc(input);
                        } finally {
                            semaphore.Release();
                        }
                    },
                    new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded } // 並列度の制限は外部的にセマフォで行う
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
