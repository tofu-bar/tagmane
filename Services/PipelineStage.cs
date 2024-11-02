using System;
using System.Threading.Tasks;

public class PipelineStage
{
    public Func<dynamic, Task<dynamic?>> ProcessFunc { get; }
    public bool IsGpuStage { get; }

    // null でパイプラインを流れる無効なタスクを表現する。
    // null check のために記述が冗長になるが、TPL Dataflow では他に良い書き方がない
    public PipelineStage(Func<dynamic?, Task<dynamic?>> processFunc, bool isGpuStage = false)
    {
        ProcessFunc = processFunc;
        IsGpuStage = isGpuStage;
    }
}
