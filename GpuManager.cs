using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace tagmane
{
    public class GpuInfo
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string DisplayName => $"GPU {Id}: {Name}";
        
        public GpuInfo(int id, string name)
        {
            Id = id;
            Name = name;
        }
    }
    
    public static class GpuManager
    {
        private static List<GpuInfo> _availableGpus = new List<GpuInfo>();
        private static bool _isInitialized = false;
        private static readonly object _lock = new object();
        
        public static List<GpuInfo> AvailableGpus
        {
            get
            {
                lock (_lock)
                {
                    return new List<GpuInfo>(_availableGpus);
                }
            }
        }
        
        public static bool IsInitialized => _isInitialized;
        
        public static async Task InitializeAsync()
        {
            if (_isInitialized)
                return;
                
            try
            {
                Debug.WriteLine("=== GpuManager.InitializeAsync() called ===");
                
                var gpus = await DetectGpusAsync();
                
                lock (_lock)
                {
                    _availableGpus = gpus;
                    _isInitialized = true;
                }
                
                Debug.WriteLine($"GPU detection completed. Found {gpus.Count} GPU(s)");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"GPU detection failed: {ex.Message}");
                
                // デフォルトGPUを追加
                lock (_lock)
                {
                    _availableGpus = new List<GpuInfo> { new GpuInfo(0, "デフォルト") };
                    _isInitialized = true;
                }
            }
        }
        
        private static async Task<List<GpuInfo>> DetectGpusAsync()
        {
            var gpus = new List<GpuInfo>();
            
            try
            {
                string appDir = AppDomain.CurrentDomain.BaseDirectory;
                string pythonExe = Path.Combine(appDir, ".venv", "Scripts", "python.exe");
                string scriptPath = Path.Combine(appDir, "caption_generator.py");
                
                // Python環境が存在しない場合はデフォルトを返す
                if (!File.Exists(pythonExe) || !File.Exists(scriptPath))
                {
                    Debug.WriteLine("Python environment not found, using default GPU");
                    return new List<GpuInfo> { new GpuInfo(0, "デフォルト") };
                }
                
                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{scriptPath}\" --list-gpus",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = appDir
                };
                
                using (var process = new Process { StartInfo = startInfo })
                {
                    process.Start();
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string errorOutput = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    Debug.WriteLine($"GPU detection output: {output}");
                    Debug.WriteLine($"GPU detection error: {errorOutput}");
                    
                    if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                    {
                        var lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                        var gpuEntries = new HashSet<string>(); // 重複を防ぐ
                        
                        foreach (var line in lines)
                        {
                            var trimmedLine = line.Trim();
                            
                            // "GPU X: Name" 形式の行を解析
                            if (trimmedLine.StartsWith("GPU ") && trimmedLine.Contains(":"))
                            {
                                gpuEntries.Add(trimmedLine);
                            }
                        }
                        
                        // GPU情報をパースしてリストに追加
                        foreach (var entry in gpuEntries.OrderBy(x => x))
                        {
                            // "GPU 0: NVIDIA GeForce RTX 3090" のような文字列を解析
                            string[] parts = entry.Split(new[] { ':' }, 2);
                            if (parts.Length == 2)
                            {
                                string gpuIdStr = parts[0].Replace("GPU ", "").Trim();
                                string gpuName = parts[1].Trim();
                                
                                if (int.TryParse(gpuIdStr, out int gpuId))
                                {
                                    gpus.Add(new GpuInfo(gpuId, gpuName));
                                }
                            }
                        }
                    }
                }
                
                // GPUが見つからない場合はデフォルトを追加
                if (gpus.Count == 0)
                {
                    gpus.Add(new GpuInfo(0, "デフォルト"));
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error detecting GPUs: {ex.Message}");
                gpus.Add(new GpuInfo(0, "デフォルト"));
            }
            
            return gpus;
        }
        
        public static GpuInfo GetGpu(int id)
        {
            lock (_lock)
            {
                return _availableGpus.FirstOrDefault(g => g.Id == id) ?? 
                       _availableGpus.FirstOrDefault() ?? 
                       new GpuInfo(0, "デフォルト");
            }
        }
    }
}