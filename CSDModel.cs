using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Numerics;

namespace tagmane
{
    public class CSDModel
    {
        private InferenceSession _session;
        private const int ImageSize = 224; // CSDモデルの入力サイズ
        private const string MODEL_FILENAME = "csd_clip_model.onnx";
        private const string MODEL_REPO = "yuxi-liu-wired/CSD";

        public ObservableCollection<string> CSDLogEntries { get; } = new ObservableCollection<string>();
        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        private void AddLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            Application.Current.Dispatcher.Invoke(() =>
            {
                CSDLogEntries.Insert(0, logMessage);
                while (CSDLogEntries.Count > 20) // 最大20エントリーまで保持
                {
                    CSDLogEntries.RemoveAt(CSDLogEntries.Count - 1);
                }
            });
            LogUpdated?.Invoke(this, $"CSDModel: {logMessage}");
        }

        public static async Task<CSDModel> LoadModel(bool useGpu = true)
        {
            var model = new CSDModel();
            model.AddLogEntry("CSDモデルのロードを開始します");

            int retryCount = 0;
            const int maxRetries = 3;
            string modelPath = null;

            while (retryCount < maxRetries)
            {
                try
                {
                    if (modelPath == null)
                    {
                        modelPath = await model.DownloadModel();
                    }

                    await model.InitializeSession(modelPath, useGpu);
                    model.AddLogEntry("CSDモデルのロードが完了しました");
                    return model;
                }
                catch (Exception ex)
                {
                    model.AddLogEntry($"モデルのロードに失敗しました: {ex.Message}");
                    retryCount++;

                    if (retryCount >= maxRetries)
                    {
                        model.AddLogEntry("最大再試行回数に達しました。モデルの読み込みに失敗しました。");
                        throw;
                    }

                    model.AddLogEntry($"モデルの再ダウンロードを試みます。試行回数: {retryCount}");
                    if (File.Exists(modelPath))
                    {
                        File.Delete(modelPath);
                        model.AddLogEntry($"既存のモデルファイルを削除しました: {modelPath}");
                    }
                    modelPath = null;
                }
            }

            throw new Exception("モデルのロードに失敗しました。");
        }

        private async Task<string> DownloadModel()
        {
            AddLogEntry($"リポジトリからモデルを読み込みます: {MODEL_REPO}");
            using var client = new HttpClient();
            var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", MODEL_REPO.Split('/').Last());
            Directory.CreateDirectory(modelDir);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);

            if (!File.Exists(modelPath))
            {
                var url = $"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}";
                AddLogEntry($"モデルをダウンロードします: {url}");
                var progress = new Progress<double>(p => Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = p * 100));
                await DownloadFileWithRetry(client, url, modelPath, progress: progress);
                AddLogEntry("モデルのダウンロードが完了しました");
            }
            else
            {
                AddLogEntry("既存のモデルファイルを使用します");
                Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 100);
            }

            Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 0);

            return modelPath;
        }

        private async Task DownloadFileWithRetry(HttpClient client, string url, string filePath, int maxRetries = 3, IProgress<double> progress = null)
        {
            for (int i = 0; i < maxRetries; i++)
            {
                try
                {
                    AddLogEntry($"ダウンロード試行 {i + 1}/{maxRetries}");
                    using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                    response.EnsureSuccessStatusCode();
                    long? totalBytes = response.Content.Headers.ContentLength;
                    using var contentStream = await response.Content.ReadAsStreamAsync();
                    using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

                    var totalRead = 0L;
                    var buffer = new byte[8192];
                    var isMoreToRead = true;

                    do
                    {
                        var read = await contentStream.ReadAsync(buffer, 0, buffer.Length);
                        if (read == 0)
                        {
                            isMoreToRead = false;
                        }
                        else
                        {
                            await fileStream.WriteAsync(buffer, 0, read);

                            totalRead += read;
                            if (totalBytes.HasValue)
                            {
                                var progressPercentage = (double)totalRead / totalBytes.Value;
                                progress?.Report(progressPercentage);
                            }
                        }
                    }
                    while (isMoreToRead);

                    AddLogEntry("ファイルのダウンロードが完了しました");
                    return;
                }
                catch (Exception ex)
                {
                    AddLogEntry($"ダウンロード中にエラーが発生しました: {ex.Message}");
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                        AddLogEntry("不完全なファイルを削除しました");
                    }

                    if (i == maxRetries - 1)
                        throw;
                }
                await Task.Delay(1000 * (i + 1));
            }
        }

        private async Task InitializeSession(string modelPath, bool useGpu)
        {
            var sessionOptions = new SessionOptions();
            if (useGpu)
            {
                AddLogEntry("GPU使用を試みます");
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                    IsGpuLoaded = true;
                    AddLogEntry("GPUの初期化に成功しました");
                }
                catch (Exception ex)
                {
                    AddLogEntry($"GPUの初期化に失敗しました: {ex.Message}");
                    AddLogEntry("CPUを使用します");
                    IsGpuLoaded = false;
                }
            }
            else
            {
                AddLogEntry("CPUを使用します");
                IsGpuLoaded = false;
            }
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            AddLogEntry("ONNX推論セッションを初期化しています");
            await Task.Run(() => _session = new InferenceSession(modelPath, sessionOptions));
            AddLogEntry("ONNX推論セッションの初期化が完了しました");
        }
        
        public async Task<Dictionary<string, float[]>> ExtractFeature(DenseTensor<Float16> inputTensor)
        {            
            var inputMetadata = _session.InputMetadata;
            var inputName = inputMetadata.Keys.FirstOrDefault() ?? throw new InvalidOperationException("モデルの入力名が見つかりません");
            // AddLogEntry($"モデルの入力名: {inputName}");

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            try
            {
                using (var outputs = await Task.Run(() => _session.Run(inputs)))
                {
                    // AddLogEntry($"出力の数: {outputs.Count}");
                    var result = new Dictionary<string, float[]>();

                    foreach (var output in outputs)
                    {
                        // AddLogEntry($"出力名: {output.Name}, 型: {output.GetType().Name}");
                        if (output.Value is DenseTensor<float> floatTensor)
                        {
                            var array = floatTensor.ToArray();
                            result[output.Name] = array;
                            // AddLogEntry($"{output.Name} の長さ: {array.Length}");
                            // AddLogEntry($"{output.Name} の最初の10要素: {string.Join(", ", array.Take(10))}");
                        }
                        else if (output.Value is DenseTensor<Float16> float16Tensor)
                        {
                            var array = float16Tensor.ToArray().Select(f => (float)f).ToArray();
                            result[output.Name] = array;
                            // AddLogEntry($"{output.Name} の長さ: {array.Length}");
                            // AddLogEntry($"{output.Name} の最初の10要素: {string.Join(", ", array.Take(10))}");
                        }
                        else
                        {
                            // AddLogEntry($"未対応の出力型: {output.Name} ({output.Value?.GetType().Name ?? "null"})");
                        }
                    }

                    // AddLogEntry("特徴量抽出が完了しました");
                    // 出力名は features, style_output, content_outputの3つ
                    return result;
                }
            }
            catch (OnnxRuntimeException ex)
            {
                AddLogEntry($"ONNX実行中にエラーが発生しました: {ex.Message}");
                throw;
            }
        }

        public async Task<DenseTensor<Float16>> PreprocessImage(BitmapImage image)
        {
            var tensor = new DenseTensor<Float16>(new[] { 1, 3, ImageSize, ImageSize });

            int sourceWidth = image.PixelWidth;
            int sourceHeight = image.PixelHeight;

            byte[] sourcePixels = new byte[4 * sourceWidth * sourceHeight];
            image.CopyPixels(sourcePixels, 4 * sourceWidth, 0);

            float xRatio = (float)sourceWidth / ImageSize;
            float yRatio = (float)sourceHeight / ImageSize;

            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    int sourceX = (int)(x * xRatio);
                    int sourceY = (int)(y * yRatio);
                    int sourceIndex = (sourceY * sourceWidth + sourceX) * 4;

                    tensor[0, 0, y, x] = (Float16)(sourcePixels[sourceIndex + 2] / 255f);     // R
                    tensor[0, 1, y, x] = (Float16)(sourcePixels[sourceIndex + 1] / 255f);     // G
                    tensor[0, 2, y, x] = (Float16)(sourcePixels[sourceIndex] / 255f);         // B
                }
            }

            return tensor;
        }

        public void Dispose()
        {
            _session?.Dispose();
            AddLogEntry("CSDモデルのリソースを解放しました");
        }
    }
}
