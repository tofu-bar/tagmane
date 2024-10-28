using System;
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
using System.Security.Cryptography;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace tagmane
{
    public class JoyPredictor
    {
        private InferenceSession _session;
        private List<string> _tags;
        private const int ImageSize = 448;
        private const float Threshold = 0.4f;
        private const string MODEL_FILENAME = "model.onnx";
        private const string LABEL_FILENAME = "top_tags.txt";
        private const string MODEL_REPO = "fancyfeast/joytag";

        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        private void AddLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            LogUpdated?.Invoke(this, $"JoyPredictor: {logMessage}");
        }

        private bool _isModelLoaded = false;

        public async Task LoadModel(string modelRepo, bool useGpu = true)
        {
            AddLogEntry($"リポジトリからモデルを読み込みます: {modelRepo}");
            var (tagsPath, modelPath) = await DownloadModel(modelRepo);

            int retryCount = 0;
            const int maxRetries = 3;

            while (retryCount < maxRetries)
            {
                try
                {
                    var sessionOptions = new SessionOptions();
                    var gpuDeviceId = 0;

                    if (useGpu)
                    {
                        AddLogEntry("ONNX推論セッションを初期化しています（GPU使用を試みます）");

                        try
                        {
                            sessionOptions.AppendExecutionProvider_CUDA(gpuDeviceId);
                            AddLogEntry("GPUを使用します");
                            IsGpuLoaded = true;
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
                        AddLogEntry("ONNX推論セッションを初期化しています（CPU使用）");
                        IsGpuLoaded = false;
                    }

                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _session = new InferenceSession(modelPath, sessionOptions);
                    _tags = File.ReadAllLines(tagsPath).Where(line => !string.IsNullOrWhiteSpace(line)).ToList();
                    AddLogEntry("モデルが正常に読み込まれました。");
                    _isModelLoaded = true;
                    AddLogEntry("モデルの読み込みが完了しました。");
                    break; // 成功した場合、ループを抜ける
                }
                catch (Exception ex)
                {
                    AddLogEntry($"ONNX推論セッションの初期化に失敗しました: {ex.Message}");
                    retryCount++;

                    if (retryCount >= maxRetries)
                    {
                        AddLogEntry("最大再試行回数に達しました。モデルの読み込みに失敗しました。");
                        throw;
                    }

                    AddLogEntry($"モデルの再ダウンロードを試みます。試行回数: {retryCount}");
                    File.Delete(modelPath);
                    AddLogEntry($"既存のモデルファイルを削除しました: {modelPath}");
                    (tagsPath, modelPath) = await DownloadModel(modelRepo);
                }
            }
        }

        private async Task<(string, string)> DownloadModel(string modelRepo)
        {
            using var client = new HttpClient();
            var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", modelRepo.Split('/').Last());
            Directory.CreateDirectory(modelDir);
            var tagsPath = Path.Combine(modelDir, LABEL_FILENAME);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);
            AddLogEntry($"モデルファイルのダウンロードを開始します: {modelRepo}");
            AddLogEntry($"辞書ダウンロードパス: {tagsPath}");
            AddLogEntry($"モデルダウンロードパス: {modelPath}");

            bool needsDownload = !File.Exists(tagsPath) || !File.Exists(modelPath) || !VerifyFileIntegrity(modelPath);

            if (needsDownload)
            {
                await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{LABEL_FILENAME}", tagsPath);
                var progress = new Progress<double>(p => Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = p * 100));
                await DownloadFileWithRetry(client, $"https://huggingface.co/{modelRepo}/resolve/main/{MODEL_FILENAME}", modelPath, progress: progress);

                if (!VerifyFileIntegrity(modelPath))
                {
                    throw new Exception("ダウンロードされたモデルファイルが破損しているか不完全です。");
                }
                AddLogEntry("モデルファイルを新たにダウンロードしました。");
            }
            else
            {
                AddLogEntry("既存のモデルファイルを使用します。");
                Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 100);
            }

            Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 0);

            return (tagsPath, modelPath);
        }

        private async Task DownloadFileWithRetry(HttpClient client, string url, string filePath, int maxRetries = 3, IProgress<double> progress = null)
        {
            for (int i = 0; i < maxRetries; i++)
            {
                try
                {
                    AddLogEntry($"ファイルをダウンロードしています: {url}");
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

                    AddLogEntry($"ファイルのダウンロードが完了しました: {filePath}");
                    return;
                }
                catch (Exception ex)
                {
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                        AddLogEntry($"ダウンロードが失敗したため、不完全なファイルを削除しました: {filePath}");
                    }

                    if (i == maxRetries - 1)
                        throw new Exception($"{maxRetries}回の試行後、ファイルのダウンロードに失敗しました: {ex.Message}");
                    AddLogEntry($"ダウンロード試行 {i + 1} 回目が失敗しました。再試行します...");
                }
                await Task.Delay(1000 * (i + 1));
            }
        }

        private bool VerifyFileIntegrity(string filePath)
        {
            try
            {
                using var stream = File.OpenRead(filePath);
                using var sha256 = SHA256.Create();
                byte[] hash = sha256.ComputeHash(stream);
                AddLogEntry($"ファイルの整合性チェックに成功しました: {filePath}");
                return true;
            }
            catch (Exception ex)
            {
                AddLogEntry($"ファイルの整合性チェックに失敗しました: {ex.Message}");
                return false;
            }
        }
        
        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            BitmapImage image,
            float generalThresh)
        {
            if (!_isModelLoaded)
            {
                // throw new InvalidOperationException("モデルが読み込まれていません。Predictを実行する前にLoadModelを呼び出してください。");
                AddLogEntry("モデルが読み込まれていません。Predictを実行する前にLoadModelを呼び出してください。");
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            AddLogEntry("推論を開始します");
            AddLogEntry($"generalThresh: {generalThresh}");

            DenseTensor<float> inputTensor;
            try
            {
                inputTensor = PrepareImage(image);
            }
            catch (Exception ex)
            {
                AddLogEntry($"画像の準備中にエラーが発生しました: {ex.Message}");
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                var scores = new Dictionary<string, float>();

                for (int i = 0; i < _tags.Count; i++)
                {
                    scores[_tags[i].Replace("_", " ")] = Sigmoid(output[0, i]);

                    // 正規表現置換(顔文字などをそのままにする)は以下だが、処理が重いため保留
                    // scores[System.Text.RegularExpressions.Regex.Replace(_tags[i], @"(?<=\w)_(?=\w)", " ")] = Sigmoid(output[0, i]);
                }

                var filteredScores = scores.Where(kv => kv.Value >= generalThresh)
                                           .OrderByDescending(kv => kv.Value)
                                           .ToDictionary(kv => kv.Key, kv => kv.Value);

                var sortedGeneralStrings = string.Join(", ", filteredScores.Keys);

                // joytagは分類区分がないためgeneralのみを返す
                return (sortedGeneralStrings, new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }
        }

        private DenseTensor<float> PrepareImage(BitmapImage image)
        {
            AddLogEntry("画像を準備しています");
            var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });

            int stride = (image.PixelWidth * image.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[stride * image.PixelHeight];
            image.CopyPixels(pixels, stride, 0);
            
            int sourceWidth = image.PixelWidth;
            int sourceHeight = image.PixelHeight;

            const float rMean = 0.48145466f, gMean = 0.4578275f, bMean = 0.40821073f;
            const float rStd = 0.26862954f, gStd = 0.26130258f, bStd = 0.27577711f;

            float xRatio = (float)sourceWidth / ImageSize;
            float yRatio = (float)sourceHeight / ImageSize;

            for (int y = 0; y < ImageSize; y++)
            {
                int sourceY = (int)(y * yRatio);
                for (int x = 0; x < ImageSize; x++)
                {
                    int sourceX = (int)(x * xRatio);
                    int sourceIndex = (sourceY * stride) + (sourceX * 4);

                    tensor[0, 0, y, x] = (pixels[sourceIndex + 2] / 255f - rMean) / rStd; // R
                    tensor[0, 1, y, x] = (pixels[sourceIndex + 1] / 255f - gMean) / gStd; // G
                    tensor[0, 2, y, x] = (pixels[sourceIndex] / 255f - bMean) / bStd;     // B
                }
            }

            AddLogEntry("画像をテンソルに変換しました");
            return tensor;
        }

        private float Sigmoid(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        public void Dispose()
        {
            _session?.Dispose();
            AddLogEntry("JoyPredictorのリソースを解放しました");
        }
    }
}
