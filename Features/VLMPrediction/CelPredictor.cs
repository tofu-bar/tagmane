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
using System.Windows.Media.Imaging;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Net.Http.Headers;
using System.Windows.Media;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using tagmane.Features.ImageProcessing;
using System.Runtime.InteropServices;

namespace tagmane
{
    public class CelPredictor
    {
        private InferenceSession _model;
        private List<string> _tagNames;
        private List<int> _ratingIndexes;
        private List<int> _generalIndexes;
        private List<int> _characterIndexes;
        private List<int> _copyrightIndexes;
        private List<int> _artistIndexes;
        private List<int> _metaIndexes;
        private int _modelTargetSize;
        private const int MaxLogEntries = 20;

        private const string MODEL_FILENAME = "model.onnx";
        private const string LABEL_FILENAME = "tag_mapping.json";
        private const string MODEL_REPO = "cella110n/cl_tagger";
        private const string MODEL_SUBDIR = "cl_tagger_1_01";

        public ObservableCollection<string> VLMLogEntries { get; } = new ObservableCollection<string>();
        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        private void AddLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            LogUpdated?.Invoke(this, $"CelPredictor: {logMessage}");
        }

        private bool _isModelLoaded = false;

        // モデル読み込み時に入力名を保存
        private string _modelInputName;

        private readonly int[] _inputShape = new[] { 1, 3, 448, 448 }; // 固定サイズを定義

        // リサイズモードの列挙型を追加
        public enum ResizeMode
        {
            Lanczos3,
            Lanczos2,
            Lanczos,
            Bicubic
        }

        // 現在のリサイズモード（デフォルトはBicubic）
        private ResizeMode _currentResizeMode = ResizeMode.Bicubic;

        // リサイズモードを設定するためのプロパティ
        public ResizeMode CurrentResizeMode
        {
            get => _currentResizeMode;
            set
            {
                _currentResizeMode = value;
                AddLogEntry($"リサイズモードを変更: {_currentResizeMode}");
            }
        }

        // クラスレベルで除外パターンリストを定義
        private readonly List<string> _excludedMetaPatterns = new List<string> { "id", "commentary", "request" };

        public async Task LoadModel(string modelRepo, bool useGpu = true, string hfToken = null)
        {
            AddLogEntry($"リポジトリからモデルを読み込みます: {modelRepo}");
            
            // モデルファイルのパスを先に確認
            var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", modelRepo.Split('/').Last());
            var jsonPath = Path.Combine(modelDir, LABEL_FILENAME);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);
            
            bool needDownload = !File.Exists(jsonPath) || !File.Exists(modelPath);
            
            // Hugging Face認証コード（オープンソース版を使用するため不要）
            /*
            // ダウンロードが必要な場合のみトークンを要求
            if (needDownload && string.IsNullOrEmpty(hfToken))
            {
                var tokenInput = MessageBox.Show(
                    "このモデルはGatedリポジトリからダウンロードする必要があります。HuggingFaceのトークンを入力してください。",
                    "HuggingFaceトークン要求",
                    MessageBoxButton.OKCancel);

                if (tokenInput == MessageBoxResult.Cancel)
                {
                    AddLogEntry("ユーザーがトークン入力をキャンセルしました。");
                    return;
                }

                var tokenDialog = new TokenInputDialog();
                bool? result = tokenDialog.ShowDialog();

                if (result != true || string.IsNullOrEmpty(tokenDialog.Token))
                {
                    AddLogEntry("有効なトークンが提供されませんでした。");
                    return;
                }

                hfToken = tokenDialog.Token;
            }
            */

            // 既存のモデルを使用するか、新たにダウンロード
            if (needDownload)
            {
                (jsonPath, modelPath) = await DownloadModel(modelRepo, hfToken);
            }
            else
            {
                AddLogEntry("既存のモデルファイルを使用します。");
            }

            _tagNames = new List<string>();
            _ratingIndexes = new List<int>();
            _generalIndexes = new List<int>();
            _characterIndexes = new List<int>();
            _copyrightIndexes = new List<int>();
            _artistIndexes = new List<int>();
            _metaIndexes = new List<int>();

            LoadLabels(jsonPath);

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
                            // CUDAオプションをシンプルに設定
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
                    _model = new InferenceSession(modelPath, sessionOptions);
                    _modelInputName = _model.InputMetadata.First().Key;
                    
                    var inputMeta = _model.InputMetadata.First().Value;
                    AddLogEntry($"モデル入力名: {_modelInputName}");
                    AddLogEntry($"モデル入力型: {inputMeta.ElementType}");
                    AddLogEntry($"モデル入力形状: {string.Join(", ", inputMeta.Dimensions.ToArray())}");

                    // 入力形状の検証
                    if (inputMeta.Dimensions.Length != 4)
                    {
                        throw new Exception($"予期しない入力形状です: {string.Join(", ", inputMeta.Dimensions.ToArray())}");
                    }

                    // モデルの読み込みが完了した後に、ターゲットサイズを設定
                    _modelTargetSize = 448;
                    AddLogEntry($"ターゲットサイズ: {_modelTargetSize}");

                    _isModelLoaded = true;
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    AddLogEntry($"エラー（{retryCount}/{maxRetries}）: {ex.Message}");
                    
                    if (retryCount >= maxRetries)
                    {
                        AddLogEntry("モデルの読み込みに失敗しました。詳細なエラー: " + ex.ToString());
                        throw;
                    }
                    
                    await Task.Delay(1000);
                }
            }
        }

        private async Task<(string jsonPath, string modelPath)> DownloadModel(string repo, string hfToken)
        {
            var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", repo.Split('/').Last());
            Directory.CreateDirectory(modelDir);

            var jsonPath = Path.Combine(modelDir, LABEL_FILENAME);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);

            // ファイルが両方存在する場合は即座に返す
            if (File.Exists(jsonPath) && File.Exists(modelPath))
            {
                AddLogEntry("既存のモデルファイルを使用します。");
                return (jsonPath, modelPath);
            }

            // オープンソース版を使用するためトークンチェック不要
            /*
            // トークンのチェック
            if (string.IsNullOrEmpty(hfToken))
            {
                AddLogEntry("有効なトークンが提供されていません。ダウンロードをスキップします。");
                throw new InvalidOperationException("有効なHuggingFaceトークンが必要です。");
            }
            */

            using (var httpClient = new HttpClient())
            {
                // オープンソース版を使用するため認証ヘッダー不要
                // httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", hfToken);
                string baseUrl = $"https://huggingface.co/{repo}/resolve/main/{MODEL_SUBDIR}/";

                if (!File.Exists(jsonPath))
                {
                    AddLogEntry($"タグマッピングファイルをダウンロードしています: {LABEL_FILENAME}");
                    var progress = new Progress<double>(p => Application.Current.Dispatcher.Invoke(() => 
                        ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = p * 50));
                    await DownloadFileWithRetry(httpClient, baseUrl + LABEL_FILENAME, jsonPath, progress: progress);
                }
                
                if (!File.Exists(modelPath))
                {
                    AddLogEntry($"モデルファイルをダウンロードしています: {MODEL_FILENAME}");
                    var progress = new Progress<double>(p => Application.Current.Dispatcher.Invoke(() => 
                        ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 50 + p * 50));
                    await DownloadFileWithRetry(httpClient, baseUrl + MODEL_FILENAME, modelPath, progress: progress);
                }
            }

            Application.Current.Dispatcher.Invoke(() => ((MainWindow)Application.Current.MainWindow).ProgressBar.Value = 0);
            return (jsonPath, modelPath);
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
                    await Task.Delay(1000 * (i + 1));
                }
            }
        }

        private void LoadLabels(string jsonPath)
        {
            AddLogEntry($"タグマッピングを読み込んでいます: {jsonPath}");
            
            try
            {
                string jsonContent = File.ReadAllText(jsonPath);
                var tagMapping = JsonSerializer.Deserialize<Dictionary<string, TagInfo>>(jsonContent);
                
                // タグマッピングを処理
                int maxIndex = -1;
                foreach (var kvp in tagMapping)
                {
                    int index = int.Parse(kvp.Key);
                    maxIndex = Math.Max(maxIndex, index);
                }
                
                // _tagNames初期化（最大インデックス+1の長さ）
                _tagNames = new List<string>(new string[maxIndex + 1]);
                
                foreach (var kvp in tagMapping)
                {
                    int index = int.Parse(kvp.Key);
                    string tag = kvp.Value.Tag;
                    string category = kvp.Value.Category;
                    
                    _tagNames[index] = tag;
                    
                    // カテゴリ別インデックスの設定
                    switch (category.ToLower())
                    {
                        case "rating":
                            _ratingIndexes.Add(index);
                            break;
                        case "general":
                            _generalIndexes.Add(index);
                            break;
                        case "character":
                            _characterIndexes.Add(index);
                            break;
                        case "copyright":
                            _copyrightIndexes.Add(index);
                            break;
                        case "artist":
                            _artistIndexes.Add(index);
                            break;
                        case "meta":
                            _metaIndexes.Add(index);
                            break;
                    }
                }
                
                AddLogEntry($"タグ数: {_tagNames.Count}, Rating: {_ratingIndexes.Count}, General: {_generalIndexes.Count}, " +
                           $"Character: {_characterIndexes.Count}, Copyright: {_copyrightIndexes.Count}, " +
                           $"Artist: {_artistIndexes.Count}, Meta: {_metaIndexes.Count}");
            }
            catch (Exception ex)
            {
                AddLogEntry($"タグマッピングの読み込みエラー: {ex.Message}");
                throw;
            }
        }

        // BitmapImageをSystem.Drawing.Bitmapに変換
        private System.Drawing.Bitmap BitmapImageToBitmap(BitmapImage bitmapImage)
        {
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new PngBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapImage));
                enc.Save(outStream);
                outStream.Seek(0, SeekOrigin.Begin);
                return new System.Drawing.Bitmap(outStream);
            }
        }

        // System.Drawing.BitmapをBitmapImageに変換
        private BitmapImage BitmapToBitmapImage(System.Drawing.Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Png);
                memory.Position = 0;
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze(); // UIスレッド以外でも使用可能にする
                return bitmapImage;
            }
        }

        // BICUBICリサイズを行うメソッド
        private byte[] ResizeWithBicubic(byte[] sourcePixels, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            try
            {
                // 元の画像のWriteableBitmapを作成
                var sourceBitmap = new WriteableBitmap(sourceWidth, sourceHeight, 96, 96, PixelFormats.Bgra32, null);
                sourceBitmap.WritePixels(new Int32Rect(0, 0, sourceWidth, sourceHeight), sourcePixels, sourceWidth * 4, 0);

                // BitmapSourceを使用した高品質リサイズ
                var transformedBitmap = new TransformedBitmap(sourceBitmap, new ScaleTransform(
                    (double)targetWidth / sourceWidth,
                    (double)targetHeight / sourceHeight));

                // 高品質(BICUBIC)スケーリングを設定
                RenderOptions.SetBitmapScalingMode(transformedBitmap, BitmapScalingMode.HighQuality);

                // リサイズ後のピクセルデータを取得
                byte[] resizedPixels = new byte[targetWidth * targetHeight * 4];
                transformedBitmap.CopyPixels(resizedPixels, targetWidth * 4, 0);

                return resizedPixels;
            }
            catch (Exception ex)
            {
                AddLogEntry($"BICUBICリサイズエラー: {ex.Message}");
                throw;
            }
        }

        // 既存のLanczosResizerを使用したリサイズメソッド
        private byte[] ResizeWithLanczosResizer(byte[] sourcePixels, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            try
            {                
                // ソースピクセルからBitmapを作成
                using (var sourceBitmap = new System.Drawing.Bitmap(sourceWidth, sourceHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
                {
                    var bmpData = sourceBitmap.LockBits(
                        new System.Drawing.Rectangle(0, 0, sourceWidth, sourceHeight),
                        ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                    
                    Marshal.Copy(sourcePixels, 0, bmpData.Scan0, sourcePixels.Length);
                    sourceBitmap.UnlockBits(bmpData);
                    
                    // リサイズモードからLanczosパラメータを取得
                    string lanczosMode;
                    switch (_currentResizeMode)
                    {
                        case ResizeMode.Lanczos3:
                            lanczosMode = "Lanczos3";
                            break;
                        case ResizeMode.Lanczos2:
                            lanczosMode = "Lanczos2";
                            break;
                        case ResizeMode.Lanczos:
                            lanczosMode = "Lanczos"; // a=1の原法
                            break;
                        default:
                            lanczosMode = "Lanczos2"; // デフォルト
                            break;
                    }
                    AddLogEntry($"モード: {lanczosMode} でリサイズ: {sourceWidth}x{sourceHeight} -> {targetWidth}x{targetHeight}");
                    using (var resizedBitmap = LanczosResizer.ResizeImage(sourceBitmap, targetWidth, targetHeight, lanczosMode))
                    {
                        // リサイズ結果のピクセルデータを取得
                        var resultData = resizedBitmap.LockBits(
                            new System.Drawing.Rectangle(0, 0, targetWidth, targetHeight),
                            ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                        
                        byte[] resultPixels = new byte[targetWidth * targetHeight * 4];
                        Marshal.Copy(resultData.Scan0, resultPixels, 0, resultPixels.Length);
                        resizedBitmap.UnlockBits(resultData);
                        
                        return resultPixels;
                    }
                }
            }
            catch (Exception ex)
            {
                AddLogEntry($"Lanczosリサイズエラー: {ex.Message}");
                throw;
            }
        }

        public DenseTensor<float> PreprocessImage(BitmapImage image)
        {
            if (!_isModelLoaded)
                throw new InvalidOperationException("モデルが読み込まれていません");

            try
            {
                AddLogEntry($"前処理開始: 画像サイズ {image.PixelWidth}x{image.PixelHeight}, リサイズモード: {_currentResizeMode}");
                
                int width = image.PixelWidth;
                int height = image.PixelHeight;
                
                // 1. 正方形にパディング (Pythonの pil_pad_square と同等)
                int squareSize = Math.Max(width, height);
                int padX = (squareSize - width) / 2;
                int padY = (squareSize - height) / 2;
                
                AddLogEntry($"正方形パディング: {squareSize}x{squareSize} (パディング: X={padX}, Y={padY})");
                
                // 白背景の正方形画像を作成
                byte[] squarePixels = new byte[squareSize * squareSize * 4];
                for (int i = 0; i < squarePixels.Length; i += 4)
                {
                    squarePixels[i] = 255;     // B
                    squarePixels[i + 1] = 255; // G
                    squarePixels[i + 2] = 255; // R
                    squarePixels[i + 3] = 255; // A
                }
                
                // 元画像のピクセルを取得
                byte[] sourcePixels = new byte[width * height * 4];
                image.CopyPixels(sourcePixels, width * 4, 0);
                
                // 元画像をパディングした正方形画像に配置
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcIdx = (y * width + x) * 4;
                        int destIdx = ((y + padY) * squareSize + (x + padX)) * 4;
                        
                        squarePixels[destIdx] = sourcePixels[srcIdx];         // B
                        squarePixels[destIdx + 1] = sourcePixels[srcIdx + 1]; // G
                        squarePixels[destIdx + 2] = sourcePixels[srcIdx + 2]; // R
                        squarePixels[destIdx + 3] = sourcePixels[srcIdx + 3]; // A
                    }
                }
                
                // 2. 選択されたアルゴリズムでリサイズ
                byte[] resizedPixels;
                
                if (_currentResizeMode == ResizeMode.Bicubic)
                {
                    AddLogEntry($"BICUBICフィルタでリサイズ: {squareSize}x{squareSize} -> {_modelTargetSize}x{_modelTargetSize}");
                    resizedPixels = ResizeWithBicubic(
                        squarePixels, squareSize, squareSize, _modelTargetSize, _modelTargetSize);
                }
                else // Lanczos
                {
                    // 既存のLanczosResizerを使用
                    AddLogEntry($"Lanczosフィルタでリサイズ: {squareSize}x{squareSize} -> {_modelTargetSize}x{_modelTargetSize}");
                    resizedPixels = ResizeWithLanczosResizer(
                        squarePixels, squareSize, squareSize, _modelTargetSize, _modelTargetSize);
                }
                
                // 3. テンソルを準備（CHW形式）
                var tensor = new DenseTensor<float>(new[] { 1, 3, _modelTargetSize, _modelTargetSize });
                
                // 4. チャンネル処理と正規化（Python のコードと一致するように）
                for (int y = 0; y < _modelTargetSize; y++)
                {
                    for (int x = 0; x < _modelTargetSize; x++)
                    {
                        int pixelIndex = (y * _modelTargetSize + x) * 4;
                        
                        // BGRA -> RGB（Pythonコードに合わせる）
                        float b = resizedPixels[pixelIndex] / 255.0f;
                        float g = resizedPixels[pixelIndex + 1] / 255.0f;
                        float r = resizedPixels[pixelIndex + 2] / 255.0f;
                        
                        // Python と同様に正規化（mean=0.5, std=0.5）
                        tensor[0, 0, y, x] = (b - 0.5f) / 0.5f;  // B
                        tensor[0, 1, y, x] = (g - 0.5f) / 0.5f;  // G 
                        tensor[0, 2, y, x] = (r - 0.5f) / 0.5f;  // R
                    }
                }
                
                // テンソル統計をログ出力
                float minVal = float.MaxValue;
                float maxVal = float.MinValue;
                foreach (var val in tensor.Buffer.Span)
                {
                    minVal = Math.Min(minVal, val);
                    maxVal = Math.Max(maxVal, val);
                }
                AddLogEntry($"テンソル作成完了: 形状={string.Join(",", tensor.Dimensions.ToArray())}, 値範囲={minVal}～{maxVal}");
                
                return tensor;
            }
            catch (Exception ex)
            {
                AddLogEntry($"前処理エラー: {ex.Message}");
                AddLogEntry($"詳細: {ex.StackTrace}");
                throw;
            }
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            DenseTensor<float> inputTensor,
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (!_isModelLoaded)
            {
                AddLogEntry("モデルが読み込まれていません。");
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            try
            {
                AddLogEntry("推論を実行しています");
                AddLogEntry($"入力テンソル形状: {string.Join(", ", inputTensor.Dimensions.ToArray())}");
                
                // テンソルの値の範囲を確認
                float minVal = float.MaxValue;
                float maxVal = float.MinValue;
                foreach (var val in inputTensor.Buffer.Span)
                {
                    minVal = Math.Min(minVal, val);
                    maxVal = Math.Max(maxVal, val);
                }
                AddLogEntry($"入力テンソルの値の範囲: {minVal} to {maxVal}");

                var inputs = new List<NamedOnnxValue> { 
                    NamedOnnxValue.CreateFromTensor(_modelInputName, inputTensor) 
                };

                using (var outputs = _model.Run(inputs))
                {
                    var predictions = outputs.First().AsEnumerable<float>().ToArray();
                    AddLogEntry($"出力テンソルサイズ: {predictions.Length}");

                    // シグモイド関数で確率に変換
                    var probs = new float[predictions.Length];
                    for (int i = 0; i < predictions.Length; i++)
                    {
                        probs[i] = 1.0f / (1.0f + (float)Math.Exp(-predictions[i]));
                    }
                    
                    // 戻り値用の辞書を準備
                    var rating = "";
                    var generalTags = new Dictionary<string, float>();
                    var characterTags = new Dictionary<string, float>();
                    var otherTags = new Dictionary<string, float>();
                    
                    // レーティングの処理（最大値を選択）
                    if (_ratingIndexes.Count > 0)
                    {
                        float maxProb = 0f;
                        int maxIndex = -1;
                        
                        foreach (var idx in _ratingIndexes)
                        {
                            if (probs[idx] > maxProb)
                            {
                                maxProb = probs[idx];
                                maxIndex = idx;
                            }
                        }
                        
                        if (maxIndex >= 0)
                        {
                            rating = _tagNames[maxIndex].Replace("_", " "); // アンダースコアをスペースに置換
                            AddLogEntry($"レーティングタグ: {rating} ({maxProb:F3})");
                        }
                    }
                    
                    // 一般タグの処理
                    foreach (var idx in _generalIndexes)
                    {
                        if (probs[idx] >= generalThresh)
                        {
                            generalTags[_tagNames[idx].Replace("_", " ")] = probs[idx]; // アンダースコアをスペースに置換
                        }
                    }
                    AddLogEntry($"一般タグ: {generalTags.Count}個（閾値: {generalThresh}）");
                    
                    // キャラクタータグの処理
                    foreach (var idx in _characterIndexes)
                    {
                        if (probs[idx] >= characterThresh)
                        {
                            characterTags[_tagNames[idx].Replace("_", " ")] = probs[idx]; // アンダースコアをスペースに置換
                        }
                    }
                    AddLogEntry($"キャラクタータグ: {characterTags.Count}個（閾値: {characterThresh}）");
                    
                    // その他のタグ（著作権とメタ）の処理
                    foreach (var idx in _copyrightIndexes.Concat(_metaIndexes))
                    {
                        if (probs[idx] >= generalThresh)
                        {
                            otherTags[_tagNames[idx].Replace("_", " ")] = probs[idx]; // アンダースコアをスペースに置換
                        }
                    }
                    AddLogEntry($"その他のタグ: {otherTags.Count}個（閾値: {generalThresh}）");
                    
                    // メタタグのフィルタリング
                    var filteredMetaTags = new List<(string, float)>();
                    foreach (var kvp in otherTags)
                    {
                        string tag = kvp.Key;
                        float confidence = kvp.Value;
                        
                        bool shouldExclude = _excludedMetaPatterns.Any(pattern => 
                            tag.ToLower().Contains(pattern.ToLower()));
                        
                        if (!shouldExclude)
                        {
                            filteredMetaTags.Add((tag, confidence));
                            AddLogEntry($"  メタタグ: {tag}: {confidence:F3}");
                        }
                        else
                        {
                            // フィルタリングされたタグもログに記録（ただし[FILTERED]マーク付き）
                            AddLogEntry($"  [FILTERED] メタタグ: {tag}: {confidence:F3}");
                        }
                    }

                    // 元の辞書をクリアして、フィルタリング済みの項目を追加
                    otherTags.Clear();
                    foreach (var (tag, confidence) in filteredMetaTags)
                    {
                        otherTags[tag] = confidence;
                    }
                    
                    // すべてのタグを結合してソート
                    var allTags = new Dictionary<string, float>();
                    
                    // レーティングを追加
                    if (!string.IsNullOrEmpty(rating))
                    {
                        allTags[rating] = 1.0f;
                    }
                    
                    // キャラクタータグを追加
                    foreach (var tag in characterTags)
                    {
                        allTags[tag.Key] = tag.Value;
                    }
                    
                    // 一般タグとその他のタグを追加
                    foreach (var tag in generalTags.Concat(otherTags))
                    {
                        allTags[tag.Key] = tag.Value;
                    }
                    
                    // すべてのタグを確率順にソート
                    var sortedGeneralStrings = string.Join(", ", allTags.OrderByDescending(x => x.Value).Select(x => x.Key));
                    
                    AddLogEntry($"ソート済みタグ（全カテゴリ）: {sortedGeneralStrings}");
                    
                    // レーティング辞書を作成
                    var ratingDict = new Dictionary<string, float>();
                    if (!string.IsNullOrEmpty(rating))
                    {
                        ratingDict.Add(rating, 1.0f);
                    }
                    
                    // 一般タグとその他のタグを結合
                    var combinedTags = generalTags.Concat(otherTags).ToDictionary(x => x.Key, x => x.Value);
                    
                    // WDPredictorと同様の順序で返す
                    return (sortedGeneralStrings, ratingDict, characterTags, combinedTags);
                }
            }
            catch (Exception ex)
            {
                AddLogEntry($"推論中にエラーが発生しました: {ex.Message}");
                if (ex.InnerException != null)
                {
                    AddLogEntry($"内部エラー: {ex.InnerException.Message}");
                }
                AddLogEntry($"スタックトレース: {ex.StackTrace}");
                throw;
            }
        }

        private List<string> PredictInternal(BitmapImage image, float threshold = 0.45f)
        {
            if (!_isModelLoaded)
                throw new InvalidOperationException("モデルが読み込まれていません");

            AddLogEntry("画像の前処理を開始します");
            var inputTensor = PreprocessImage(image);
            
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            
            AddLogEntry("推論を実行しています");
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            
            try
            {
                results = _model.Run(inputs);
            }
            catch (Exception ex)
            {
                AddLogEntry($"推論エラー: {ex.Message}");
                throw;
            }
            
            AddLogEntry("推論結果を処理しています");
            
            var output = results.First().AsEnumerable<float>().ToArray();
            
            // シグモイド関数で確率に変換
            var probs = new float[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                probs[i] = 1.0f / (1.0f + (float)Math.Exp(-output[i]));
            }
            
            var tagList = new List<string>();
            
            // レーティングの処理（最大値を選択）
            if (_ratingIndexes.Count > 0)
            {
                float maxProb = 0f;
                int maxIndex = -1;
                
                foreach (var idx in _ratingIndexes)
                {
                    if (probs[idx] > maxProb)
                    {
                        maxProb = probs[idx];
                        maxIndex = idx;
                    }
                }
                
                if (maxIndex >= 0)
                {
                    tagList.Add(_tagNames[maxIndex]);
                    AddLogEntry($"レーティングタグ: {_tagNames[maxIndex]} ({maxProb:F3})");
                }
            }
            
            // 一般タグの処理
            int generalCount = 0;
            foreach (var idx in _generalIndexes)
            {
                if (probs[idx] >= threshold)
                {
                    tagList.Add(_tagNames[idx]);
                    generalCount++;
                }
            }
            AddLogEntry($"一般タグ: {generalCount}個（閾値: {threshold}）");
            
            // キャラクタータグの処理
            int charCount = 0;
            foreach (var idx in _characterIndexes)
            {
                if (probs[idx] >= threshold)
                {
                    tagList.Add(_tagNames[idx]);
                    charCount++;
                }
            }
            AddLogEntry($"キャラクタータグ: {charCount}個（閾値: {threshold}）");
            
            // コピーライトタグの処理（GeneralTagと同様に扱う）
            int copyrightCount = 0;
            foreach (var idx in _copyrightIndexes)
            {
                if (probs[idx] >= threshold)
                {
                    tagList.Add(_tagNames[idx]);
                    copyrightCount++;
                }
            }
            AddLogEntry($"著作権タグ: {copyrightCount}個（閾値: {threshold}）");
            
            // アーティストタグは返さない（要件に基づく）
            
            // メタタグの処理
            int metaCount = 0;
            foreach (var idx in _metaIndexes)
            {
                if (probs[idx] >= threshold)
                {
                    tagList.Add(_tagNames[idx]);
                    metaCount++;
                }
            }
            AddLogEntry($"メタタグ: {metaCount}個（閾値: {threshold}）");
            
            AddLogEntry($"合計: {tagList.Count}個のタグが見つかりました");
            return tagList;
        }

        // タグ情報を格納するクラス - JSON属性を修正
        private class TagInfo
        {
            [JsonPropertyName("tag")]
            public string Tag { get; set; }
            
            [JsonPropertyName("category")]
            public string Category { get; set; }
        }

        // Disposeメソッドを追加
        public void Dispose()
        {
            if (_model != null)
            {
                _model.Dispose();
                _model = null;
            }
            _isModelLoaded = false;
            AddLogEntry("モデルをアンロードしました");
        }
    }

    // トークン入力用ダイアログ
    public class TokenInputDialog : Window
    {
        public string Token { get; private set; }
        
        public TokenInputDialog()
        {
            Title = "HuggingFaceトークン入力";
            Width = 400;
            Height = 150;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            
            var grid = new System.Windows.Controls.Grid();
            grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
            grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
            grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
            
            var label = new System.Windows.Controls.Label
            {
                Content = "モデルへのアクセスには、HuggingFaceのトークンが必要です。\nトークンを入力してください：",
                Margin = new Thickness(10, 10, 10, 0)
            };
            System.Windows.Controls.Grid.SetRow(label, 0);
            
            var textBox = new System.Windows.Controls.TextBox
            {
                Margin = new Thickness(10, 5, 10, 5)
            };
            System.Windows.Controls.Grid.SetRow(textBox, 1);
            
            var buttonPanel = new System.Windows.Controls.StackPanel
            {
                Orientation = System.Windows.Controls.Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(10, 5, 10, 10)
            };
            System.Windows.Controls.Grid.SetRow(buttonPanel, 2);
            
            var okButton = new System.Windows.Controls.Button
            {
                Content = "OK",
                Width = 75,
                Margin = new Thickness(0, 0, 10, 0),
                IsDefault = true
            };
            okButton.Click += (s, e) => {
                Token = textBox.Text;
                DialogResult = true;
            };
            
            var cancelButton = new System.Windows.Controls.Button
            {
                Content = "キャンセル",
                Width = 75,
                IsCancel = true
            };
            
            buttonPanel.Children.Add(okButton);
            buttonPanel.Children.Add(cancelButton);
            
            grid.Children.Add(label);
            grid.Children.Add(textBox);
            grid.Children.Add(buttonPanel);
            
            Content = grid;
        }
    }
} 