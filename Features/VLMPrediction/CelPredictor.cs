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

        private const string MODEL_FILENAME = "checkpoint_epoch_1_merged.onnx";
        private const string LABEL_FILENAME = "tag_mapping.json";
        private const string MODEL_REPO = "celstk/wd-eva02-lora-onnx";
        private const string MODEL_SUBDIR = "merged_model_0325_1ep_onnx";

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

        public async Task LoadModel(string modelRepo, bool useGpu = true, string hfToken = null)
        {
            AddLogEntry($"リポジトリからモデルを読み込みます: {modelRepo}");
            
            // モデルファイルのパスを先に確認
            var modelDir = Path.Combine(Path.GetTempPath(), "tagmane", modelRepo.Split('/').Last());
            var jsonPath = Path.Combine(modelDir, LABEL_FILENAME);
            var modelPath = Path.Combine(modelDir, MODEL_FILENAME);
            
            bool needDownload = !File.Exists(jsonPath) || !File.Exists(modelPath);
            
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

            // トークンのチェック
            if (string.IsNullOrEmpty(hfToken))
            {
                AddLogEntry("有効なトークンが提供されていません。ダウンロードをスキップします。");
                throw new InvalidOperationException("有効なHuggingFaceトークンが必要です。");
            }

            using (var httpClient = new HttpClient())
            {
                httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", hfToken);
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

        public DenseTensor<float> PreprocessImage(BitmapImage image)
        {
            if (!_isModelLoaded)
                throw new InvalidOperationException("モデルが読み込まれていません");

            int width = image.PixelWidth;
            int height = image.PixelHeight;
            int stride = width * 4; // BGRA形式（4バイト/ピクセル）
            byte[] pixelData = new byte[height * stride];
            
            image.CopyPixels(pixelData, stride, 0); 
            
            // モデル入力サイズの正方形にパディング
            int squareSize = Math.Max(width, height);
            byte[] squarePixelData = new byte[squareSize * squareSize * 4];
            
            // 初期化（白背景）
            for (int i = 0; i < squareSize * squareSize * 4; i += 4)
            {
                squarePixelData[i] = 255;     // B
                squarePixelData[i + 1] = 255; // G
                squarePixelData[i + 2] = 255; // R
                squarePixelData[i + 3] = 255; // A
            }
            
            // 元画像を中央に配置
            int offsetX = (squareSize - width) / 2;
            int offsetY = (squareSize - height) / 2;
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcIdx = (y * width + x) * 4;
                    int destIdx = ((y + offsetY) * squareSize + (x + offsetX)) * 4;
                    
                    squarePixelData[destIdx] = pixelData[srcIdx];         // B
                    squarePixelData[destIdx + 1] = pixelData[srcIdx + 1]; // G
                    squarePixelData[destIdx + 2] = pixelData[srcIdx + 2]; // R
                    squarePixelData[destIdx + 3] = pixelData[srcIdx + 3]; // A
                }
            }
            
            // リサイズのためのテンソルを準備（リサイズ後の次元: NCHW形式）
            var tensor = new DenseTensor<float>(new[] { 1, 3, _modelTargetSize, _modelTargetSize });
            
            // リサイズとカラー変換（BGRA -> RGB、正規化）
            float[] resizeFactors = { (float)squareSize / _modelTargetSize, (float)squareSize / _modelTargetSize };
            
            for (int i = 0; i < _modelTargetSize; i++)
            {
                for (int j = 0; j < _modelTargetSize; j++)
                {
                    // 最近傍法でリサイズ
                    int origY = (int)(i * resizeFactors[0]);
                    int origX = (int)(j * resizeFactors[1]);
                    
                    int idx = (origY * squareSize + origX) * 4;
                    
                    // BGR -> RGB変換と正規化（0-1範囲）
                    float b = squarePixelData[idx] / 255.0f;
                    float g = squarePixelData[idx + 1] / 255.0f;
                    float r = squarePixelData[idx + 2] / 255.0f;
                    
                    // 正規化（平均0.5、標準偏差0.5）
                    tensor[0, 0, i, j] = (r - 0.5f) / 0.5f;
                    tensor[0, 1, i, j] = (g - 0.5f) / 0.5f;
                    tensor[0, 2, i, j] = (b - 0.5f) / 0.5f;
                }
            }
            
            return tensor;
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

        public DenseTensor<float> PrepareTensor(BitmapImage image)
        {
            try
            {
                if (image == null || image.PixelWidth <= 0 || image.PixelHeight <= 0)
                {
                    AddLogEntry($"無効な画像サイズ: {(image == null ? "null" : $"{image.PixelWidth}x{image.PixelHeight}")}");
                    throw new ArgumentException("有効な画像が提供されていません");
                }

                int targetSize = _modelTargetSize > 0 ? _modelTargetSize : 448; 
                
                // 1. まずWDPredictorと同様のNHWC形式で処理（実績のある方法）
                var tensor = new DenseTensor<float>(new[] { 1, targetSize, targetSize, 3 });
                AddLogEntry($"テンソルを作成: 1, {targetSize}, {targetSize}, 3（NHWC形式）");

                // ソース画像サイズを取得
                int sourceWidth = image.PixelWidth;
                int sourceHeight = image.PixelHeight;
                AddLogEntry($"入力画像サイズ: {sourceWidth}x{sourceHeight}");

                // ソース画像のピクセルデータを取得
                byte[] sourcePixels = new byte[4 * sourceWidth * sourceHeight];
                image.CopyPixels(sourcePixels, 4 * sourceWidth, 0);

                // リサイズ比率を計算
                float xRatio = (float)sourceWidth / targetSize;
                float yRatio = (float)sourceHeight / targetSize;

                // WDPredictorと同様の処理でテンソルを作成
                for (int y = 0; y < targetSize; y++)
                {
                    for (int x = 0; x < targetSize; x++)
                    {
                        int sourceX = Math.Min((int)(x * xRatio), sourceWidth - 1);
                        int sourceY = Math.Min((int)(y * yRatio), sourceHeight - 1);
                        int sourceIndex = (sourceY * sourceWidth + sourceX) * 4;

                        if (sourceIndex + 2 >= sourcePixels.Length)
                        {
                            continue;
                        }

                        // ★重要: WDPredictorとは逆の順序で格納（RGBとBGRの切り替え）
                        // WDPredictor: tensor[0, y, x, 2] = R, tensor[0, y, x, 0] = B
                        // ここでは: tensor[0, y, x, 0] = R, tensor[0, y, x, 2] = B
                        tensor[0, y, x, 0] = sourcePixels[sourceIndex + 2];  // R
                        tensor[0, y, x, 1] = sourcePixels[sourceIndex + 1];  // G
                        tensor[0, y, x, 2] = sourcePixels[sourceIndex];      // B
                    }
                }

                // 2. 次にNCHW形式に変換（Pythonコードでは最終的にはこの形式）
                var reshapedTensor = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });
                
                // NHWC -> NCHW変換
                for (int h = 0; h < targetSize; h++)
                {
                    for (int w = 0; w < targetSize; w++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            reshapedTensor[0, c, h, w] = tensor[0, h, w, c];
                        }
                    }
                }

                // 3. 値を正規化（-1～1の範囲に）
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < targetSize; h++)
                    {
                        for (int w = 0; w < targetSize; w++)
                        {
                            // 0-255を0-1にスケーリングしてから、mean=0.5, std=0.5で正規化
                            reshapedTensor[0, c, h, w] = (reshapedTensor[0, c, h, w] / 255.0f - 0.5f) / 0.5f;
                        }
                    }
                }

                // テンソルの統計情報をログに記録
                float minVal = float.MaxValue;
                float maxVal = float.MinValue;
                foreach (var val in reshapedTensor.Buffer.Span)
                {
                    minVal = Math.Min(minVal, val);
                    maxVal = Math.Max(maxVal, val);
                }
                
                AddLogEntry($"テンソル統計: 範囲={minVal}～{maxVal}");
                AddLogEntry($"サンプル値[0,0,0,0]={reshapedTensor[0, 0, 0, 0]}, [0,1,0,0]={reshapedTensor[0, 1, 0, 0]}, [0,2,0,0]={reshapedTensor[0, 2, 0, 0]}");
                
                return reshapedTensor;
            }
            catch (Exception ex)
            {
                AddLogEntry($"テンソル準備エラー: {ex.Message}");
                AddLogEntry($"詳細: {ex.StackTrace}");
                throw;
            }
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