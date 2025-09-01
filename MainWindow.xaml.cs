using Microsoft.WindowsAPICodePack.Dialogs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Path = System.IO.Path;
using System.Text.RegularExpressions;
using System.Diagnostics;
using UMAP;
using R3;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;  // Float16のため
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks.Dataflow;
using tagmane.Features.TagShuffle;  // 追加：名前空間のusing
using tagmane.Features.ImageProcessing;
using tagmane.Subwindows;
using System.Drawing; // System.Drawing.Imageのために追加
using DrawingPoint = System.Drawing.Point;
using DrawingImage = System.Drawing.Image;
using DrawingColor = System.Drawing.Color;
using DrawingBrushes = System.Drawing.Brushes;

using WindowsPoint = System.Windows.Point;
using WindowsImage = System.Windows.Controls.Image;
using WindowsColor = System.Windows.Media.Color;
using WindowsBrushes = System.Windows.Media.Brushes;

namespace tagmane
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string _currentVersion = "1.0.7";
        private CancellationTokenSource _logCancellationTokenSource;
        private RingBuffer<string> _logQueue = new RingBuffer<string>(100);
        private RingBuffer<string> _debugLogQueue = new RingBuffer<string>(100);
        private RingBuffer<string> _uiErrorLogQueue = new RingBuffer<string>(50);
        private RingBuffer<string> _vlmLogQueue = new RingBuffer<string>(100);
        private RingBuffer<string> _vlmErrorLogQueue = new RingBuffer<string>(50);
        private RingBuffer<string> _pipelineLogQueue = new RingBuffer<string>(100);
        private RingBuffer<string> _pythonLogQueue = new RingBuffer<string>(500);
        
        // 各ログボックスの表示済み項目数を追跡
        private int _mainLogDisplayedCount = 0;
        private int _debugLogDisplayedCount = 0;
        private int _vlmLogDisplayedCount = 0;
        private int _pipelineLogDisplayedCount = 0;
        private int _pythonLogDisplayedCount = 0;
        
        private int _logUpdateIntervalMs = 500;
        private int _vlmUpdateIntervalMs = 1000;

        private bool _isInitializeSuccess = false;
        private FileExplorer _fileExplorer;
        private string _selectedFolderPath;

        private List<ImageInfo> _originalImageInfos;
        private List<ImageInfo> _clusteredImageInfos; // クラスタリング結果の画像リスト
        private List<ImageInfo> _imageInfos;

        // タグの管理
        private Dictionary<string, int> _allTags;
        private bool _isUpdatingSelection = false;

        private HashSet<string> _selectedTags = new HashSet<string>();
        private HashSet<string> _currentImageTags = new HashSet<string>();

        private Stack<ITagAction> _undoStack = new Stack<ITagAction>();
        private Stack<ITagAction> _redoStack = new Stack<ITagAction>();
        
        // 検索最適化用キャッシュ
        private List<string> _cachedDictionaryTags = null;
        private System.Threading.Timer _searchDelayTimer = null;
        private readonly object _searchLock = new object();
        
        // 高度なフィルタリング
        private ObservableCollection<FilterCondition> _filterConditions = new ObservableCollection<FilterCondition>();
        public ObservableCollection<FilterCondition> FilterConditions => _filterConditions;
        
        // タグ追加履歴管理
        private LinkedList<string> _recentAddedTags = new LinkedList<string>();
        private Dictionary<string, int> _tagFrequency = new Dictionary<string, int>();
        private const int MaxRecentTags = 10;
        private bool _isShowingTagHistory = false;
        
        // キーボードナビゲーション用の専用選択状態
        private int _keyboardNavigationIndex = -1;
        
        // JSONファイルから読み込んだタグのカウント（Danbooru頻度）
        private Dictionary<string, int> _jsonTagCounts = new Dictionary<string, int>();

        // 検索結果用のタグ情報クラス
        public class SearchTagInfo
        {
            public string Tag { get; set; } = "";
            public int Count { get; set; } = 0;
            
            public override string ToString() => Tag;
        }

        private ObservableCollection<ActionLogItem> _actionLogItems;
        private const int MaxLogEntries = 20; // 100から20に変更

        private VLMPredictor _vlmPredictor;
        private bool _isLoadingVLMModel = false;
        private CancellationTokenSource _cts;
        
        // 連続キャプション生成用
        private bool _isContinuousCaptionGeneration = false;
        private CancellationTokenSource _captionCancellationTokenSource;
        private int _currentCaptionIndex = 0;
        
        // 永続Pythonセッション用
        private Process _persistentPythonProcess;
        private StreamWriter _pythonInput;
        private StreamReader _pythonOutput;
        private StreamReader _pythonError;
        private List<(string Name, double GeneralThreshold)> _vlmModels = new List<(string, double)> 
        {
            ("SmilingWolf/wd-eva02-large-tagger-v3", 0.50),
            ("SmilingWolf/wd-vit-large-tagger-v3", 0.25),
            ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", 0.35),
            ("SmilingWolf/wd-vit-tagger-v3", 0.25),
            ("SmilingWolf/wd-swinv2-tagger-v3", 0.25),
            ("SmilingWolf/wd-convnext-tagger-v3", 0.25),
            ("SmilingWolf/wd-v1-4-moat-tagger-v2", 0.35),
            ("SmilingWolf/wd-v1-4-convnext-tagger-v2", 0.35),
            ("SmilingWolf/wd-v1-4-vit-tagger-v2", 0.35),
            ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", 0.35),
            ("fancyfeast/joytag", 0.5),
            ("cella110n/cl_tagger", 0.55)
        };
        private const double DefaultCharacterThreshold = 0.85;

        private static readonly string[] DefaultCategoryFiles = {
            "tagcount/Rating.json",
            "tagcount/Quality.json",
            "tagcount/Model.json",
            "tagcount/Meta.json",
            "tagcount/Character.json",
            "tagcount/Artist.json",
            "tagcount/Copyright.json",
            "tagcount/General.json"
        };
        private static readonly string[] CustomCategoryFiles = {
            "tagcount_custom/PersonCounts.json",
            "tagcount_custom/Face.json"
        };
        private Dictionary<string, TagCategory> _tagCategories;
        private Dictionary<string, TagCategory> _defaultTagCategories;
        private Dictionary<string, TagCategory> _customTagCategories;
        private Dictionary<string, TagCategory> _userAddedTagCategories;
        private ObservableCollection<CategoryItem> _tagCategoryNames;
        private bool _useCustomCategories = true;
        private List<string> _prefixOrder;
        private List<string> _suffixOrder;

        // インターフェースを追加
        private interface ITagAction
        {
            void DoAction();
            void UndoAction();
            string Description { get; }
        }
        private class TagPositionInfo
        {
            public string Tag { get; set; }
            public int Position { get; set; }
        }
        private class TagAction : ITagAction
        {
            public ImageInfo Image { get; set; }
            public TagPositionInfo TagInfo { get; set; }
            public bool IsAdd { get; set; }
            public Action DoAction { get; set; }
            public Action UndoAction { get; set; }
            public string Description { get; set; }

            void ITagAction.DoAction() => DoAction();
            void ITagAction.UndoAction() => UndoAction();
        }
        private class TagGroupAction : ITagAction
        {
            public ImageInfo Image { get; set; }
            public List<TagPositionInfo> TagInfos { get; set; }
            public bool IsAdd { get; set; }
            public Action DoAction { get; set; }
            public Action UndoAction { get; set; }
            public string Description { get; set; }

            void ITagAction.DoAction() => DoAction();
            void ITagAction.UndoAction() => UndoAction();
        }
        private class TagCategory
        {
            [JsonPropertyName("0")]
            public Dictionary<string, int> Tags { get; set; }
        }
        private class CategoryItem
        {
            public string Name { get; set; }
            public string OrderType { get; set; } // "Prefix", "Suffix", or ""
        }

        public ObservableCollection<string> Tags { get; set; }    
        private enum ClusterMode { Off, CSD }
        private ClusterMode _currentClusterMode = ClusterMode.Off;

        // クラスタグループは可変のため、enumではなくlistで管理
        private List<string> _clusterGroups = new List<string> { "ALL" };
        private int _currentClusterGroup = 0;

        private enum ClusterAnnotationMode { Default, Tags_And, Tags_Or }
        private ClusterAnnotationMode _currentClusterAnnotationMode = ClusterAnnotationMode.Default;

        private string _webpDllPath;
        private WebPHandler _webPHandler;

        // 非同期処理のフラグ
        private bool _isAsyncProcessing = false;

        // 処理速度計
        public string ProcessingSpeed
        {
            get { return _processingSpeed; }
            set
            {
                _processingSpeed = value;
                Dispatcher.Invoke(() => ProcessingSpeedTextBlock.Text = value);
            }
        }

        private string _processingSpeed = "";

        private int _clusterCount;
        private int[] _clusterAssignments;
        private float[][] _clusterEmbeddings;
        private float[][] _umapEmbeddings;

        private int _loadImgProcessedImagesCount;
        private int _predictProcessedImagesCount;
        private int _totalProcessedImagesCount;

        private WindowsPoint? _startPoint;  // WPF用のPoint
        private bool _isSelecting;
        private bool _isInSelectionMode;
        private bool _isDragging;

        public MainWindow()
        {
            try
            {
                _isInitializeSuccess = false;

                InitializeComponent();

                if (!CheckLicenseAgreement())
                {
                    Close();
                    return;
                }

                _fileExplorer = new FileExplorer();
                _allTags = new Dictionary<string, int>();
                _actionLogItems = new ObservableCollection<ActionLogItem>();
                ActionListView.ItemsSource = _actionLogItems;
                
                // デバッグ用のメッセージを追加
                MessageBox.Show("MainWindowが初期化されました。");
                
                // ウィンドウを表示
                this.Show();

                InitializeVLMPredictor();

                //各種設定を読み込む
                LoadSettings();

                Tags = new ObservableCollection<string>();
                TagListView.ItemsSource = Tags;

                _tagCategories = new Dictionary<string, TagCategory>();
                _defaultTagCategories = new Dictionary<string, TagCategory>();
                _customTagCategories = new Dictionary<string, TagCategory>();
                _tagCategoryNames = new ObservableCollection<CategoryItem>();
                _userAddedTagCategories = new Dictionary<string, TagCategory>();
                TagCategoryListView.ItemsSource = _tagCategoryNames;

                _prefixOrder = new List<string>();
                _suffixOrder = new List<string>();

                LoadTagCategories();
                
                // デフォルトのカテゴリ順序設定はコメントアウト（ユーザーが個別に設定する）
                // SetDefaultCategoryOrder();

                _isInitializeSuccess = true;
                
                // フィルタ条件の初期化
                FilterConditionsListView.DataContext = this;

                StartLogProcessing(); // ログ処理を開始

                // リサイズモードの設定
                ResizeModeComboBox.Items.Clear();
                ResizeModeComboBox.Items.Add(new ComboBoxItem { Content = "縮小のみ", IsSelected = true });
                ResizeModeComboBox.Items.Add(new ComboBoxItem { Content = "拡大のみ" });
                ResizeModeComboBox.Items.Add(new ComboBoxItem { Content = "リサイズ" });

                // リサンプルモードの設定
                ResampleModeComboBox.Items.Clear();
                ResampleModeComboBox.Items.Add(new ComboBoxItem { Content = "Lanczos2", IsSelected = true });
                ResampleModeComboBox.Items.Add(new ComboBoxItem { Content = "Lanczos3" });
                ResampleModeComboBox.Items.Add(new ComboBoxItem { Content = "Bilinear" });

                // 出力形式の設定
                OutputFormatComboBox.Items.Clear();
                OutputFormatComboBox.Items.Add(new ComboBoxItem { Content = "WebP", IsSelected = true });
                OutputFormatComboBox.Items.Add(new ComboBoxItem { Content = "PNG" });
                OutputFormatComboBox.Items.Add(new ComboBoxItem { Content = "JPEG" });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"MainWindowの初期化中にエラーが発生しました: {ex.Message}\n\nStackTrace: {ex.StackTrace}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private bool CheckLicenseAgreement()
        {
            string agreedVersion = Properties.Settings.Default.AgreedLicenseVersion;

            if (string.IsNullOrEmpty(agreedVersion) || agreedVersion != _currentVersion)
            {
                var licenseWindow = new LicenseAgreementWindow(_currentVersion);
                if (licenseWindow.ShowDialog() == true)
                {
                    Properties.Settings.Default.AgreedLicenseVersion = _currentVersion;
                    Properties.Settings.Default.Save();
                    return true;
                }
                else
                {
                    return false;
                }
            }
            return true;
        }

        /*
        基本モジュール
        */

        // キーイベントハンドラ（ショートカットキー）
        protected override void OnKeyDown(KeyEventArgs e)
        {
            base.OnKeyDown(e);
            if (e.Handled) return;

            if (Keyboard.Modifiers == ModifierKeys.Control)
            {
                switch (e.Key)
                {
                    case Key.O:
                        SelectFolder_Click(null, null); // フォルダを選択
                        e.Handled = true;
                        break;
                    case Key.S:
                        SaveTagsButton_Click(null, null); // 現在のタグ状態を保存
                        e.Handled = true;
                        break;
                    case Key.M:
                        MoveDirectoryButton_Click(null, null); // フィルタ中の画像のフォルダを移動
                        e.Handled = true;
                        break;
                    case Key.Z:
                        UndoButton_Click(null, null); // 元に戻す
                        e.Handled = true;
                        break;
                    case Key.Y:
                        RedoButton_Click(null, null); // やり直し
                        e.Handled = true;
                        break;
                    case Key.T:
                        AddTagButton_Click(null, null); // タグを追加
                        e.Handled = true;
                        break;
                    case Key.D:
                        RemoveTagButton_Click(null, null); // タグを削除
                        e.Handled = true;
                        break;
                    case Key.Up:
                        MoveTopButton_Click(null, null); // 選択しているタグをtopに移動
                        e.Handled = true;
                        break;
                    case Key.Down:
                        MoveBottomButton_Click(null, null); // 選択しているタグをbottomに移動
                        e.Handled = true;
                        break;
                    case Key.E:
                        DeselectTagButton_Click(null, null); // 選択されているタグを解除
                        e.Handled = true;
                        break;
                    case Key.R:
                        SortByCategoryButton_Click(null, null); // カテゴリ順に並び替え
                        e.Handled = true;
                        break;
                    case Key.P:
                        VLMPredictButton_Click(null, null); // VLMでタグを作成
                        e.Handled = true;
                        break;
                    case Key.G:
                        if ((Keyboard.Modifiers & ModifierKeys.Shift) == ModifierKeys.Shift)
                        {
                            GenerateAllCaptionsButton_Click(null, null); // 全画像にキャプションを生成
                        }
                        else
                        {
                            GenerateCaptionButton_Click(null, null); // GLM-4.1Vでキャプションを生成
                        }
                        e.Handled = true;
                        break;
                    case Key.F:
                        // フィルタ機能は新しい高度フィルタに置き換えられました
                        e.Handled = true;
                        break;
                    case Key.H:
                        ReplaceTagButton_Click(null, null); // タグの置換
                        e.Handled = true;
                        break;
                    case Key.Enter:
                        if (Keyboard.Modifiers == ModifierKeys.Control)
                        {
                            AddTagAndMoveNext(); // タグ追加後、次画像へ
                        }
                        else if (Keyboard.Modifiers == (ModifierKeys.Control | ModifierKeys.Shift))
                        {
                            AddTagAndMovePrevious(); // タグ追加後、前画像へ
                        }
                        else
                        {
                            AddTextboxinputButton_Click(null, null); // 個別タグに追加
                        }
                        e.Handled = true;
                        break;
                    case Key.OemQuestion: // "/" key
                        SearchTextBox.Focus(); // 検索欄にフォーカス
                        e.Handled = true;
                        break;
                    // Ctrl+0-9 で履歴から追加
                    case Key.D0:
                    case Key.NumPad0:
                        AddRecentTag(0);
                        e.Handled = true;
                        break;
                    case Key.D1:
                    case Key.NumPad1:
                        AddRecentTag(1);
                        e.Handled = true;
                        break;
                    case Key.D2:
                    case Key.NumPad2:
                        AddRecentTag(2);
                        e.Handled = true;
                        break;
                    case Key.D3:
                    case Key.NumPad3:
                        AddRecentTag(3);
                        e.Handled = true;
                        break;
                    case Key.D4:
                    case Key.NumPad4:
                        AddRecentTag(4);
                        e.Handled = true;
                        break;
                    case Key.D5:
                    case Key.NumPad5:
                        AddRecentTag(5);
                        e.Handled = true;
                        break;
                    case Key.D6:
                    case Key.NumPad6:
                        AddRecentTag(6);
                        e.Handled = true;
                        break;
                    case Key.D7:
                    case Key.NumPad7:
                        AddRecentTag(7);
                        e.Handled = true;
                        break;
                    case Key.D8:
                    case Key.NumPad8:
                        AddRecentTag(8);
                        e.Handled = true;
                        break;
                    case Key.D9:
                    case Key.NumPad9:
                        AddRecentTag(9);
                        e.Handled = true;
                        break;
                }
            }
            else if (Keyboard.Modifiers == (ModifierKeys.Control | ModifierKeys.Shift))
            {
                switch (e.Key)
                {
                    // 危ないので今はコメントアウト
                    // case Key.Delete:
                    //     DeleteFilteredImageAndTags_Click(null, null); // フィルタ中の画像とタグを削除
                    //     e.Handled = true;
                    //     break;
                    case Key.T:
                        AddAllTagsButton_Click(null, null); // 選択しているタグを全画像に追加
                        e.Handled = true;
                        break;
                    case Key.D:
                        RemoveAllTagsButton_Click(null, null); // 選択しているタグを全画像から削除
                        e.Handled = true;
                        break;
                    case Key.Up:
                        MoveTopAllButton_Click(null, null); // 選択しているタグを全画像でtopに移動
                        e.Handled = true;
                        break;
                    case Key.Down:
                        MoveBottomAllButton_Click(null, null); // 選択しているタグを全画像でbottomに移動
                        e.Handled = true;
                        break;
                    case Key.E:
                        DeselectAllTagsButton_Click(null, null); // 選択されているタグをすべて解除
                        e.Handled = true;
                        break;
                    case Key.R:
                        SortByCategoryAllButton_Click(null, null); // すべての画像をカテゴリ順に並び替え
                        e.Handled = true;
                        break;
                    case Key.P:
                        VLMPredictAllButton_Click(null, null); // VLMですべての画像にタグを作成
                        e.Handled = true;
                        break;
                    case Key.F:
                        AddFolderNameButton_Click(null, null); // フォルダ名をタグに追加
                        e.Handled = true;
                        break;
                    case Key.Enter:
                        AddAllTextboxinputButton_Click(null, null); // 全タグに追加
                        e.Handled = true;
                        break;
                }
            }
            else
            {
                switch (e.Key)
                {
                    case Key.Escape:
                        CancelButton_Click(null, null); // キャンセル
                        e.Handled = true;
                        break;
                    case Key.PageUp:
                        MoveToPreviousImage();
                        e.Handled = true;
                        break;
                    case Key.PageDown:
                        MoveToNextImage();
                        e.Handled = true;
                        break;
                    case Key.Home:
                        MoveToFirstImage();
                        e.Handled = true;
                        break;
                    case Key.End:
                        MoveToLastImage();
                        e.Handled = true;
                        break;
                    case Key.Delete:
                        if (ImageListBox.IsKeyboardFocusWithin)  // IsFocusedの代わりにIsKeyboardFocusWithinを使用
                        {
                            DeleteSelectedImageAndTags_Click(null, null);
                            e.Handled = true;
                        }
                        break;
                    case Key.T:
                        if (ImageListBox.IsKeyboardFocusWithin)  // IsFocusedの代わりにIsKeyboardFocusWithinを使用
                        {
                            AddTagButton_Click(null, null); // タグを追加
                            e.Handled = true;
                        }
                        break;
                    case Key.D:
                        if (ImageListBox.IsKeyboardFocusWithin)  // IsFocusedの代わりにIsKeyboardFocusWithinを使用
                        {
                            RemoveTagButton_Click(null, null); // タグを削除
                            e.Handled = true;
                        }
                        break;
                }
            }
        }

        private void LoadSettings()
        {
            _webpDllPath = Properties.Settings.Default.WebPDllPath;
            WebPDllPathTextBox.Text = _webpDllPath;
            _webPHandler = new WebPHandler(_webpDllPath);

            // VLMモデルの設定を読み込む
            string savedModel = Properties.Settings.Default.SelectedVLMModel;
            VLMModelComboBox.ItemsSource = _vlmModels.Select(m => m.Name);
            if (!string.IsNullOrEmpty(savedModel) && _vlmModels.Any(m => m.Name == savedModel)) 
            { 
                VLMModelComboBox.SelectedIndex = _vlmModels.FindIndex(m => m.Name == savedModel);
            }
            else { VLMModelComboBox.SelectedIndex = 0; }
            UpdateThresholds(_vlmModels[VLMModelComboBox.SelectedIndex].GeneralThreshold, DefaultCharacterThreshold);

            AddMainLogEntry("設定を復元しました。");
        }

        private void SaveSettings()
        {
            Properties.Settings.Default.WebPDllPath = _webpDllPath;
            
            // 選択されたVLMモデルを保存
            if (VLMModelComboBox.SelectedItem is string selectedModel) { Properties.Settings.Default.SelectedVLMModel = selectedModel; }
            
            Properties.Settings.Default.Save();
            AddMainLogEntry("設定を保存しました。");
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            // 永続Pythonセッションのクリーンアップ
            StopPersistentPythonSession();
            
            SaveSettings();
            _logCancellationTokenSource.Cancel(); // ログ処理を停止
            base.OnClosing(e);
        }

        private ImageSource LoadImage(string imagePath)
        {
            try
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(imagePath);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;  // 重要: これによりファイルのストリームが即座に閉じられる
                bitmap.EndInit();
                return bitmap;
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"画像の読み込みに失敗: {ex.Message}");
                return null;
            }
        }

        private string FormatTag(string tag)
        {
            var format = (TagFormatComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();
            switch (format)
            {
                case "aaaa \\(bbbb\\)":
                    return tag.Replace("(", "\\(").Replace(")", "\\)");
                case "aaaa_(bbbb)":
                    return tag.Replace(" ", "_");
                default:
                    return tag;
            }
        }

        private string ProcessTag(string tag)
        {
            // 先頭と末尾のスペースを削除
            tag = tag.Trim();
            // アンダースコアをスペースに置換
            tag = tag.Replace('_', ' ');
            // エスケープされたカッコを戻す
            tag = tag.Replace("\\(", "(").Replace("\\)", ")");
            return tag;
        }

        private TagGroupAction CreateAddTagsAction(ImageInfo imageInfo, List<string> newTags)
        {
            return new TagGroupAction
            {
                Image = imageInfo,
                TagInfos = newTags.Select(tag => new TagPositionInfo { Tag = tag, Position = imageInfo.Tags.Count }).ToList(),
                IsAdd = true,
                DoAction = () =>
                {
                    foreach (var tag in newTags)
                    {
                        imageInfo.Tags.Add(tag);
                    }
                    AddMainLogEntry($"{imageInfo.ImagePath}に{newTags.Count}個のタグを追加しました");
                },
                UndoAction = () =>
                {
                    foreach (var tag in newTags)
                    {
                        imageInfo.Tags.Remove(tag);
                    }
                    AddMainLogEntry($"{imageInfo.ImagePath}から{newTags.Count}個のタグの追加を取り消しました");
                },
                Description = $"{imageInfo.ImagePath}に{newTags.Count}個のタグを追加"
            };
        }

        private void UpdateProgressBar(double progress)
        {
            Dispatcher.Invoke(() =>
            {
                // プログレスバーの更新処理
                ProgressBar.Value = progress * 100;
            });
        }

        // 画像リストの更新 (すべての表示内容を更新する)
        private void UpdateUIAfterImageInfosChange()
        {
            if (_imageInfos == null) { return; }
            UpdateImageList();
            UpdateImageCountDisplay();
            if (_allTags != null)
            {
                UpdateUIAfterTagsChange();
            }
        }

        private void UpdateUIAfterTagsChange()
        {
            if (_allTags == null) { return; }
            // ロックが取れなかった場合は、そのままreturnする。
            // Taskが並列で動いて更新している場合のみにreturnされるので、
            // 単一の操作時にはロック取得は何ら影響を与えない。
            // TODO ここのロックがない場合に、UIが固まることがある。理由の調査
            if (Monitor.TryEnter(_allTags, 0))
            {
                try
                {
                    UpdateCurrentTags();
                    UpdateAllTags();
                    UpdateTagListView();
                    UpdateAllTagsListView();
                    UpdateSelectedTagsListBox();
                    UpdateSearchedTagsListView();
                    UpdateButtonStates();
                }
                finally
                {
                    Monitor.Exit(_allTags);
                }
            } else {
                _uiErrorLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} ロック失敗 @UpdateUIAfterTagsChange");
            }
        }

        // 選択された画像の更新 (選択が変化したときのみ)
        private void UpdateUIAfterSelectionChange(bool updateAllTagSelection = true)
        {
            if (_imageInfos == null || _allTags == null) { return; }
            
            // TODO ここのロックがない場合に、UIが固まることがある。理由の調査
            if (Monitor.TryEnter(_allTags, 0))
            {
                try
                {
                    UpdateCurrentTags();
                    UpdateTagListView();
                    UpdateAllTagsListView(updateTagSelection: updateAllTagSelection);
                    // UpdateSelectedTagsListBox();
                    // UpdateSearchedTagsListView();
                }
                finally
                {
                    Monitor.Exit(_allTags);
                }
            } else {
                _uiErrorLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} ロック失敗 @UpdateUIAfterSelectionChange");
            }
        }

        // タグの選択が変更されたときの更新
        private void UpdateUIAfterTagSelectionChange()
        {
            if (_imageInfos == null || _allTags == null) { return; }
            UpdateTagListView();
            UpdateAllTagsListView();
            UpdateSelectedTagsListBox();
            UpdateSearchedTagsListView();
            if (_umapEmbeddings != null && _clusterAssignments != null) { VisualizeClusteringResult(_umapEmbeddings, _clusterAssignments); }
        }

        // ボタンの状態を更新
        private void UpdateButtonStates()
        {
            UndoButton.IsEnabled = _undoStack.Count > 0;
            RedoButton.IsEnabled = _redoStack.Count > 0;
        }

        /// <summary>
        /// TextBoxのログを制限サイズ内に保つヘルパーメソッド
        /// </summary>
        private void TrimLogTextBox(System.Windows.Controls.TextBox textBox, bool wasAtBottom, int maxSize = 102400)
        {
            if (textBox.Text.Length > maxSize)
            {
                // 古いログを削除（先頭から半分を削除）
                int removeLength = textBox.Text.Length - (maxSize / 2);
                // 改行位置を探して行単位で削除
                int newlineIndex = textBox.Text.IndexOf(Environment.NewLine, removeLength);
                if (newlineIndex > 0)
                {
                    textBox.Text = textBox.Text.Substring(newlineIndex + Environment.NewLine.Length);
                    // トリミング後も最下部にいた場合はスクロール
                    if (wasAtBottom)
                    {
                        textBox.ScrollToEnd();
                    }
                }
            }
        }

        private void StartLogProcessing()
        {
            _logCancellationTokenSource = new CancellationTokenSource();
            Task.Run(async () => {
                var logObservable = Observable.Interval(TimeSpan.FromMilliseconds(_logUpdateIntervalMs)).ToAsyncEnumerable();
                await foreach (var _ in logObservable)
                {
                    var newLogEntries = _logQueue.GetNewItems();
                    Dispatcher.Invoke(() => {
                        if (newLogEntries.Count > 0)
                        {
                            // スクロール位置が最下部付近かどうかを確認
                            var atBottom = MainLogTextBox.VerticalOffset >= MainLogTextBox.ExtentHeight - MainLogTextBox.ViewportHeight - 10;
                            
                            // 新しいログエントリを追加
                            foreach (var entry in newLogEntries)
                            {
                                if (MainLogTextBox.Text.Length > 0)
                                {
                                    MainLogTextBox.AppendText(Environment.NewLine);
                                }
                                MainLogTextBox.AppendText(entry);
                            }
                            
                            // ログサイズを制限
                            TrimLogTextBox(MainLogTextBox, atBottom);
                            
                            // 以前最下部にいた場合は自動スクロール
                            if (atBottom)
                            {
                                MainLogTextBox.ScrollToEnd();
                            }
                        }
                    });

                    // TODO: ログ出力タブを追加
                    // var uilogEntries = _uiErrorLogQueue.GetRecentItems();
                    // Dispatcher.Invoke(() => MainLogTextBox.Text = string.Join(Environment.NewLine, uilogEntries));

                    var newDebugLogEntries = _debugLogQueue.GetNewItems();
                    Dispatcher.Invoke(() => {
                        if (newDebugLogEntries.Count > 0)
                        {
                            var atBottom = DebugLogTextBox.VerticalOffset >= DebugLogTextBox.ExtentHeight - DebugLogTextBox.ViewportHeight - 10;
                            
                            foreach (var entry in newDebugLogEntries)
                            {
                                if (DebugLogTextBox.Text.Length > 0)
                                {
                                    DebugLogTextBox.AppendText(Environment.NewLine);
                                }
                                DebugLogTextBox.AppendText(entry);
                            }
                            
                            // ログサイズを制限
                            TrimLogTextBox(DebugLogTextBox, atBottom);
                            
                            if (atBottom)
                            {
                                DebugLogTextBox.ScrollToEnd();
                            }
                        }
                    });

                    var newVlmLogEntries = _vlmLogQueue.GetNewItems();
                    Dispatcher.Invoke(() => {
                        if (newVlmLogEntries.Count > 0)
                        {
                            var atBottom = VLMLogTextBox.VerticalOffset >= VLMLogTextBox.ExtentHeight - VLMLogTextBox.ViewportHeight - 10;
                            
                            foreach (var entry in newVlmLogEntries)
                            {
                                if (VLMLogTextBox.Text.Length > 0)
                                {
                                    VLMLogTextBox.AppendText(Environment.NewLine);
                                }
                                VLMLogTextBox.AppendText(entry);
                            }
                            
                            // ログサイズを制限
                            TrimLogTextBox(VLMLogTextBox, atBottom);
                            
                            if (atBottom)
                            {
                                VLMLogTextBox.ScrollToEnd();
                            }
                        }
                    });

                    var newPipelineLogEntries = _pipelineLogQueue.GetNewItems();
                    Dispatcher.Invoke(() => {
                        if (newPipelineLogEntries.Count > 0)
                        {
                            var atBottom = PipelineLogTextBox.VerticalOffset >= PipelineLogTextBox.ExtentHeight - PipelineLogTextBox.ViewportHeight - 10;
                            
                            foreach (var entry in newPipelineLogEntries)
                            {
                                if (PipelineLogTextBox.Text.Length > 0)
                                {
                                    PipelineLogTextBox.AppendText(Environment.NewLine);
                                }
                                PipelineLogTextBox.AppendText(entry);
                            }
                            
                            // ログサイズを制限
                            TrimLogTextBox(PipelineLogTextBox, atBottom);
                            
                            if (atBottom)
                            {
                                PipelineLogTextBox.ScrollToEnd();
                            }
                        }
                    });

                    var newPythonLogEntries = _pythonLogQueue.GetNewItems();
                    Dispatcher.Invoke(() => {
                        if (newPythonLogEntries.Count > 0)
                        {
                            var atBottom = PythonLogTextBox.VerticalOffset >= PythonLogTextBox.ExtentHeight - PythonLogTextBox.ViewportHeight - 10;
                            
                            foreach (var entry in newPythonLogEntries)
                            {
                                if (PythonLogTextBox.Text.Length > 0)
                                {
                                    PythonLogTextBox.AppendText(Environment.NewLine);
                                }
                                PythonLogTextBox.AppendText(entry);
                            }
                            
                            // ログサイズを制限
                            TrimLogTextBox(PythonLogTextBox, atBottom);
                            
                            if (atBottom)
                            {
                                PythonLogTextBox.ScrollToEnd();
                            }
                        }
                    });

                    if (_logCancellationTokenSource.IsCancellationRequested) break;
                }
            });
        }

        // デバッグログを追加するメソッド
        private void AddDebugLogEntry(string message)
        {
            _debugLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} - {message}");
        }

        public void AddMainLogEntry(string message)
        {
            _logQueue.Enqueue($"{DateTime.Now:HH:mm:ss} - {message}");
        }

        private void AddPythonLogEntry(string message)
        {
            _pythonLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} - {message}");
        }

        // アクションログを追加するメソッド
        private void AddActionLogItem(string actionType, string description)
        {
            if (_actionLogItems == null) { return; }

            _actionLogItems.Insert(0, new ActionLogItem { ActionType = actionType, Description = description });
            while (_actionLogItems.Count > MaxLogEntries)
            {
                _actionLogItems.RemoveAt(_actionLogItems.Count - 1);
            }
        }

        // ActionLogItemクラスを追加
        public class ActionLogItem
        {
            public string ActionType { get; set; }
            public string Description { get; set; }
        }
        
        /*
        ここまで基本モジュール
        */

        // 左ペイン: フォルダ選択と画像リスト表示
        private void SelectFolder_Click(object sender, RoutedEventArgs e)
        {
            SelectFolder();
        }

        private void SelectFolder()
        {
            var dialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "フォルダを選択してください"
            };

            if (dialog.ShowDialog() == CommonFileDialogResult.Ok)
            {
                _selectedFolderPath = dialog.FileName;

                _originalImageInfos = _fileExplorer.GetImageInfos(dialog.FileName);
                _clusteredImageInfos = null;
                _imageInfos = new List<ImageInfo>(_originalImageInfos);

                _userAddedTagCategories = null;

                _currentClusterMode = ClusterMode.Off;
                _clusterCount = 0;
                _clusterGroups = new List<string> { "ALL" };
                _clusterAssignments = null;
                _clusterEmbeddings = null;
                _umapEmbeddings = null;
                UpdateClusterGroupComboBox(0);

                // 古いフィルタ機能は削除されました
                
                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();
                
                UpdateUIAfterImageInfosChange();
                
                AddMainLogEntry($"{_imageInfos.Count}個の画像が見つかりました。");
                AddMainLogEntry($"フォルダを選択しました: {dialog.FileName}");
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");
                AddMainLogEntry("UserAddedカテゴリをクリアしました。");
            }
        }

        // タグの保存
        private void SaveTagsButton_Click(object sender, RoutedEventArgs e)
        {
            SaveAllTags();
        }

        // すべての画像のタグを保存
        private void SaveAllTags()
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("すべての画像のタグを保存しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("タグの保存がキャンセルされました。");
                    return;
                }
            }
            foreach (var imageInfo in _imageInfos)
            {
                SaveTagsToFile(imageInfo);
                imageInfo.AssociatedText = string.Join(", ", imageInfo.Tags);
            }
            MessageBox.Show("すべての画像のタグを保存しました。", "保存完了", MessageBoxButton.OK, MessageBoxImage.Information);
            
            // 中央ペインの更新
            if (ImageListBox.SelectedItem is ImageInfo selectedImage)
            {
                AssociatedText.Text = selectedImage.AssociatedText;
            }
        }

        private void MoveDirectoryButton_Click(object sender, RoutedEventArgs e)
        {
            MoveDirectory();
        }

        private void MoveDirectory()
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("移動対象の画像がありません。");
                return;
            }

            if (string.IsNullOrEmpty(_selectedFolderPath))
            {
                AddMainLogEntry("元のフォルダが選択されていません。");
                return;
            }

            // 移動先フォルダを選択するダイアログを表示
            var dialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "移動先フォルダを選択してください",
                InitialDirectory = _selectedFolderPath // 開いているフォルダを初期ディレクトリに設定
            };

            if (dialog.ShowDialog() == CommonFileDialogResult.Ok)
            {
                // 選択されたパスが元のフォルダのサブディレクトリかチェック
                if (!dialog.FileName.StartsWith(_selectedFolderPath))
                {
                    MessageBox.Show("選択されたフォルダは現在開いているフォルダの中にある必要があります。", 
                        "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                if (ConfirmCheckBox.IsChecked == true)
                {
                    var result = MessageBox.Show(
                        $"{_imageInfos.Count}個の画像とタグを移動しますか？\n移動先: {dialog.FileName}", 
                        "確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
                    if (result != MessageBoxResult.Yes)
                    {
                        AddMainLogEntry("移動がキャンセルされました。");
                        return;
                    }
                }

                try
                {
                    int successCount = 0;
                    int failureCount = 0;
                    var failedFiles = new List<string>();

                    foreach (var imageInfo in _imageInfos.ToList()) // ToList()で複製を作成
                    {
                        try
                        {
                            string fileName = Path.GetFileName(imageInfo.ImagePath);
                            string newImagePath = Path.Combine(dialog.FileName, fileName);
                            string textTagFilePath = Path.ChangeExtension(imageInfo.ImagePath, ".txt");
                            string jsonTagFilePath = Path.ChangeExtension(imageInfo.ImagePath, ".json");
                            string newTextTagFilePath = Path.ChangeExtension(newImagePath, ".txt");
                            string newJsonTagFilePath = Path.ChangeExtension(newImagePath, ".json");

                            // 画像を移動
                            File.Move(imageInfo.ImagePath, newImagePath);

                            // タグファイルが存在する場合は移動
                            if (File.Exists(textTagFilePath))
                            {
                                File.Move(textTagFilePath, newTextTagFilePath);
                            }
                            else if (File.Exists(jsonTagFilePath))
                            {
                                File.Move(jsonTagFilePath, newJsonTagFilePath);
                            }
                            else // タグファイルが存在しない場合は作成
                            {
                                // まず古いパスで保存
                                SaveTagsToFile(imageInfo);
                                
                                // 作成されたファイルを新しい場所に移動
                                string createdTxtPath = Path.ChangeExtension(imageInfo.ImagePath, ".txt");
                                string createdJsonPath = Path.ChangeExtension(imageInfo.ImagePath, ".json");
                                
                                if (File.Exists(createdTxtPath))
                                {
                                    File.Move(createdTxtPath, newTextTagFilePath);
                                }
                                if (File.Exists(createdJsonPath))
                                {
                                    File.Move(createdJsonPath, newJsonTagFilePath);
                                }
                            }

                            // パスを更新
                            imageInfo.ImagePath = newImagePath;
                            successCount++;
                        }
                        catch (Exception ex)
                        {
                            failureCount++;
                            failedFiles.Add(Path.GetFileName(imageInfo.ImagePath));
                            AddMainLogEntry($"ファイルの移動中にエラーが発生: {ex.Message}");
                        }
                    }

                    // 結果を表示
                    string resultMessage = $"{successCount}個のファイルを移動しました。";
                    if (failureCount > 0)
                    {
                        resultMessage += $"\n{failureCount}個のファイルの移動に失敗しました:";
                        resultMessage += $"\n{string.Join("\n", failedFiles)}";
                        MessageBox.Show(resultMessage, "移動結果", MessageBoxButton.OK, 
                            failureCount > 0 ? MessageBoxImage.Warning : MessageBoxImage.Information);
                    }
                    AddMainLogEntry(resultMessage);

                    // UIを更新
                    UpdateUIAfterImageInfosChange();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"移動処理中にエラーが発生しました: {ex.Message}", 
                        "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                    AddMainLogEntry($"移動処理中にエラーが発生: {ex.Message}");
                }
            }
        }

        // 画像のタグとキャプションをファイルに保存
        private void SaveTagsToFile(ImageInfo imageInfo)
        {
            var formattedTags = imageInfo.Tags.Select(FormatTag);
            string tagString = string.Join(", ", formattedTags);
            
            bool saveTxt = SaveTxtFormatCheckBox.IsChecked ?? false;
            bool saveJson = SaveJsonFormatCheckBox.IsChecked ?? false;
            
            string textFilePath = System.IO.Path.ChangeExtension(imageInfo.ImagePath, ".txt");
            string jsonFilePath = System.IO.Path.ChangeExtension(imageInfo.ImagePath, ".json");
            
            bool txtExists = File.Exists(textFilePath);
            bool jsonExists = File.Exists(jsonFilePath);
            
            // チェックボックスの設定に基づいて保存処理を実行
            if (saveTxt)
            {
                SaveToTxtFile(textFilePath, tagString);
            }
            
            if (saveJson)
            {
                if (jsonExists)
                {
                    UpdateJsonFile(jsonFilePath, tagString, imageInfo.Caption);
                }
                else
                {
                    CreateJsonFile(jsonFilePath, tagString, imageInfo.Caption);
                }
            }
            
            // どちらもチェックされていない場合は、既存のファイル形式に保存
            if (!saveTxt && !saveJson)
            {
                if (jsonExists)
                {
                    UpdateJsonFile(jsonFilePath, tagString, imageInfo.Caption);
                }
                else if (txtExists)
                {
                    SaveToTxtFile(textFilePath, tagString);
                }
                else
                {
                    // どちらも存在しない場合はtxtファイルとして保存
                    SaveToTxtFile(textFilePath, tagString);
                }
            }
        }
        
        private void SaveToTxtFile(string filePath, string tagString)
        {
            try
            {
                File.WriteAllText(filePath, tagString);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"txtファイルの保存中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"txtファイルの保存に失敗: {Path.GetFileName(filePath)} - {ex.Message}");
            }
        }
        
        private void UpdateJsonFile(string filePath, string tagString, string caption = "")
        {
            try
            {
                var jsonContent = File.ReadAllText(filePath);
                using var document = JsonDocument.Parse(jsonContent);
                var root = document.RootElement;
                
                var options = new JsonWriterOptions { Indented = true };
                using var stream = new MemoryStream();
                using var writer = new Utf8JsonWriter(stream, options);
                
                writer.WriteStartObject();
                foreach (var property in root.EnumerateObject())
                {
                    if (property.Name == "tags")
                    {
                        writer.WriteString("tags", tagString);
                    }
                    else if (property.Name == "caption")
                    {
                        writer.WriteString("caption", caption);
                    }
                    else
                    {
                        property.WriteTo(writer);
                    }
                }
                
                // captionフィールドが存在しない場合は追加
                if (!root.TryGetProperty("caption", out _))
                {
                    writer.WriteString("caption", caption);
                }
                
                // tagsフィールドが存在しない場合は追加（GLM-4.1V使用時など）
                if (!root.TryGetProperty("tags", out _) && !string.IsNullOrEmpty(tagString))
                {
                    writer.WriteString("tags", tagString);
                }
                
                writer.WriteEndObject();
                writer.Flush();
                
                var updatedJson = System.Text.Encoding.UTF8.GetString(stream.ToArray());
                File.WriteAllText(filePath, updatedJson);
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"JSONファイルの更新に失敗: {Path.GetFileName(filePath)} - {ex.Message}");
            }
        }
        
        private void CreateJsonFile(string filePath, string tagString, string caption = "")
        {
            try
            {
                var jsonObject = new
                {
                    tags = tagString,
                    caption = caption
                };
                var options = new JsonSerializerOptions { WriteIndented = true };
                string jsonString = JsonSerializer.Serialize(jsonObject, options);
                File.WriteAllText(filePath, jsonString);
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"JSONファイルの作成に失敗: {Path.GetFileName(filePath)} - {ex.Message}");
            }
        }

        // 選択された画像とそのタグを削除
        private async void DeleteSelectedImageAndTags_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null)
            {
                AddMainLogEntry("画像が選択されていません。");
                return;
            }

            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show($"選択された画像 '{System.IO.Path.GetFileName(selectedImage.ImagePath)}' とそのタグを削除しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("削除がキャンセルされました。");
                    return;
                }
            }

            try
            {
                // 削除前に現在のインデックスを保持
                int currentIndex = ImageListBox.SelectedIndex;

                // ItemsSourceを一時的にnullに設定
                ImageListBox.ItemsSource = null;
                
                _imageInfos.Remove(selectedImage);
                _originalImageInfos.Remove(selectedImage);

                // GCを強制的に実行してリソースを解放
                GC.Collect();
                GC.WaitForPendingFinalizers();

                if (File.Exists(selectedImage.ImagePath))
                {
                    File.Delete(selectedImage.ImagePath);
                }
                // .txt または .json ファイルを削除
                string textFilePath = System.IO.Path.ChangeExtension(selectedImage.ImagePath, ".txt");
                string jsonFilePath = System.IO.Path.ChangeExtension(selectedImage.ImagePath, ".json");
                
                if (File.Exists(textFilePath))
                {
                    File.Delete(textFilePath);
                }
                else if (File.Exists(jsonFilePath))
                {
                    File.Delete(jsonFilePath);
                }

                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();

                // ItemsSourceを再設定
                ImageListBox.ItemsSource = _imageInfos;

                // インデックスを適切に設定
                if (_imageInfos.Count > 0)
                {
                    if (currentIndex >= _imageInfos.Count)
                    {
                        currentIndex = _imageInfos.Count - 1;
                    }
                    ImageListBox.SelectedIndex = currentIndex;
                    ImageListBox.ScrollIntoView(ImageListBox.SelectedItem);

                    ListBoxItem item = ImageListBox.ItemContainerGenerator.ContainerFromIndex(ImageListBox.SelectedIndex) as ListBoxItem;
                    item.Focus();
                }

                AddMainLogEntry($"画像 '{System.IO.Path.GetFileName(selectedImage.ImagePath)}' とそのタグを削除しました。");
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"画像の削除中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"画像の削除中にエラーが発生: {ex.Message}");
            }
            finally
            {
                UpdateUIAfterImageInfosChange();
                UpdateButtonStates();
            }
        }

        private void DeleteFilteredImageAndTags_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("削除対象の画像がありません。");
                return;
            }

            // このメソッドは、確認を必ず行う
            var result = MessageBox.Show(
                $"フィルタされた{_imageInfos.Count}個の画像とそのタグを削除しますか？", 
                "確認", 
                MessageBoxButton.YesNo, 
                MessageBoxImage.Question,
                MessageBoxResult.No  // デフォルトを「いいえ」に設定
            );
            if (result != MessageBoxResult.Yes)
            {
                AddMainLogEntry("削除がキャンセルされました。");
                return;
            }

            try
            {
                // ImageListBoxの選択をクリア
                ImageListBox.SelectedItem = null;

                // 画像とタグファイルを削除
                foreach (var imageInfo in _imageInfos.ToList())
                {
                    // GCを強制的に実行してリソースを解放
                    GC.Collect();
                    GC.WaitForPendingFinalizers();

                    if (File.Exists(imageInfo.ImagePath))
                    {
                        File.Delete(imageInfo.ImagePath);
                    }
                    // .txt または .json ファイルを削除
                    string textFilePath = System.IO.Path.ChangeExtension(imageInfo.ImagePath, ".txt");
                    string jsonFilePath = System.IO.Path.ChangeExtension(imageInfo.ImagePath, ".json");
                    
                    if (File.Exists(textFilePath))
                    {
                        File.Delete(textFilePath);
                    }
                    else if (File.Exists(jsonFilePath))
                    {
                        File.Delete(jsonFilePath);
                    }

                    _imageInfos.Remove(imageInfo);
                    _originalImageInfos.Remove(imageInfo);
                }

                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();

                UpdateUIAfterImageInfosChange();
                UpdateButtonStates();
                AddMainLogEntry($"{_imageInfos.Count}個の画像とそのタグを削除しました。");
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"画像の削除中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"画像の削除中にエラーが発生: {ex.Message}");
            }
        }

        private async void DeleteOrphanedTagsButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedFolderPath)) return;

            var result = MessageBox.Show(
                "紐づいている画像が存在しないtxtファイルをすべて削除しますか？",
                "確認",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question,
                MessageBoxResult.No
            );
            if (result != MessageBoxResult.Yes) return;

            try
            {
                var txtFiles = Directory.GetFiles(_selectedFolderPath, "*.txt", SearchOption.AllDirectories);
                int deletedCount = 0;

                foreach (var txtFile in txtFiles)
                {
                    var imageFile = Path.ChangeExtension(txtFile, ".jpg");
                    var imageFileWebp = Path.ChangeExtension(txtFile, ".webp");
                    var imageFilePng = Path.ChangeExtension(txtFile, ".png");

                    if (!File.Exists(imageFile) && !File.Exists(imageFileWebp) && !File.Exists(imageFilePng))
                    {
                        File.Delete(txtFile);
                        deletedCount++;
                    }
                }

                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");

                UpdateUIAfterImageInfosChange();
                UpdateButtonStates();
                AddMainLogEntry($"{deletedCount}個の孤立したタグファイルを削除しました");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"削除中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"削除中にエラーが発生: {ex.Message}");
            }
        }

        private async void DeleteLowResButton_Click(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(ResolutionTextBox.Text, out int threshold))
            {
                MessageBox.Show("有効な解像度を入力してください", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            var result = MessageBox.Show(
                $"{(LongSideRadio.IsChecked == true ? "長辺" : "短辺")}が{threshold}ピクセル以下の画像とタグファイルを削除しますか？",
                "確認",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question,
                MessageBoxResult.No
            );
            if (result != MessageBoxResult.Yes) return;

            try
            {
                int deletedCount = 0;
                var imageFiles = Directory.GetFiles(_selectedFolderPath, "*.*", SearchOption.AllDirectories)
                    .Where(f => f.ToLower().EndsWith(".jpg") || f.ToLower().EndsWith(".png") || f.ToLower().EndsWith(".webp"));

                foreach (var imageFile in imageFiles)
                {
                    try
                    {
                        using (var image = DrawingImage.FromFile(imageFile))
                        {
                            int relevantSize = LongSideRadio.IsChecked == true ?
                                Math.Max(image.Width, image.Height) :
                                Math.Min(image.Width, image.Height);

                            if (relevantSize <= threshold)
                            {
                                var txtFile = Path.ChangeExtension(imageFile, ".txt");
                                var jsonFile = Path.ChangeExtension(imageFile, ".json");
                                if (File.Exists(txtFile)) File.Delete(txtFile);
                                if (File.Exists(jsonFile)) File.Delete(jsonFile);
                                File.Delete(imageFile);
                                deletedCount++;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        AddMainLogEntry($"画像の処理中にエラー: {Path.GetFileName(imageFile)} - {ex.Message}");
                    }
                }

                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");

                UpdateUIAfterImageInfosChange();
                UpdateButtonStates();
                AddMainLogEntry($"{deletedCount}個の低解像度画像とタグファイルを削除しました");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"削除中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"削除中にエラーが発生: {ex.Message}");
            }
        }

        private async void DeleteLowTagCountButton_Click(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(TagCountTextBox.Text, out int threshold))
            {
                MessageBox.Show("有効なタグ数を入力してください", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            var result = MessageBox.Show(
                $"{threshold}個以下のタグを持つ画像とタグファイルを削除しますか？（タグがない画像は除外されます）",
                "確認",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question,
                MessageBoxResult.No
            );
            if (result != MessageBoxResult.Yes) return;

            try
            {
                int deletedCount = 0;
                var txtFiles = Directory.GetFiles(_selectedFolderPath, "*.txt", SearchOption.AllDirectories);

                foreach (var txtFile in txtFiles)
                {
                    var tags = File.ReadAllText(txtFile).Split(',').Select(t => t.Trim()).Where(t => !string.IsNullOrEmpty(t));
                    int tagCount = tags.Count();

                    if (tagCount > 0 && tagCount <= threshold)
                    {
                        var imageFile = Path.ChangeExtension(txtFile, ".jpg");
                        var imageFileWebp = Path.ChangeExtension(txtFile, ".webp");
                        var imageFilePng = Path.ChangeExtension(txtFile, ".png");

                        if (File.Exists(imageFile)) File.Delete(imageFile);
                        if (File.Exists(imageFileWebp)) File.Delete(imageFileWebp);
                        if (File.Exists(imageFilePng)) File.Delete(imageFilePng);
                        
                        File.Delete(txtFile);
                        deletedCount++;
                    }
                }

                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");

                UpdateUIAfterImageInfosChange();
                UpdateButtonStates();
                AddMainLogEntry($"{deletedCount}個の画像とタグファイルを削除しました");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"削除中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"削除中にエラーが発生: {ex.Message}");
            }
        }

        // キャンセルボタンのクリックイベントハンドラ
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            _cts?.Cancel();
            AddMainLogEntry("処理のキャンセルが要求されました");
        }

        private void SelectWebPDllButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "DLLファイル (*.dll)|*.dll",
                Title = "WebP.dllを選択してください"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                _webpDllPath = openFileDialog.FileName;
                WebPDllPathTextBox.Text = _webpDllPath;
                SaveSettings();
            }
        }

        // 画像リストの更新
        private void UpdateImageList()
        {
            ImageListBox.ItemsSource = _imageInfos;
        }

        private void UpdateImageCountDisplay()
        {
            int imageCount = _imageInfos?.Count ?? 0;
            int originalCount = _originalImageInfos?.Count ?? 0;
            ImageCountDisplay.Text = $"画像数: {imageCount}/{originalCount}";
        }

        // 中央ペイン: 選択された画像の表示と関連テキストの表示
        private void UpdateCentralDisplay()
        {
            if (ImageListBox.SelectedItem is ImageInfo selectedImage)
            {
                // 最新の画像ファイルを読み込む
                SelectedImage.Source = LoadImage(selectedImage.ImagePath);
                AssociatedText.Text = selectedImage.AssociatedText;
            }
        }
        
        private void ImageListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isUpdatingSelection) return;

            // 画像の更新前に表示を解除し、メモリを解放する
            SelectedImage.Source = null;
            AssociatedText.Text = "";

            if (ImageListBox.SelectedItem is ImageInfo selectedImage)
            {
                try
                {
                    _isUpdatingSelection = true;
                    
                    // UpdateCentralDisplay()の代わりにUpdateSelectedImageを使用
                    UpdateSelectedImage(selectedImage);
                    
                    _currentImageTags = new HashSet<string>(selectedImage.Tags);
                    
                    // タグの更新は不要なのでfalseにして若干UIの更新処理を軽くする
                    UpdateUIAfterSelectionChange(updateAllTagSelection: false);

                    if (_currentClusterMode != ClusterMode.Off)
                    {
                        DrawSelectedImagePoint(_umapEmbeddings, _clusterAssignments);
                    }
                    
                    AddMainLogEntry($"画像を選択しました: {System.IO.Path.GetFileName(selectedImage.ImagePath)}");
                }
                catch (Exception ex)
                {
                    AddMainLogEntry($"画像の読み込み中にエラーが発生しました: {ex.Message}");
                }
                finally
                {
                    _isUpdatingSelection = false;
                }
            }
            else
            {
                // 選択が解除された場合
                UpdateSelectedImage(null);
            }
        }

        private void UpdateSelectedImage(ImageInfo imageInfo)
        {
            if (imageInfo == null)
            {
                SelectedImage.Source = null;
                AssociatedText.Text = string.Empty;
                CaptionTextBox.Text = string.Empty;
                UpdateImageInfo(null);  // この呼び出しが実行されているか確認
                return;
            }

            try
            {
                // 画像の読み込みと表示
                SelectedImage.Source = LoadImage(imageInfo.ImagePath);
                
                // テキストの更新
                AssociatedText.Text = imageInfo.AssociatedText;
                
                // 画像情報の更新 - この呼び出しが実行されているか確認
                UpdateImageInfo(imageInfo);
                
                // キャプションの更新
                CaptionTextBox.Text = imageInfo.Caption ?? string.Empty;
                
                // デバッグログの追加
                AddDebugLogEntry($"画像情報を更新: {imageInfo.ImagePath}");
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"画像の表示に失敗: {ex.Message}");
                SelectedImage.Source = null;
                AssociatedText.Text = string.Empty;
                CaptionTextBox.Text = string.Empty;
                UpdateImageInfo(null);
            }
        }

        private void UpdateImageInfo(ImageInfo imageInfo)
        {
            // デバッグログの追加
            AddDebugLogEntry($"UpdateImageInfo called with: {(imageInfo?.ImagePath ?? "null")}");

            if (imageInfo == null)
            {
                ImagePathTextBox.Text = string.Empty;
                ImageExtensionTextBox.Text = string.Empty;
                ImageWidthTextBox.Text = string.Empty;
                ImageHeightTextBox.Text = string.Empty;
                ImageFileSizeTextBox.Text = string.Empty;
                return;
            }

            try
            {
                // フルパスと拡張子
                ImagePathTextBox.Text = imageInfo.ImagePath;
                ImageExtensionTextBox.Text = Path.GetExtension(imageInfo.ImagePath);

                // ファイルサイズ
                var fileInfo = new FileInfo(imageInfo.ImagePath);
                ImageFileSizeTextBox.Text = FormatFileSize(fileInfo.Length);

                // 画像サイズ
                using (var fs = new FileStream(imageInfo.ImagePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    if (Path.GetExtension(imageInfo.ImagePath).ToLower() == ".webp")
                    {
                        var bitmapSource = _webPHandler.LoadWebPImage(imageInfo.ImagePath);
                        ImageWidthTextBox.Text = $"{bitmapSource.PixelWidth}px";
                        ImageHeightTextBox.Text = $"{bitmapSource.PixelHeight}px";
                    }
                    else
                    {
                        using (var bitmap = new System.Drawing.Bitmap(fs))
                        {
                            ImageWidthTextBox.Text = $"{bitmap.Width}px";
                            ImageHeightTextBox.Text = $"{bitmap.Height}px";
                        }
                    }
                }

                // デバッグログの追加
                AddDebugLogEntry($"画像情報を更新完了: {imageInfo.ImagePath}");
            }
            catch (Exception ex)
            {
                ImageWidthTextBox.Text = "エラー";
                ImageHeightTextBox.Text = "エラー";
                ImageFileSizeTextBox.Text = "エラー";
                AddMainLogEntry($"画像情報の取得に失敗: {ex.Message}");
            }
        }

        private string FormatFileSize(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;
            
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size = size / 1024;
            }

            return $"{size:0.##} {sizes[order]}";
        }

        private void SelectedImage_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
                {
                    // 範囲選択モードの場合は、通常のドラッグを防ぐ
                    e.Handled = true;
                }
                else
                {
                    // 通常のドラッグ処理
                    _startPoint = e.GetPosition(null);  // WPFのマウスイベントはSystem.Windows.Pointを返す
                }
            }
        }

        private void SelectedImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && !_isSelecting && SelectedImage.Source != null)
            {
                if (_startPoint.HasValue)  // nullチェックを追加
                {
                    WindowsPoint currentPoint = e.GetPosition(null);
                    Vector diff = currentPoint - _startPoint.Value;

                    if (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance ||
                        Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance)
                    {
                        var bitmap = SelectedImage.Source as BitmapSource;
                        if (bitmap == null) return;

                        string tempFile = Path.Combine(Path.GetTempPath(), $"dragdrop_image_{Guid.NewGuid()}.png");
                        
                        try
                        {
                            using (var fileStream = new FileStream(tempFile, FileMode.Create))
                            {
                                BitmapEncoder encoder = new PngBitmapEncoder();
                                encoder.Frames.Add(BitmapFrame.Create(bitmap));
                                encoder.Save(fileStream);
                            }

                            var dataObject = new DataObject();
                            dataObject.SetData(DataFormats.FileDrop, new[] { tempFile });
                            dataObject.SetData(DataFormats.Bitmap, bitmap);

                            // DoDragDropの後の処理を修正
                            DragDrop.DoDragDrop(SelectedImage, dataObject, DragDropEffects.Copy);
                            
                            // 別のタスクとして一時ファイルの削除を実行
                            Task.Run(async () =>
                            {
                                try
                                {
                                    // 1秒待機
                                    await Task.Delay(1000);
                                    
                                    await Application.Current.Dispatcher.InvokeAsync(() =>
                                    {
                                        if (File.Exists(tempFile))
                                        {
                                            try
                                            {
                                                File.Delete(tempFile);
                                            }
                                            catch { /* 削除に失敗しても続行 */ }
                                        }
                                    });
                                }
                                catch { /* エラーは無視 */ }
                            });
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show($"画像のドラッグ中にエラーが発生しました: {ex.Message}");
                            if (File.Exists(tempFile))
                            {
                                try
                                {
                                    File.Delete(tempFile);
                                }
                                catch { /* 削除に失敗しても続行 */ }
                            }
                        }
                    }
                }
            }
        }

        private void SelectedImage_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                SelectionHint.Visibility = Visibility.Visible;
                SelectionCanvas.IsHitTestVisible = true;
                SelectionCanvas.Background = WindowsBrushes.Transparent;  // 追加
            }
            else
            {
                SelectionHint.Visibility = Visibility.Collapsed;
                SelectionCanvas.IsHitTestVisible = false;
                SelectionCanvas.Background = null;  // 追加: 通常のマウス操作を可能にする
            }
        }

        private void SelectionCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _startPoint = e.GetPosition(SelectionCanvas);
            
            if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                _isSelecting = true;
                SelectionRectangle.Visibility = Visibility.Visible;
                Canvas.SetLeft(SelectionRectangle, _startPoint.Value.X);
                Canvas.SetTop(SelectionRectangle, _startPoint.Value.Y);
                SelectionRectangle.Width = 0;
                SelectionRectangle.Height = 0;
            }
            else
            {
                _isDragging = true;
            }
            e.Handled = true;
        }

        private void SelectionCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_startPoint.HasValue) return;

            if (_isSelecting)
            {
                // 範囲選択の処理
                WindowsPoint currentPoint = e.GetPosition(SelectionCanvas);
                var x = Math.Min(_startPoint.Value.X, currentPoint.X);
                var y = Math.Min(_startPoint.Value.Y, currentPoint.Y);
                var width = Math.Abs(currentPoint.X - _startPoint.Value.X);
                var height = Math.Abs(currentPoint.Y - _startPoint.Value.Y);

                Canvas.SetLeft(SelectionRectangle, x);
                Canvas.SetTop(SelectionRectangle, y);
                SelectionRectangle.Width = width;
                SelectionRectangle.Height = height;
            }
            else if (_isDragging && e.LeftButton == MouseButtonState.Pressed && SelectedImage.Source != null)
            {
                // ドラッグ処理
                WindowsPoint currentPoint = e.GetPosition(null);
                Vector diff = currentPoint - _startPoint.Value;

                if (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance ||
                    Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance)
                {
                    var bitmap = SelectedImage.Source as BitmapSource;
                    if (bitmap == null) return;

                    string tempFile = Path.Combine(Path.GetTempPath(), $"dragdrop_image_{Guid.NewGuid()}.png");
                    
                    try
                    {
                        using (var fileStream = new FileStream(tempFile, FileMode.Create))
                        {
                            BitmapEncoder encoder = new PngBitmapEncoder();
                            encoder.Frames.Add(BitmapFrame.Create(bitmap));
                            encoder.Save(fileStream);
                        }

                        var dataObject = new DataObject();
                        dataObject.SetData(DataFormats.FileDrop, new[] { tempFile });
                        dataObject.SetData(DataFormats.Bitmap, bitmap);

                        DragDrop.DoDragDrop(SelectedImage, dataObject, DragDropEffects.Copy);
                        
                        // 一時ファイルの削除を別タスクで実行
                        Task.Run(async () =>
                        {
                            try
                            {
                                await Task.Delay(1000);
                                await Application.Current.Dispatcher.InvokeAsync(() =>
                                {
                                    if (File.Exists(tempFile))
                                    {
                                        try { File.Delete(tempFile); }
                                        catch { /* 削除に失敗しても続行 */ }
                                    }
                                });
                            }
                            catch { /* エラーは無視 */ }
                        });
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"画像のドラッグ中にエラーが発生しました: {ex.Message}");
                        if (File.Exists(tempFile))
                        {
                            try { File.Delete(tempFile); }
                            catch { /* 削除に失敗しても続行 */ }
                        }
                    }
                }
            }
            e.Handled = true;
        }

        private void SelectionCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isSelecting)
            {
                _isSelecting = false;
                // 選択範囲が十分な大きさかチェック
                if (SelectionRectangle.Width < 10 || SelectionRectangle.Height < 10)
                {
                    SelectionRectangle.Visibility = Visibility.Collapsed;
                    AddDebugLogEntry("選択範囲が小さすぎるため、キャンセルされました");
                }
                else
                {
                    AddDebugLogEntry($"範囲選択完了: {SelectionRectangle.Width:F0}x{SelectionRectangle.Height:F0}");
                }
            }
            
            _isDragging = false;
            _startPoint = null;
            e.Handled = true;
        }

        // キーボードイベントを監視して、Ctrlキーの状態に応じてカーソルを変更
        private void SelectionCanvas_MouseEnter(object sender, MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
            {
                SelectionCanvas.Cursor = Cursors.Cross;
            }
        }

        private void SelectionCanvas_MouseLeave(object sender, MouseEventArgs e)
        {
            SelectionCanvas.Cursor = Cursors.Arrow;
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape && SelectionRectangle.Visibility == Visibility.Visible)
            {
                SelectionRectangle.Visibility = Visibility.Collapsed;
                _isSelecting = false;
                _isInSelectionMode = false;
                SelectionCanvas.ReleaseMouseCapture();
                AddDebugLogEntry("範囲選択がキャンセルされました");
            }
            else if (e.Key == Key.LeftCtrl || e.Key == Key.RightCtrl)
            {
                SelectionCanvas.Cursor = Cursors.Cross;
            }
        }

        private void Window_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.LeftCtrl || e.Key == Key.RightCtrl)
            {
                SelectionCanvas.Cursor = Cursors.Arrow;
            }
        }

        private void MainWindow_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            // 既存のOnKeyDownメソッドがキーハンドリングを行うため、
            // このメソッドは不要（削除予定）
        }

        // 元に戻す
        private void UndoButton_Click(object sender, RoutedEventArgs e)
        {
            if (_undoStack.Count > 0)
            {
                var action = _undoStack.Pop();
                action.UndoAction();
                _redoStack.Push(action);
                UpdateUIAfterTagsChange();
                AddActionLogItem("元に戻す", action.Description);
            }
        }

        // やり直し
        private void RedoButton_Click(object sender, RoutedEventArgs e)
        {
            if (_redoStack.Count > 0)
            {
                var action = _redoStack.Pop();
                action.DoAction();
                _undoStack.Push(action);
                UpdateUIAfterTagsChange();
                AddActionLogItem("やり直し", action.Description);
            }
        }

        // 右ペイン1: 現在の画像のタグリスト表示と選択
        // タグリストビューの更新
        private void UpdateTagListView()
        {
            AddDebugLogEntry("UpdateTagListView");

            var currentTags = _currentImageTags.ToList();
            TagListView.ItemsSource = currentTags;

            _isUpdatingSelection = true;
            try
            {
                // 選択状態を更新
                TagListView.SelectionChanged -= TagListView_SelectionChanged;
                TagListView.SelectedItems.Clear();
                var tagsToSelect = currentTags.Where(tag => _selectedTags.Contains(tag)).ToList();
                foreach (var tag in tagsToSelect)
                {
                    TagListView.SelectedItems.Add(tag);
                }
                TagListView.SelectionChanged += TagListView_SelectionChanged;
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        // 個別タグリストの選択
        private void TagListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            AddDebugLogEntry("TagListView_SelectionChanged");

            // ここで選択は処理しないので、SelectionChangedでの選択状態の反映はキャンセルする
            foreach (var item in e.AddedItems)
            {
                if (TagListView.SelectedItems.Contains(item))
                {
                    TagListView.SelectedItems.Remove(item);
                }
            }
        }

        // 個別タグリストの選択解除
        private void DeselectTagButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedTags = _selectedTags.ToList();
            foreach (var tag in selectedTags)
            {
                _selectedTags.Remove(tag);
            }
            UpdateUIAfterTagSelectionChange();
        }

        // 個別タグの更新
        private void UpdateCurrentTags()
        {
            _currentImageTags.Clear();
            var imageInfo = ImageListBox.SelectedItem as ImageInfo;
            if (imageInfo != null)
            {
                _currentImageTags = new HashSet<string>(imageInfo.Tags);
            }
        }

        // タグの追加
        private void AddTagButton_Click(object sender, RoutedEventArgs e)
        {
            AddMainLogEntry("AddTagButton_Click が呼び出されました");
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                AddMainLogEntry($"選択された画像: {Path.GetFileName(selectedImage.ImagePath)}");
                var selectedTags = _selectedTags.ToList();
                AddMainLogEntry($"選択されたタグ数: {selectedTags.Count}");
                var addedTags = new List<TagPositionInfo>();

                foreach (var tag in selectedTags)
                {
                    if (!selectedImage.Tags.Contains(tag))
                    {
                        int insertPosition = selectedImage.Tags.Count;
                        addedTags.Add(new TagPositionInfo { Tag = tag, Position = insertPosition });
                    }
                }

                AddMainLogEntry($"追加予定のタグ数: {addedTags.Count}");
                if (addedTags.Count > 0)
                {
                    var action = new TagGroupAction
                    {
                        Image = selectedImage,
                        TagInfos = addedTags,
                        IsAdd = true,
                        DoAction = () =>
                        {
                            foreach (var tagInfo in addedTags)
                            {
                                selectedImage.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                            AddMainLogEntry($"{addedTags.Count}個のタグを追加しました");
                        },
                        UndoAction = () =>
                        {   
                            foreach (var tagInfo in addedTags.OrderByDescending(t => t.Position))
                            {
                                selectedImage.Tags.RemoveAt(tagInfo.Position);
                            }
                            AddMainLogEntry($"{addedTags.Count}個のタグの追加を取り消しました");
                        },
                        Description = $"{addedTags.Count}個のタグ追加"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                }
                else
                {
                    AddMainLogEntry("追加するタグがありません（既に存在するか選択されていません）");
                }
            }
            else
            {
                AddMainLogEntry("画像が選択されていません");
            }
        }

        // タグの削除
        private void RemoveTagButton_Click(object sender, RoutedEventArgs e)
        {
            AddMainLogEntry("RemoveTagButton_Click が呼び出されました");
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                var selectedTags = _selectedTags.ToList();
                AddMainLogEntry($"削除対象のタグ数: {selectedTags.Count}");
                var removedTags = new List<TagPositionInfo>();

                foreach (var tag in selectedTags)
                {
                    int removePosition = selectedImage.Tags.IndexOf(tag);
                    if (removePosition != -1)
                    {
                        removedTags.Add(new TagPositionInfo { Tag = tag, Position = removePosition });
                    }
                }

                if (removedTags.Count > 0)
                {
                    var action = new TagGroupAction
                    {
                        Image = selectedImage,
                        TagInfos = removedTags,
                        IsAdd = false,
                        DoAction = () =>
                        {
                            foreach (var tagInfo in removedTags.OrderByDescending(t => t.Position))
                            {
                                selectedImage.Tags.RemoveAt(tagInfo.Position);
                                // _selectedTags.Remove(tagInfo.Tag);
                            }
                            AddMainLogEntry($"{removedTags.Count}個のタグを削除しました");
                        },
                        UndoAction = () =>
                        {
                            foreach (var tagInfo in removedTags)
                            {
                                selectedImage.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                            AddMainLogEntry($"{removedTags.Count}個のタグの削除を取り消しました");
                        },
                        Description = $"{removedTags.Count}個のタグを削除"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                }
            }
        }

        private void MoveTopButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                var selectedTags = _selectedTags.ToList();
                var movedTags = new List<TagPositionInfo>();

                foreach (var tag in selectedTags)
                {
                    int currentPosition = selectedImage.Tags.IndexOf(tag);
                    if (currentPosition > 0)
                    {
                        movedTags.Add(new TagPositionInfo { Tag = tag, Position = currentPosition });
                    }
                }

                if (movedTags.Count > 0)
                {
                    var action = new TagGroupAction
                    {
                        Image = selectedImage,
                        TagInfos = movedTags,
                        IsAdd = false, // 移動操作なのでfalse
                        DoAction = () =>
                        {
                            foreach (var tagInfo in movedTags.OrderBy(t => t.Position))
                            {
                                selectedImage.Tags.RemoveAt(tagInfo.Position);
                                selectedImage.Tags.Insert(0, tagInfo.Tag);
                            }
                            AddMainLogEntry($"{movedTags.Count}個のタグを先頭に移動しました");
                        },
                        UndoAction = () =>
                        {
                            foreach (var tagInfo in movedTags.OrderByDescending(t => t.Position))
                            {
                                selectedImage.Tags.Remove(tagInfo.Tag);
                                selectedImage.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                            AddMainLogEntry($"{movedTags.Count}個のタグの移動を元に戻しました");
                        },
                        Description = $"{movedTags.Count}個のタグを先頭に移動"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                }
            }
        }

        private void MoveBottomButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                var selectedTags = _selectedTags.ToList();
                var movedTags = new List<TagPositionInfo>();

                int lastIndex = selectedImage.Tags.Count - 1;
                foreach (var tag in selectedTags)
                {
                    int currentPosition = selectedImage.Tags.IndexOf(tag);
                    if (currentPosition < lastIndex)
                    {
                        movedTags.Add(new TagPositionInfo { Tag = tag, Position = currentPosition });
                    }
                }

                if (movedTags.Count > 0)
                {
                    var action = new TagGroupAction
                    {
                        Image = selectedImage,
                        TagInfos = movedTags,
                        IsAdd = false, // 移動操作なのでfalse
                        DoAction = () =>
                        {
                            foreach (var tagInfo in movedTags.OrderByDescending(t => t.Position))
                            {
                                selectedImage.Tags.RemoveAt(tagInfo.Position);
                                selectedImage.Tags.Add(tagInfo.Tag);
                            }
                            AddMainLogEntry($"{movedTags.Count}個のタグを末尾に移動しました");
                        },
                        UndoAction = () =>
                        {
                            foreach (var tagInfo in movedTags)
                            {
                                selectedImage.Tags.RemoveAt(selectedImage.Tags.Count - 1);
                                selectedImage.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                            AddMainLogEntry($"{movedTags.Count}個のタグの移動を元に戻しました");
                        },
                        Description = $"{movedTags.Count}個のタグを末尾に移動"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                }
            }
        }

        // 右ペイン2: 全タグリストの表示、選択、ソート
        // 全タグリストビューの更新
        private void UpdateAllTagsListView(bool updateTagSelection = true)
        {
            AddDebugLogEntry($"UpdateAllTagsListView(updateTagSelection:{updateTagSelection})");

            var sortedTags = _allTags
                .Select(kvp => new
                {
                    Tag = kvp.Key,
                    Count = kvp.Value,
                    IsSelected = _selectedTags.Contains(kvp.Key),
                    IsCurrentImageTag = _currentImageTags.Contains(kvp.Key),
                    Category = GetTagCategory(kvp.Key)
                })
                .OrderByDescending(item => item.IsSelected)
                .ThenByDescending(item => item.Count)
                .ThenBy(item => item.Tag)
                .ToList();

            AllTagsListView.SelectionChanged -= AllTagsListView_SelectionChanged;
            AllTagsListView.ItemsSource = sortedTags;

            // 選択状態を更新
            if (updateTagSelection)
            {
                AllTagsListView.SelectedItems.Clear();
                var selectedItems = AllTagsListView.Items.Cast<dynamic>()
                    .Where(item => _selectedTags.Contains(item.Tag))
                    .ToList();
                foreach (var item in selectedItems)
                {
                    AllTagsListView.SelectedItems.Add(item);
                }
            }
            AllTagsListView.SelectionChanged += AllTagsListView_SelectionChanged;
        }

        // 全タグリストの選択
        private void AllTagsListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            AddDebugLogEntry("AllTagsListView_SelectionChanged");
            if (_isUpdatingSelection) return;

            _isUpdatingSelection = true;
            try
            {
                foreach (var item in e.RemovedItems)
                {
                    var removedTag = ((dynamic)item).Tag as string;
                    _selectedTags.Remove(removedTag);
                }

                foreach (var item in e.AddedItems)
                {
                    var addedTag = ((dynamic)item).Tag as string;
                    _selectedTags.Add(addedTag);
                }

                // UpdateAllTagsListView();
                // UpdateTagListView();  // 個別タグリストを更新
                // UpdateSelectedTagsListBox();
                // UpdateSearchedTagsListView();
                UpdateUIAfterTagSelectionChange();
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        // 全タグリストの選択解除
        private void DeselectAllTagsButton_Click(object sender, RoutedEventArgs e)
        {
            _selectedTags.Clear();
            // UpdateTagListView();
            // UpdateAllTagsListView();
            // UpdateSelectedTagsListBox();
            // UpdateSearchedTagsListView();
            UpdateUIAfterTagSelectionChange();
        }

        // 全タグの更新
        private void UpdateAllTags()
        {
            if (_allTags == null) { _allTags = new Dictionary<string, int>(); }
            _allTags.Clear();
            // TODO: 変更の衝突問題
            // System.InvalidOperationException: 'Collection was modified; enumeration operation may not execute.'
            _imageInfos.ForEach(imageInfo =>
            {
                imageInfo.Tags.ForEach(tag =>
                {
                    if (tag == null) return; // なぜここで null...? 
                    _allTags[tag] = _allTags.ContainsKey(tag) ? _allTags[tag] + 1 : 1;
                });
            });
        }

        // 全画像にタグの追加
        private void AddAllTagsButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("選択したタグをすべての画像に追加しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("タグの追加がキャンセルされました。");
                    return;
                }
            }

            var selectedTags = AllTagsListView.SelectedItems.Cast<dynamic>().Select(item => item.Tag as string).ToList();
            if (selectedTags.Count == 0)
            {
                AddMainLogEntry("追加するタグが選択されていません。");
                return;
            }

            var addedToImages = new List<ImageInfo>();

            foreach (var imageInfo in _imageInfos)
            {
                bool tagsAdded = false;
                foreach (var tag in selectedTags)
                {
                    if (!imageInfo.Tags.Contains(tag))
                    {
                        imageInfo.Tags.Add(tag);
                        tagsAdded = true;
                    }
                }
                if (tagsAdded)
                {
                    addedToImages.Add(imageInfo);
                }
            }

            if (addedToImages.Count > 0)
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        AddMainLogEntry($"選択したタグを {addedToImages.Count} 個の画像に追加しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var imageInfo in addedToImages)
                        {
                            foreach (var tag in selectedTags)
                            {
                                imageInfo.Tags.Remove(tag);
                            }
                        }
                        AddMainLogEntry($"選択したタグの追加を {addedToImages.Count} 個の画像から取り消しました。");
                    },
                    Description = $"選択したタグを {addedToImages.Count} 個の画像に追加"
                };

                _undoStack.Push(action);
                _redoStack.Clear();
                UpdateUIAfterTagsChange();
                action.DoAction();
            }
            else
            {
                AddMainLogEntry("選択したタグは既にすべての画像に存在します。");
            }
        }

        // 全画像からタグの削除
        private void RemoveAllTagsButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("選択したタグをすべての画像から削除しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("タグの削除がキャンセルされました。");
                    return;
                }
            }
            var selectedTags = AllTagsListView.SelectedItems.Cast<dynamic>().Select(item => item.Tag as string).ToList();
            var removedTags = new Dictionary<ImageInfo, List<TagPositionInfo>>();

            foreach (var imageInfo in _imageInfos)
            {
                var tagsToRemove = imageInfo.Tags
                    .Select((tag, index) => new { Tag = tag, Index = index })
                    .Where(item => selectedTags.Contains(item.Tag))
                    .Select(item => new TagPositionInfo { Tag = item.Tag, Position = item.Index })
                    .ToList();

                if (tagsToRemove.Count > 0)
                {
                    removedTags[imageInfo] = tagsToRemove;
                }
            }

            if (removedTags.Count > 0)
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        foreach (var kvp in removedTags)
                        {
                            foreach (var tagInfo in kvp.Value.OrderByDescending(t => t.Position))
                            {
                                //テスト中
                                if (tagInfo.Position < kvp.Key.Tags.Count)
                                {
                                    kvp.Key.Tags.RemoveAt(tagInfo.Position);
                                }
                                else
                                {
                                    AddMainLogEntry($"タグの削除に失敗しました: インデックス {tagInfo.Position} が範囲外です。対象画像: {kvp.Key.ImagePath}、タグ: {tagInfo.Tag}");
                                }
                            }
                        }
                        foreach (var tag in selectedTags)
                        {
                            _selectedTags.Remove(tag);
                        }
                        AddMainLogEntry($"{removedTags.Sum(kvp => kvp.Value.Count)}個のタグを削除しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var kvp in removedTags)
                        {
                            foreach (var tagInfo in kvp.Value)
                            {
                                kvp.Key.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                        }
                        AddMainLogEntry($"{removedTags.Sum(kvp => kvp.Value.Count)}個のタグを復元しました。");
                    },
                    Description = $"{removedTags.Sum(kvp => kvp.Value.Count)}個のタグを全画像から削除"
                };
                _undoStack.Push(action);
                _redoStack.Clear();
                action.DoAction();
                UpdateUIAfterTagsChange();
            }
        }

        private void MoveTopAllButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("選択したタグをすべての画像の先頭に移動しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("タグの移動がキャンセルされました。");
                    return;
                }
            }

            var selectedTags = AllTagsListView.SelectedItems.Cast<dynamic>().Select(item => item.Tag as string).ToList();
            if (selectedTags.Count == 0)
            {
                AddMainLogEntry("移動するタグが選択されていません。");
                return;
            }

            var movedTags = new Dictionary<ImageInfo, List<TagPositionInfo>>();

            foreach (var imageInfo in _imageInfos)
            {
                var tagsToMove = imageInfo.Tags
                    .Select((tag, index) => new { Tag = tag, Index = index })
                    .Where(item => selectedTags.Contains(item.Tag) && item.Index > 0)
                    .Select(item => new TagPositionInfo { Tag = item.Tag, Position = item.Index })
                    .ToList();

                if (tagsToMove.Count > 0)
                {
                    movedTags[imageInfo] = tagsToMove;
                }
            }

            if (movedTags.Count > 0)
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        foreach (var kvp in movedTags)
                        {
                            foreach (var tagInfo in kvp.Value.OrderBy(t => t.Position))
                            {
                                kvp.Key.Tags.RemoveAt(tagInfo.Position);
                                kvp.Key.Tags.Insert(0, tagInfo.Tag);
                            }
                        }
                        AddMainLogEntry($"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグを先頭に移動しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var kvp in movedTags)
                        {
                            foreach (var tagInfo in kvp.Value.OrderByDescending(t => t.Position))
                            {
                                kvp.Key.Tags.RemoveAt(0);
                                kvp.Key.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                        }
                        AddMainLogEntry($"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグの移動を元に戻しました。");
                    },
                    Description = $"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグを全画像の先頭に移動"
                };

                _undoStack.Push(action);
                _redoStack.Clear();
                action.DoAction();
                UpdateUIAfterTagsChange();
            }
            else
            {
                AddMainLogEntry("移動するタグがありませんでした。");
            }
        }

        private void MoveBottomAllButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("選択したタグをすべての画像の末尾に移動しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("タグの移動がキャンセルされました。");
                    return;
                }
            }

            var selectedTags = AllTagsListView.SelectedItems.Cast<dynamic>().Select(item => item.Tag as string).ToList();
            if (selectedTags.Count == 0)
            {
                AddMainLogEntry("移動するタグが選択されていません。");
                return;
            }

            var movedTags = new Dictionary<ImageInfo, List<TagPositionInfo>>();

            foreach (var imageInfo in _imageInfos)
            {
                int lastIndex = imageInfo.Tags.Count - 1;
                var tagsToMove = imageInfo.Tags
                    .Select((tag, index) => new { Tag = tag, Index = index })
                    .Where(item => selectedTags.Contains(item.Tag) && item.Index < lastIndex)
                    .Select(item => new TagPositionInfo { Tag = item.Tag, Position = item.Index })
                    .ToList();

                if (tagsToMove.Count > 0)
                {
                    movedTags[imageInfo] = tagsToMove;
                }
            }

            if (movedTags.Count > 0)
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        foreach (var kvp in movedTags)
                        {
                            foreach (var tagInfo in kvp.Value.OrderByDescending(t => t.Position))
                            {
                                kvp.Key.Tags.RemoveAt(tagInfo.Position);
                                kvp.Key.Tags.Add(tagInfo.Tag);
                            }
                        }
                        AddMainLogEntry($"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグを末尾に移動しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var kvp in movedTags)
                        {
                            foreach (var tagInfo in kvp.Value)
                            {
                                kvp.Key.Tags.RemoveAt(kvp.Key.Tags.Count - 1);
                                kvp.Key.Tags.Insert(tagInfo.Position, tagInfo.Tag);
                            }
                        }
                        AddMainLogEntry($"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグの移動を元に戻しました。");
                    },
                    Description = $"{movedTags.Sum(kvp => kvp.Value.Count)}個のタグを全画像の末尾に移動"
                };

                _undoStack.Push(action);
                _redoStack.Clear();
                action.DoAction();
                UpdateUIAfterTagsChange();
            }
            else
            {
                AddMainLogEntry("移動するタグがありませんでした。");
            }
        }




        // ボタンエリア:特殊処理
        private void ReplaceTagButton_Click(object sender, RoutedEventArgs e)
        {
            AddDebugLogEntry("ReplaceTagButton_Click");
            var replaceTagWindow = new ReplaceTagWindow(_allTags.Keys.ToList());
            replaceTagWindow.Owner = this;
            if (replaceTagWindow.ShowDialog() == true)
            {
                ReplaceTag(
                    replaceTagWindow.SourceTag,
                    replaceTagWindow.DestinationTag,
                    replaceTagWindow.UseRegex,
                    replaceTagWindow.UsePartialMatch,
                    replaceTagWindow.ApplyToAll,
                    replaceTagWindow.ReplaceProbability
                );
            }
        }

        private void ReplaceTag(string sourceTag, string destinationTag, bool useRegex, bool usePartialMatch, bool applyToAll, double replaceProbability)
        {
            var targetImages = applyToAll ? _imageInfos : new List<ImageInfo> { ImageListBox.SelectedItem as ImageInfo };
            if (targetImages == null || !targetImages.Any())
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            var replacedTags = new Dictionary<ImageInfo, List<(string OldTag, string NewTag)>>();
            var random = new Random();

            foreach (var image in targetImages)
            {
                var tagsToReplace = new List<(string OldTag, string NewTag)>();

                for (int i = 0; i < image.Tags.Count; i++)
                {
                    string currentTag = image.Tags[i];
                    bool matchFound = false;

                    if (useRegex)
                    {
                        try
                        {
                            var regex = new Regex(sourceTag);
                            if (regex.IsMatch(currentTag))
                            {
                                string newTag = regex.Replace(currentTag, destinationTag);
                                if (newTag != currentTag && random.NextDouble() < replaceProbability)
                                {
                                    tagsToReplace.Add((currentTag, newTag));
                                    matchFound = true;
                                }
                            }
                        }
                        catch (ArgumentException ex)
                        {
                            AddMainLogEntry($"無効な正規表現: {ex.Message}");
                            return;
                        }
                    }
                    else
                    {
                        if (usePartialMatch)
                        {
                            if (currentTag.Contains(sourceTag) && random.NextDouble() < replaceProbability)
                            {
                                string newTag = currentTag.Replace(sourceTag, destinationTag);
                                tagsToReplace.Add((currentTag, newTag));
                                matchFound = true;
                            }
                        }
                        else
                        {
                            if (currentTag == sourceTag && random.NextDouble() < replaceProbability)
                            {
                                tagsToReplace.Add((currentTag, destinationTag));
                                matchFound = true;
                            }
                        }
                    }

                    if (matchFound)
                    {
                        replacedTags[image] = tagsToReplace;
                    }
                }
            }

            if (replacedTags.Any())
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        foreach (var kvp in replacedTags)
                        {
                            var image = kvp.Key;
                            foreach (var (oldTag, newTag) in kvp.Value)
                            {
                                int index = image.Tags.IndexOf(oldTag);
                                if (index != -1)
                                {
                                    image.Tags[index] = newTag;
                                }
                            }
                        }
                        AddMainLogEntry($"{replacedTags.Sum(kvp => kvp.Value.Count)}個のタグを置換しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var kvp in replacedTags)
                        {
                            var image = kvp.Key;
                            foreach (var (oldTag, newTag) in kvp.Value)
                            {
                                int index = image.Tags.IndexOf(newTag);
                                if (index != -1)
                                {
                                    image.Tags[index] = oldTag;
                                }
                            }
                        }
                        AddMainLogEntry($"{replacedTags.Sum(kvp => kvp.Value.Count)}個のタグの置換を元に戻しました。");
                    },
                    Description = $"{replacedTags.Sum(kvp => kvp.Value.Count)}個のタグを置換"
                };

                action.DoAction();
                _undoStack.Push(action);
                _redoStack.Clear();
                UpdateUIAfterTagsChange();
            }
            else
            {
                AddMainLogEntry("置換対象のタグが見つかりませんでした。");
            }
        }

        private void AddFolderNameButton_Click(object sender, RoutedEventArgs e)
        {
            AddDebugLogEntry("AddFolderNameButton_Click");

            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            if (ImageListBox.SelectedItem == null)
            {
                AddMainLogEntry("対象の画像が選択されていません。");
                return;
            }

            // 最大ディレクトリレベルを計算
            int maxLevels = _imageInfos.Max(img => 
                Path.GetDirectoryName(img.ImagePath).Split(Path.DirectorySeparatorChar).Length - 
                _selectedFolderPath.Split(Path.DirectorySeparatorChar).Length);

            var addFolderNameWindow = new AddFolderNameWindow(maxLevels);
            addFolderNameWindow.Owner = this;
            if (addFolderNameWindow.ShowDialog() == true)
            {
                AddFolderNameAsTag(
                    addFolderNameWindow.DirectoryLevels,
                    addFolderNameWindow.FromEnd,
                    addFolderNameWindow.ParseCommas,
                    addFolderNameWindow.ApplyToAll,
                    addFolderNameWindow.AddProbability
                );
            }
        }

        private void AddFolderNameAsTag(int directoryLevels, bool fromEnd, bool parseCommas, bool applyToAll, double addProbability)
        {
            var targetImages = applyToAll ? _imageInfos : new List<ImageInfo> { ImageListBox.SelectedItem as ImageInfo };
            if (targetImages == null || !targetImages.Any())
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            var addedTags = new Dictionary<ImageInfo, List<string>>();
            var random = new Random();

            foreach (var image in targetImages)
            {
                var imagePath = Path.GetDirectoryName(image.ImagePath);
                var folderNames = imagePath.Split(Path.DirectorySeparatorChar)
                                        .Skip(_selectedFolderPath.Split(Path.DirectorySeparatorChar).Length);

                if (fromEnd)
                {
                    folderNames = folderNames.Reverse().Take(directoryLevels);
                }
                else
                {
                    folderNames = folderNames.Take(directoryLevels);
                }

                var tagsToAdd = new List<string>();

                foreach (var folderName in folderNames)
                {
                    var tags = parseCommas ? folderName.Split(',') : new[] { folderName };
                    foreach (var tag in tags)
                    {
                        if (random.NextDouble() < addProbability)
                        {
                            var processedTag = ProcessTag(tag);
                            if (!string.IsNullOrWhiteSpace(processedTag) && !image.Tags.Contains(processedTag))
                            {
                                tagsToAdd.Add(processedTag);
                                AddTagToUserAddedCategory(processedTag);
                            }
                        }
                    }
                }

                if (tagsToAdd.Any())
                {
                    addedTags[image] = tagsToAdd;
                }
            }

            if (addedTags.Any())
            {
                var action = new TagGroupAction
                {
                    DoAction = () =>
                    {
                        foreach (var kvp in addedTags)
                        {
                            var image = kvp.Key;
                            image.Tags.AddRange(kvp.Value);
                        }
                        AddMainLogEntry($"{addedTags.Sum(kvp => kvp.Value.Count)}個のタグを追加しました。");
                    },
                    UndoAction = () =>
                    {
                        foreach (var kvp in addedTags)
                        {
                            var image = kvp.Key;
                            foreach (var tag in kvp.Value)
                            {
                                image.Tags.Remove(tag);
                            }
                        }
                        AddMainLogEntry($"{addedTags.Sum(kvp => kvp.Value.Count)}個のタグの追加を元に戻しました。");
                    },
                    Description = $"{addedTags.Sum(kvp => kvp.Value.Count)}個のフォルダ名タグを追加"
                };

                action.DoAction();
                _undoStack.Push(action);
                _redoStack.Clear();
                UpdateUIAfterTagsChange();
            }
            else
            {
                AddMainLogEntry("追加するタグが見つかりませんでした。");
            }
        }

        private CSDModel _csdModel;

        private async Task InitializeCSDModel()
        {
            _csdModel = await CSDModel.LoadModel(UseGPUCheckBox.IsChecked ?? false);
            _csdModel.LogUpdated += (sender, message) =>
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    // ここでUIのログ表示を更新
                    AddMainLogEntry(message);
                });
            };
        }

        private async void CSDClusteringButton_Click(object sender, RoutedEventArgs e)
        {
            AddDebugLogEntry("CSDClusteringButton_Click");

            if (_originalImageInfos == null || _originalImageInfos.Count == 0)
            {
                AddMainLogEntry("クラスタリングを行う画像がありません。");
                return;
            }

            if (_isAsyncProcessing)
            {
                AddMainLogEntry("前の処理が完了するまで待機しています。");
                return;
            }

            if (_currentClusterMode == ClusterMode.CSD)
            { 
                _currentClusterMode = ClusterMode.Off;
                _clusteredImageInfos = null;
                // 古いフィルタ機能は削除されました
                _imageInfos = new List<ImageInfo>(_originalImageInfos);
                UpdateImageList();
                AddMainLogEntry("クラスタリングを解除しました。");
                return;
            }

            _isAsyncProcessing = true;

            try
            {
                // CSDモデルのロード
                await InitializeCSDModel();

                _clusterEmbeddings = await ProcessCSDInAsyncPipeline();
                // クラスタリングの実行
                _umapEmbeddings = PerformUmapEncoding(_clusterEmbeddings);

                // k-meansクラスタリングの実行
                _clusterCount = (int)Math.Sqrt(_umapEmbeddings.Length / 2); // クラスター数の決定（ここでは簡単な方法を使用）

                _clusterAssignments = PerformKMeansClustering(_umapEmbeddings, _clusterCount);

                // クラスタリング結果の可視化   
                VisualizeClusteringResult(_umapEmbeddings, _clusterAssignments);

                // 開いている画像のクラスターを取得
                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                int selectedCluster = 0;

                if (selectedImage == null)
                {
                    AddMainLogEntry("選択された画像がありません。クラスター1を表示します。");
                    selectedCluster = 1;
                }
                else
                {
                    selectedCluster = _clusterAssignments[_originalImageInfos.IndexOf(selectedImage)];
                }

                UpdateClusterGroupComboBox(selectedCluster);

                // クラスタリング結果に基づくフィルタリング
                ApplyClusterFiltering(_clusterAssignments, selectedCluster);
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"CSDクラスタリング中にエラーが発生しました: {ex.Message}");
            }
            finally
            {
                // 処理が完了したら処理速度表示をクリア
                ProcessingSpeed = "";
                UpdateProgressBar(0);
                
                _csdModel.Dispose();
                _isAsyncProcessing = false;
                AddMainLogEntry("CSDクラスタリングが完了しました。");
            }
        }
        
        private void ClusterGroupComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_clusterAssignments == null) return;
            ApplyClusterFiltering(_clusterAssignments, ClusterGroupComboBox.SelectedIndex);
        }

        private void UpdateClusterGroupComboBox(int selectedIndex)
        {
            _clusterGroups.Clear();
            ClusterGroupComboBox.Items.Clear();

            if (_clusterCount == null || _clusterCount <= 0)
            {
                _clusterGroups.Add("ALL");
                ClusterGroupComboBox.Items.Add("ALL");
                ClusterGroupComboBox.SelectedIndex = 0;
                return;
            }
            
            _clusterGroups.Add("Select Image");
            _clusterGroups.AddRange(Enumerable.Range(1, _clusterCount).Select(i => i.ToString()));
            foreach (var item in _clusterGroups) { ClusterGroupComboBox.Items.Add(item); }
            if (selectedIndex != null && selectedIndex >= 0 && selectedIndex < _clusterGroups.Count)
            {
                ClusterGroupComboBox.SelectedIndex = selectedIndex;
            }
            else
            {
                ClusterGroupComboBox.SelectedIndex = 0;
            }
        }

        private void ClusterAnnotationComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            _currentClusterAnnotationMode = (ClusterAnnotationMode)ClusterAnnotationComboBox.SelectedIndex;
            AddMainLogEntry($"クラスタリング結果の可視化モードを{_currentClusterAnnotationMode}に変更しました。");
            if (_umapEmbeddings != null && _clusterAssignments != null) { VisualizeClusteringResult(_umapEmbeddings, _clusterAssignments); }
        }

        private void UpdateClusterAnnotationComboBox() // 今は使わない
        {
            ClusterAnnotationComboBox.ItemsSource = new[] { "Clusters", "Tags" };
        }

        private async Task<float[][]> ProcessCSDInAsyncPipeline()
        {   
            _cts = new CancellationTokenSource();

            if (UseGPUCheckBox.IsChecked == true && !_csdModel.IsGpuLoaded) {
                AddMainLogEntry("GPUが有効になっていますが、GPUモデルが読み込まれていません。");
                return Array.Empty<float[]>();
            }
            var usingGPU = UseGPUCheckBox.IsChecked == true && _csdModel.IsGpuLoaded;
            var _CPUConcurrencyLimit_VLMPrediction = (int)VLMConcurrencySlider.Value; // ここは共有する
                    
            // 進捗計算用の総画像数を保持
            var totalImages = _originalImageInfos.Count;
            
            // 処理スピードを表示するためのタスク
            var processingSpeedTask = Task.Run(async () => {
                var stopwatch = Stopwatch.StartNew();
                var progressObservable = Observable.Interval(TimeSpan.FromMilliseconds(_vlmUpdateIntervalMs)).ToAsyncEnumerable(); // ここは共有する
                await foreach (var _ in progressObservable)
                {
                    if (_cts.IsCancellationRequested) break;
                    double loadImagesPerSecond = _loadImgProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                    // double predictImagesPerSecond = _predictProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                    double totalImagesPerSecond = _totalProcessedImagesCount / (stopwatch.ElapsedMilliseconds / 1000.0);
                    double progress = _totalProcessedImagesCount / (double)totalImages;
                    Dispatcher.Invoke(() =>
                    {
                        ProcessingSpeed = $"CSD LoadImg: {loadImagesPerSecond:F1} Total: {totalImagesPerSecond:F1} 枚/秒";
                        UpdateProgressBar(progress);
                        // UpdateUIAfterTagsChange(); // ここは不要
                    });
                }
            });

            // CPU並列度設定の基準は (Environment.ProcessorCount - 2) = 14 (GPU処理の軽いjoytag利用時の最速設定)
            SemaphoreSlim semaphoreCPU = new SemaphoreSlim(_CPUConcurrencyLimit_VLMPrediction); // ここは共有する

            var features = new ConcurrentDictionary<int, float[]>();
            var styleEmbeddings = new ConcurrentDictionary<int, float[]>();
            var contentEmbeddings = new ConcurrentDictionary<int, float[]>();

            // 画像をテンソルに変換するブロック
            var prepareTensorBlock = new TransformBlock<(int index, ImageInfo imageInfo), (int, ImageInfo, DenseTensor<Float16>)?>(
                async input =>
                {
                    await semaphoreCPU.WaitAsync();
                    try
                    {
                        var bitmapImage = new BitmapImage(new Uri(input.imageInfo.ImagePath));
                        var tensor = await _csdModel.PrepareTensor(bitmapImage);
                        Interlocked.Increment(ref _loadImgProcessedImagesCount);
                        return (input.index, input.imageInfo, tensor);
                    }
                    catch (Exception ex)
                    {
                        AddMainLogEntry($"画像の前処理中にエラーが発生しました: {ex.Message}");
                        return null;
                    }
                    finally
                    {
                        semaphoreCPU.Release();
                    }
                },
                new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded }
            );

            // 特徴量を抽出するブロック
            var extractFeaturesBlock = new ActionBlock<(int, ImageInfo, DenseTensor<Float16>)?>(
                async item =>
                {
                    if (!item.HasValue) return;
                    
                    await semaphoreCPU.WaitAsync();
                    try
                    {
                        var (index, imageInfo, tensor) = item.Value;
                        var featureDict = await _csdModel.ExtractFeature(tensor);
                        
                        if (featureDict.TryGetValue("features", out var feature) &&
                            featureDict.TryGetValue("style_output", out var styleEmbedding) &&
                            featureDict.TryGetValue("content_output", out var contentEmbedding))
                        {
                            features.TryAdd(index, feature);
                            styleEmbeddings.TryAdd(index, styleEmbedding);
                            contentEmbeddings.TryAdd(index, contentEmbedding);
                            Interlocked.Increment(ref _totalProcessedImagesCount);
                        }
                        else
                        {
                            AddMainLogEntry($"画像 '{imageInfo.ImagePath}' の特徴量抽出に失敗しました。");
                        }
                    }
                    finally
                    {
                        semaphoreCPU.Release();
                    }
                },
                new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = DataflowBlockOptions.Unbounded }
            );

            // ブロックをリンク
            prepareTensorBlock.LinkTo(extractFeaturesBlock, new DataflowLinkOptions { PropagateCompletion = true });

            // 画像をパイプラインに投入
            for (int i = 0; i < _originalImageInfos.Count; i++)
            {
                // 全量を投入してもパイプラインは順次処理可能だが、ここではキャンセルの反応速度を上げるため、流量をセマフォで制御している（同時に取得するロックは最大で1)
                if (_cts.Token.IsCancellationRequested) break;
                await semaphoreCPU.WaitAsync(); // ロックを取得 or 並列度の上限に達している間はループを止める
                try {
                    await prepareTensorBlock.SendAsync((i, _originalImageInfos[i]));
                } catch (OperationCanceledException) {
                    // キャンセルされた場合はロックを開放してループを抜ける
                    semaphoreCPU.Release();
                    break;
                } finally {
                    semaphoreCPU.Release();
                }
            }

            // パイプラインの完了を通知
            prepareTensorBlock.Complete();
            await extractFeaturesBlock.Completion;

            // 処理スピード計測タスクをキャンセル
            _cts.Cancel();
            await processingSpeedTask;

            // 結果を順序通りに並べ直す
            var orderedFeatures = features.OrderBy(x => x.Key).Select(x => x.Value).ToArray();
            var orderedStyleEmbeddings = styleEmbeddings.OrderBy(x => x.Key).Select(x => x.Value).ToArray();
            var orderedContentEmbeddings = contentEmbeddings.OrderBy(x => x.Key).Select(x => x.Value).ToArray();

            // カウンターのリセット
            _loadImgProcessedImagesCount = 0;
            _predictProcessedImagesCount = 0;
            _totalProcessedImagesCount = 0;
            ProcessingSpeed = "";
            UpdateProgressBar(0);

            // クラスタリングとフィルタリングは外で行うため、特徴量として使用するorderedContentEmbeddingsを返す
            return orderedContentEmbeddings;
        }
        private float[][] PerformUmapEncoding (float[][] embeddings)
        {
            // UMAPの設定
            // var umap = new Umap(dimensions: 2, numberOfNeighbors: 15, random: new Random(42));
            var umap = new Umap();

            // UMAPの実行
            var numberOfEpochs = umap.InitializeFit(embeddings);
            for (var i = 0; i < numberOfEpochs; i++)
            {
                umap.Step();
            }

            // 2次元に縮小された埋め込みを取得
            var reducedEmbeddings = umap.GetEmbedding();

            return reducedEmbeddings;  
        }

        private int[] PerformKMeansClustering(float[][] data, int k)
        {
            // k-meansクラスタリングの実装
            // この実装は簡略化されています。実際のプロジェクトではより堅牢な実装を使用することをお勧めします。
            var random = new Random(42);
            var centroids = data.OrderBy(x => random.Next()).Take(k).ToArray();
            var clusters = new int[data.Length];

            for (int iter = 0; iter < 100; iter++)
            {
                // 各点を最も近いセントロイドに割り当て
                for (int i = 0; i < data.Length; i++)
                {
                    clusters[i] = Enumerable.Range(0, k)
                        .MinBy(j => Distance(data[i], centroids[j]));
                }

                // セントロイドを更新
                for (int j = 0; j < k; j++)
                {
                    var clusterPoints = data.Where((_, i) => clusters[i] == j).ToArray();
                    if (clusterPoints.Any())
                    {
                        centroids[j] = clusterPoints.Aggregate(
                            (a, b) => a.Zip(b, (x, y) => x + y).ToArray()
                        ).Select(sum => sum / clusterPoints.Length).ToArray();
                    }
                }
            }

            AddMainLogEntry($"k-meansクラスタリングが完了しました。クラスタ数: {k}");

            return clusters;
        }

        private float Distance(float[] a, float[] b)
        {
            return (float)Math.Sqrt(a.Zip(b, (x, y) => (x - y) * (x - y)).Sum());
        }

        private BitmapSource _baseClusteringBitmap; // ベースとなる画像を保持するフィールドを追加

        private void VisualizeClusteringResult(float[][] reducedEmbeddings, int[] clusters)
        {
            int width = 400;
            int height = 400;
            var bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);

            float Normalize(float value, float min, float max) => (value - min) / (max - min);

            float minX = reducedEmbeddings.Min(e => e[0]);
            float maxX = reducedEmbeddings.Max(e => e[0]);
            float minY = reducedEmbeddings.Min(e => e[1]);
            float maxY = reducedEmbeddings.Max(e => e[1]);

            var random = new Random(42);
            var clusterColors = Enumerable.Range(0, clusters.Max() + 1)
                .Select(_ => WindowsColor.FromRgb((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256)))
                .ToArray();

            // タグモード時の透明度を計算
            var opacities = new double[reducedEmbeddings.Length];

            int matchingImagesCount = 0;
            switch (_currentClusterAnnotationMode)
            {
                case ClusterAnnotationMode.Tags_And:
                    // AND条件: 選択されたタグの全てを含む画像は不透明(1.0)、それ以外は半透明(0.2)
                    for (int i = 0; i < _originalImageInfos.Count; i++)
                    {
                        bool hasAllSelectedTags = _selectedTags.All(tag => _originalImageInfos[i].Tags.Contains(tag));
                        opacities[i] = hasAllSelectedTags ? 1.0 : 0.2;
                        if (hasAllSelectedTags) matchingImagesCount++;
                    }
                    AddMainLogEntry($"選択されたタグを全て含む画像: {matchingImagesCount}枚");
                    break;

                case ClusterAnnotationMode.Tags_Or:
                    // OR条件: 選択されたタグのいずれかを含む画像は不透明(1.0)、それ以外は半透明(0.2)
                    for (int i = 0; i < _originalImageInfos.Count; i++)
                    {
                        bool hasAnySelectedTag = _selectedTags.Any(tag => _originalImageInfos[i].Tags.Contains(tag));
                        opacities[i] = hasAnySelectedTag ? 1.0 : 0.2;
                        if (hasAnySelectedTag) matchingImagesCount++;
                    }
                    AddMainLogEntry($"選択されたタグのいずれかを含む画像: {matchingImagesCount}枚");
                    break;

                default:
                    // クラスターモードまたはタグが選択されていない場合は全て不透明
                    Array.Fill(opacities, 1.0);
                    break;
            }

            bitmap.Lock();

            // 全ての点を描画（選択された画像以外）
            for (int i = 0; i < reducedEmbeddings.Length; i++)
            {
                int x = (int)(Normalize(reducedEmbeddings[i][0], minX, maxX) * (width - 1));
                int y = (int)(Normalize(reducedEmbeddings[i][1], minY, maxY) * (height - 1));
                WindowsColor baseColor = clusterColors[clusters[i]];

                WindowsColor color = WindowsColor.FromArgb(
                    (byte)(255 * opacities[i]),
                    baseColor.R,
                    baseColor.G,
                    baseColor.B
                );

                for (int dx = -2; dx <= 2; dx++)
                {
                    for (int dy = -2; dy <= 2; dy++)
                    {
                        int px = x + dx;
                        int py = y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height)
                        {
                            int colorData = (color.A << 24) | (color.R << 16) | (color.G << 8) | color.B;
                            bitmap.WritePixels(new Int32Rect(px, py, 1, 1), new[] { colorData }, 4, 0);
                        }
                    }
                }
            }

            bitmap.Unlock();
            ClusteringVisualizationImage.Source = bitmap;

            _baseClusteringBitmap = bitmap;

            // 選択された画像があれば、その点を描画
            DrawSelectedImagePoint(reducedEmbeddings, clusters);
        }

        private void DrawSelectedImagePoint(float[][] reducedEmbeddings, int[] clusters)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null) return;

            var selectedIndex = _originalImageInfos.IndexOf(selectedImage);
            if (selectedIndex < 0) return;

            var bitmap = new WriteableBitmap(_baseClusteringBitmap);
            if (bitmap == null) return;

            int width = bitmap.PixelWidth;
            int height = bitmap.PixelHeight;

            float minX = reducedEmbeddings.Min(e => e[0]);
            float maxX = reducedEmbeddings.Max(e => e[0]);
            float minY = reducedEmbeddings.Min(e => e[1]);
            float maxY = reducedEmbeddings.Max(e => e[1]);

            float Normalize(float value, float min, float max) => (value - min) / (max - min);

            int x = (int)(Normalize(reducedEmbeddings[selectedIndex][0], minX, maxX) * (width - 1));
            int y = (int)(Normalize(reducedEmbeddings[selectedIndex][1], minY, maxY) * (height - 1));
            WindowsColor color = Colors.Black;

            bitmap.Lock();

            // 選択された画像を20x20の円で描画（完全不透明）
            for (int dx = -10; dx <= 10; dx++)
            {
                for (int dy = -10; dy <= 10; dy++)
                {
                    if (dx * dx + dy * dy <= 100)
                    {
                        int px = x + dx;
                        int py = y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height)
                        {
                            int colorData = (255 << 24) | (color.R << 16) | (color.G << 8) | color.B;
                            bitmap.WritePixels(new Int32Rect(px, py, 1, 1), new[] { colorData }, 4, 0);
                        }
                    }
                }
            }

            bitmap.Unlock();

            ClusteringVisualizationImage.Source = bitmap;
        }

        // private void VisualizeClusteringResult(float[][] reducedEmbeddings, int[] clusters)
        // {
        //     var selectedImage = ImageListBox.SelectedItem as ImageInfo;
        //     if (selectedImage == null)
        //     {
        //         AddMainLogEntry("選択された画像がありません。");
        //         return;
        //     }

        //     var selectedIndex = _originalImageInfos.IndexOf(selectedImage);

        //     int width = 400;
        //     int height = 400;
        //     var bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);

        //     // 正規化関数
        //     float Normalize(float value, float min, float max) => (value - min) / (max - min);

        //     // x座標とy座標の最小値と最大値を取得
        //     float minX = reducedEmbeddings.Min(e => e[0]);
        //     float maxX = reducedEmbeddings.Max(e => e[0]);
        //     float minY = reducedEmbeddings.Min(e => e[1]);
        //     float maxY = reducedEmbeddings.Max(e => e[1]);

        //     // クラスターごとの色を生成
        //     var random = new Random(42);
        //     var clusterColors = Enumerable.Range(0, clusters.Max() + 1)
        //         .Select(_ => Color.FromRgb((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256)))
        //         .ToArray();

        //     bitmap.Lock();

        //     for (int i = 0; i < reducedEmbeddings.Length; i++)
        //     {
        //         int x = (int)(Normalize(reducedEmbeddings[i][0], minX, maxX) * (width - 1));
        //         int y = (int)(Normalize(reducedEmbeddings[i][1], minY, maxY) * (height - 1));
        //         Color color = clusterColors[clusters[i]];
        //         // 選択された画像の場合は20x20の円、それ以外は5x5の正方形を描画
        //         if (i == selectedIndex)
        //         {
        //             // 20x20の円を描画
        //             for (int dx = -10; dx <= 10; dx++)
        //             {
        //                 for (int dy = -10; dy <= 10; dy++)
        //                 {
        //                     // 円の方程式: x^2 + y^2 <= r^2
        //                     if (dx * dx + dy * dy <= 100)
        //                     {
        //                         int px = x + dx;
        //                         int py = y + dy;
        //                         if (px >= 0 && px < width && py >= 0 && py < height)
        //                         {
        //                             int colorData = (color.R << 16) | (color.G << 8) | color.B;
        //                             bitmap.WritePixels(new Int32Rect(px, py, 1, 1), new[] { colorData }, 4, 0);
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //         else
        //         {
        //             // 通常の5x5の正方形を描画
        //             for (int dx = -2; dx <= 2; dx++)
        //             {
        //                 for (int dy = -2; dy <= 2; dy++)
        //                 {
        //                     int px = x + dx;
        //                     int py = y + dy;
        //                     if (px >= 0 && px < width && py >= 0 && py < height)
        //                     {
        //                         int colorData = (color.R << 16) | (color.G << 8) | color.B;
        //                         bitmap.WritePixels(new Int32Rect(px, py, 1, 1), new[] { colorData }, 4, 0);
        //                     }
        //                 }
        //             }
        //         }
        //     }

        //     bitmap.Unlock();

        //     ClusteringVisualizationImage.Source = bitmap;
        // }

        private void ApplyClusterFiltering(int[] clusters, int selectedCluster = 0)
        {
            if (selectedCluster == 0)
            {
                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                if (selectedImage == null)
                {
                    AddMainLogEntry("選択された画像がありません。フィルタリングを適用できません。");
                    return;
                }
                var selectedIndex = _originalImageInfos.IndexOf(selectedImage);
                selectedCluster = clusters[selectedIndex];
            }

            // 選択されたクラスターに属する画像のインデックスを保存
            var filteredImageIndices = new List<int>();
            for (int i = 0; i < _originalImageInfos.Count; i++)
            {
                if (clusters[i] == selectedCluster)
                {
                    filteredImageIndices.Add(i);
                }
            }

            AddMainLogEntry($"選択されたクラスターに属する画像の数: {filteredImageIndices.Count}");

            _currentClusterMode = ClusterMode.CSD;

            AddDebugLogEntry($"_clusterAssignments: {string.Join(", ", _clusterAssignments)}");

            // 古いフィルタ機能は削除されました

            // _imageInfosをfilteredImageIndicesにフィルター
            _clusteredImageInfos = filteredImageIndices.Select(index => _originalImageInfos[index]).ToList();
            _imageInfos = new List<ImageInfo>(_clusteredImageInfos);

            UpdateImageList();
            UpdateUIAfterTagsChange();
        }

        // 右ペイン3: ユーザー入力タグの追加
        private void AddTextboxinputButton_Click(object sender, RoutedEventArgs e)
        {
            AddTagFromTextBox(false);
        }

        private void AddAllTextboxinputButton_Click(object sender, RoutedEventArgs e)
        {
            AddTagFromTextBox(true);
        }

        private void AddTagFromTextBox(bool addToAllTags)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            string newTag = SearchTextBox.Text.Trim();
            if (string.IsNullOrEmpty(newTag))
            {
                AddMainLogEntry("タグを入力してください。");
                return;
            }

            // 新しいタグをUserAddedカテゴリに追加
            AddTagToUserAddedCategory(newTag);
            
            // タグ履歴に追加
            AddTagToHistory(newTag);

            if (addToAllTags)
            {
                if (_imageInfos == null || _imageInfos.Count == 0)
                {
                    AddMainLogEntry("対象の画像がありません。");
                    return;
                }

                var addedToImages = new List<ImageInfo>();

                foreach (var imageInfo in _imageInfos)
                {
                    if (!imageInfo.Tags.Contains(newTag))
                    {
                        imageInfo.Tags.Add(newTag);
                        addedToImages.Add(imageInfo);
                    }
                }

                if (addedToImages.Count > 0)
                {
                    var action = new TagGroupAction
                    {
                        DoAction = () =>
                        {
                            AddMainLogEntry($"タグ '{newTag}' を {addedToImages.Count} 個の画像に追加しました。");
                        },
                        UndoAction = () =>
                        {
                            foreach (var imageInfo in addedToImages)
                            {
                                imageInfo.Tags.Remove(newTag);
                            }
                            AddMainLogEntry($"タグ '{newTag}' の追加を {addedToImages.Count} 個の画像から取り消しました。");
                        },
                        Description = $"タグ '{newTag}' を {addedToImages.Count} 個の画像に追加"
                    };

                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                    action.DoAction();
                }
                else
                {
                    AddMainLogEntry($"タグ '{newTag}' は既にすべての画像に存在します。");
                }
            }
            else
            {
                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                if (selectedImage == null)
                {
                    AddMainLogEntry("画像が選択されていません。");
                    return;
                }

                if (!selectedImage.Tags.Contains(newTag))
                {
                    var action = new TagAction
                    {
                        Image = selectedImage,
                        TagInfo = new TagPositionInfo { Tag = newTag, Position = selectedImage.Tags.Count },
                        IsAdd = true,
                        DoAction = () =>
                        {
                            selectedImage.Tags.Add(newTag);
                            AddMainLogEntry($"タグ '{newTag}' を追加しました。");
                        },
                        UndoAction = () =>
                        {
                            selectedImage.Tags.Remove(newTag);
                            AddMainLogEntry($"タグ '{newTag}' の追加を取り消しました。");
                        },
                        Description = $"タグ '{newTag}' を追加"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateUIAfterTagsChange();
                }
                else
                {
                    AddMainLogEntry($"タグ '{newTag}' は既に存在します。");
                }
            }

            SearchTextBox.Clear();
            UpdateSearchedTagsListView();
        }

        private void SearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            // 履歴表示中は検索しない
            if (_isShowingTagHistory && SearchTextBox.Text != "[履歴]")
            {
                _isShowingTagHistory = false;
            }
            
            if (!_isShowingTagHistory)
            {
                // 遅延検索（300ms待機してから検索実行）
                lock (_searchLock)
                {
                    _searchDelayTimer?.Dispose();
                    _searchDelayTimer = new System.Threading.Timer(_ =>
                    {
                        Dispatcher.Invoke(() => UpdateSearchedTagsListView());
                    }, null, 300, System.Threading.Timeout.Infinite);
                }
            }
        }
        
        private void SearchTextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Down:
                    // キーボードナビゲーションで最初の項目に移動
                    if (SearchedTagsListView.Items.Count > 0)
                    {
                        _keyboardNavigationIndex = 0;
                        UpdateKeyboardNavigationDisplay();
                        var item = SearchedTagsListView.ItemContainerGenerator.ContainerFromIndex(0) as ListViewItem;
                        item?.Focus();
                        e.Handled = true;
                    }
                    break;
                    
                case Key.Up:
                    // キーボードナビゲーションで最後の項目に移動
                    if (SearchedTagsListView.Items.Count > 0)
                    {
                        _keyboardNavigationIndex = SearchedTagsListView.Items.Count - 1;
                        UpdateKeyboardNavigationDisplay();
                        var item = SearchedTagsListView.ItemContainerGenerator.ContainerFromIndex(_keyboardNavigationIndex) as ListViewItem;
                        item?.Focus();
                        e.Handled = true;
                    }
                    break;
                    
                case Key.Enter:
                    // キーボードナビゲーション中の項目またはクリック選択された項目を追加
                    string selectedTag = null;
                    if (_keyboardNavigationIndex >= 0 && _keyboardNavigationIndex < SearchedTagsListView.Items.Count)
                    {
                        var item = SearchedTagsListView.Items[_keyboardNavigationIndex];
                        selectedTag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                    }
                    else if (SearchedTagsListView.SelectedItem != null)
                    {
                        var item = SearchedTagsListView.SelectedItem;
                        selectedTag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                    }

                    if (!string.IsNullOrEmpty(selectedTag))
                    {
                        AddTagToCurrentImage(selectedTag);
                        SearchTextBox.Clear();
                        _keyboardNavigationIndex = -1; // リセット
                        
                        if (Keyboard.Modifiers == ModifierKeys.Control)
                        {
                            MoveToNextImage();
                        }
                        else if (Keyboard.Modifiers == (ModifierKeys.Control | ModifierKeys.Shift))
                        {
                            MoveToPreviousImage();
                        }
                        e.Handled = true;
                    }
                    break;
                    
                case Key.Escape:
                    // 検索をクリアしてフォーカスを外す
                    SearchTextBox.Clear();
                    _isShowingTagHistory = false;
                    _keyboardNavigationIndex = -1; // リセット
                    ImageListBox.Focus();
                    e.Handled = true;
                    break;
                    
                case Key.F3:
                    // 検索候補⇔履歴の切り替え
                    if (_isShowingTagHistory)
                    {
                        _isShowingTagHistory = false;
                        UpdateSearchedTagsListView();
                    }
                    else
                    {
                        ShowTagHistory();
                    }
                    e.Handled = true;
                    break;
                    
                case Key.PageUp:
                    MoveToPreviousImage();
                    e.Handled = true;
                    break;
                    
                case Key.PageDown:
                    MoveToNextImage();
                    e.Handled = true;
                    break;
                    
                case Key.Home:
                    MoveToFirstImage();
                    e.Handled = true;
                    break;
                    
                case Key.End:
                    MoveToLastImage();
                    e.Handled = true;
                    break;
            }
        }

        private void SearchedTagsListView_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            var listView = sender as ListView;
            if (listView == null) return;

            switch (e.Key)
            {
                case Key.Enter:
                    // キーボードナビゲーション中の項目またはクリック選択された項目を追加
                    string selectedTag = null;
                    if (_keyboardNavigationIndex >= 0 && _keyboardNavigationIndex < listView.Items.Count)
                    {
                        var item = listView.Items[_keyboardNavigationIndex];
                        selectedTag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                    }
                    else if (listView.SelectedItem != null)
                    {
                        var item = listView.SelectedItem;
                        selectedTag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                    }

                    if (!string.IsNullOrEmpty(selectedTag))
                    {
                        AddTagToCurrentImage(selectedTag);
                        SearchTextBox.Clear();
                        SearchTextBox.Focus();
                        _keyboardNavigationIndex = -1; // リセット
                        
                        if (Keyboard.Modifiers == ModifierKeys.Control)
                        {
                            MoveToNextImage();
                        }
                        else if (Keyboard.Modifiers == (ModifierKeys.Control | ModifierKeys.Shift))
                        {
                            MoveToPreviousImage();
                        }
                        e.Handled = true;
                    }
                    break;
                    
                case Key.Escape:
                    // フォーカスをSearchTextBoxに戻し、キーボードナビゲーションをリセット
                    _keyboardNavigationIndex = -1;
                    UpdateKeyboardNavigationDisplay();
                    SearchTextBox.Focus();
                    e.Handled = true;
                    break;
                    
                case Key.Up:
                    if (_keyboardNavigationIndex > 0)
                    {
                        _keyboardNavigationIndex--;
                        UpdateKeyboardNavigationDisplay();
                        e.Handled = true;
                    }
                    else if (_keyboardNavigationIndex == 0)
                    {
                        // 最初の項目で上キーが押された場合、SearchTextBoxにフォーカスを戻す
                        _keyboardNavigationIndex = -1;
                        UpdateKeyboardNavigationDisplay();
                        SearchTextBox.Focus();
                        e.Handled = true;
                    }
                    break;
                    
                case Key.Down:
                    if (_keyboardNavigationIndex < listView.Items.Count - 1)
                    {
                        if (_keyboardNavigationIndex == -1)
                        {
                            _keyboardNavigationIndex = 0;
                        }
                        else
                        {
                            _keyboardNavigationIndex++;
                        }
                        UpdateKeyboardNavigationDisplay();
                        e.Handled = true;
                    }
                    break;
            }
        }

        private void UpdateKeyboardNavigationDisplay()
        {
            // 現在のキーボードナビゲーション状態を視覚的に反映
            for (int i = 0; i < SearchedTagsListView.Items.Count; i++)
            {
                var container = SearchedTagsListView.ItemContainerGenerator.ContainerFromIndex(i) as ListViewItem;
                if (container != null)
                {
                    if (i == _keyboardNavigationIndex && !container.IsSelected)
                    {
                        // キーボードナビゲーション中かつ非選択状態の項目は青背景
                        // 選択状態（緑）の項目は緑を優先
                        container.Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Colors.LightBlue);
                    }
                    else if (i != _keyboardNavigationIndex && !container.IsSelected)
                    {
                        // 通常の項目は背景をクリア（選択状態の緑はXAMLで処理）
                        container.ClearValue(ListViewItem.BackgroundProperty);
                    }
                    // 選択状態の項目（緑）はXAMLのTriggerで処理され、こちらでは触らない
                }
            }
        }

        private int GetTagCount(string tag)
        {
            // JSONファイルから読み込んだカウント（Danbooru頻度）を取得
            if (_jsonTagCounts != null && _jsonTagCounts.ContainsKey(tag))
            {
                return _jsonTagCounts[tag];
            }
            
            return 0;
        }

        private void SearchTargetComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateSearchedTagsListView();
        }

        private void SearchOptionComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateSearchedTagsListView();
        }

        private void HideZeroCountTagsCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            UpdateSearchedTagsListView();
        }

        private void ListView_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            var listView = sender as ListView;
            if (listView == null) return;

            var gridView = listView.View as GridView;
            if (gridView == null) return;

            // ListViewの実際の幅を取得（スクロールバーの幅を考慮）
            double workingWidth = listView.ActualWidth - SystemParameters.VerticalScrollBarWidth - 20; // 20はパディング
            
            // 最小幅の設定
            const double minTagWidth = 150;
            const double countWidth = 60;
            const double starWidth = 40; // 星マーク用（AllTagsListViewの場合）
            
            if (gridView.Columns.Count == 1)
            {
                // TagListView（タグのみ）
                double tagWidth = Math.Max(minTagWidth, workingWidth);
                gridView.Columns[0].Width = tagWidth;
            }
            else if (gridView.Columns.Count == 2)
            {
                // SearchedTagsListView（タグ、カウント）
                double tagWidth = Math.Max(minTagWidth, workingWidth - countWidth);
                gridView.Columns[0].Width = tagWidth;
                // カウント列は固定幅のまま
            }
            else if (gridView.Columns.Count == 3)
            {
                // AllTagsListView（タグ、カウント＋星）
                double tagWidth = Math.Max(minTagWidth, workingWidth - countWidth - starWidth);
                gridView.Columns[0].Width = tagWidth;
                // カウントと星列は固定幅のまま
            }
        }

        private void UpdateSearchedTagsListView()
        {
            // デバッグログを簡潔に
            AddDebugLogEntry($"UpdateSearchedTagsListView: '{SearchTextBox.Text}'");

            // 検索結果が変更されるため、キーボードナビゲーション状態をリセット
            _keyboardNavigationIndex = -1;

            string searchText = SearchTextBox.Text.ToLower();
            SearchedTagsListView.SelectionChanged -= SearchedTagsListView_SelectionChanged;

            // 辞書検索の場合は最小2文字から検索（パフォーマンス対策）
            bool isDictionarySearch = SearchTargetComboBox.SelectedIndex == 2;
            int minSearchLength = isDictionarySearch ? 2 : 1;

            if (!string.IsNullOrEmpty(searchText) && searchText.Length >= minSearchLength)
            {
                IEnumerable<string> searchSource;
                switch (SearchTargetComboBox.SelectedIndex)
                {
                    case 0: // AllTags
                        searchSource = _allTags.Keys;
                        break;
                    case 1: // OriginalImageTags
                        searchSource = _originalImageInfos?.SelectMany(info => info.Tags).Distinct() ?? Enumerable.Empty<string>();
                        break;
                    case 2: // BooruTags (Dictionary)
                        searchSource = _cachedDictionaryTags ?? Enumerable.Empty<string>();
                        break;
                    default:
                        searchSource = Enumerable.Empty<string>();
                        break;
                }

                Func<string, bool> matchPredicate;
                switch (SearchOptionComboBox.SelectedIndex)
                {
                    case 0: // Partial Matc
                        matchPredicate = tag => tag.ToLower().Contains(searchText);
                        break;
                    case 1: // Prefix Match
                        matchPredicate = tag => tag.ToLower().StartsWith(searchText);
                        break;
                    case 2: // Phrase Match
                        var searchPhrases = searchText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        matchPredicate = tag => searchPhrases.All(phrase => tag.ToLower().Contains(phrase));
                        break;
                    default:
                        matchPredicate = _ => false;
                        break;
                }

                // 並列処理でパフォーマンス向上（大規模データセットの場合）
                var matchingTags = isDictionarySearch && _cachedDictionaryTags?.Count > 10000
                    ? searchSource.AsParallel()
                        .Where(matchPredicate)
                        .Take(200)  // 並列処理では少し多めに取得
                        .Select(tag => new SearchTagInfo 
                        { 
                            Tag = tag, 
                            Count = GetTagCount(tag) 
                        })
                        .OrderByDescending(info => info.Count)
                        .ThenBy(info => info.Tag)
                        .Take(100)
                        .ToList()
                    : searchSource
                        .Where(matchPredicate)
                        .Select(tag => new SearchTagInfo 
                        { 
                            Tag = tag, 
                            Count = GetTagCount(tag) 
                        })
                        .OrderByDescending(info => info.Count)
                        .ThenBy(info => info.Tag)
                        .Take(100)
                        .ToList();

                // カウント0のタグを非表示にするオプションが有効な場合、フィルタリング
                if (HideZeroCountTagsCheckBox?.IsChecked == true)
                {
                    matchingTags = matchingTags.Where(t => t.Count > 0).ToList();
                }

                // デバッグログは件数のみ表示（詳細表示は重いため省略）
                AddDebugLogEntry($"matchingTags: {matchingTags.Count}件");

                SearchedTagsListView.ItemsSource = matchingTags;
                SearchedTagsListView.SelectedItems.Clear();

                var tagsToSelect = matchingTags.Where(tagInfo => _selectedTags.Contains(tagInfo.Tag)).ToList();
                foreach (var tagInfo in tagsToSelect)
                {
                    SearchedTagsListView.SelectedItems.Add(tagInfo);
                }

                if (matchingTags.Count == 100)
                {
                    AddMainLogEntry("検索結果が100件を超えています。最初の100件のみ表示しています。");
                }
            }
            else
            {
                // 検索バーが空の場合、SearchedTagsListViewを空にする
                SearchedTagsListView.ItemsSource = null;
            }

            SearchedTagsListView.SelectionChanged += SearchedTagsListView_SelectionChanged;
        }

        private void SearchedTagsListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            AddDebugLogEntry("SearchedTagsListView_SelectionChanged");
            if (_isUpdatingSelection) return;

            _isUpdatingSelection = true;
            
            foreach (var item in e.RemovedItems)
            {
                string tag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                _selectedTags.Remove(tag);
            }

            foreach (var item in e.AddedItems)
            {
                string tag = item is SearchTagInfo tagInfo ? tagInfo.Tag : item.ToString();
                _selectedTags.Add(tag);
            }

            UpdateUIAfterTagSelectionChange();
            _isUpdatingSelection = false;
        }

        // 選択されたタグリストの更新
        private void UpdateSelectedTagsListBox()
        {
            SelectedTagsListBox.ItemsSource = _selectedTags.ToList();
        }


        /*
        ドラッグアンドドロップ関連の操作
        */

        private ListViewItem _draggedItem;

        private void TagListView_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            AddDebugLogEntry("TagListView_PreviewMouseLeftButtonDown");
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                _startPoint = e.GetPosition(null);
                _draggedItem = FindAncestor<ListViewItem>((DependencyObject)e.OriginalSource);
                _isDragging = false;
            }
        }

        private void TagListView_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && _startPoint.HasValue && _draggedItem != null)
            {
                WindowsPoint currentPoint = e.GetPosition(null);
                Vector diff = _startPoint.Value - currentPoint;

                if (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance ||
                    Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance)
                {
                    _isDragging = true;
                    ListView listView = sender as ListView;
                    ListViewItem listViewItem = _draggedItem;
                    
                    if (listViewItem != null && listViewItem.Content is string tagData)
                    {
                        DataObject dragData = new DataObject("TagData", tagData);
                        DragDrop.DoDragDrop(listViewItem, dragData, DragDropEffects.Move);
                    }
                }
            }
        }

        private void TagListView_Drop(object sender, DragEventArgs e)
        {
            AddDebugLogEntry("TagListView_Drop");
            if (e.Data.GetDataPresent("TagData"))
            {
                string droppedTag = (string)e.Data.GetData("TagData");
                ListViewItem targetItem = FindAncestor<ListViewItem>((DependencyObject)e.OriginalSource);
                AddDebugLogEntry($"droppedTag: {droppedTag}");
                AddDebugLogEntry($"targetItem: {targetItem}");

                if (targetItem != null && targetItem.Content is string)
                {
                    int targetIndex = TagListView.Items.IndexOf(targetItem.Content);
                    int sourceIndex = TagListView.Items.IndexOf(droppedTag);

                    AddDebugLogEntry($"sourceIndex: {sourceIndex}, targetIndex: {targetIndex}");

                    if (sourceIndex != -1 && targetIndex != -1 && sourceIndex != targetIndex)
                    {
                        var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                        AddDebugLogEntry($"selectedImage: {selectedImage}");
                        if (selectedImage != null)
                        {
                            var action = new TagGroupAction
                            {
                                Image = selectedImage,
                                TagInfos = new List<TagPositionInfo> 
                                { 
                                    new TagPositionInfo { Tag = droppedTag, Position = sourceIndex },
                                    new TagPositionInfo { Tag = droppedTag, Position = targetIndex }
                                },
                                IsAdd = false,
                                DoAction = () =>
                                {
                                    string movedTag = selectedImage.Tags[sourceIndex];
                                    selectedImage.Tags.RemoveAt(sourceIndex);
                                    selectedImage.Tags.Insert(targetIndex, movedTag);
                                    AddMainLogEntry($"タグ '{droppedTag}' を移動しました: {sourceIndex} -> {targetIndex}");
                                },
                                UndoAction = () =>
                                {
                                    string movedTag = selectedImage.Tags[targetIndex];
                                    selectedImage.Tags.RemoveAt(targetIndex);
                                    selectedImage.Tags.Insert(sourceIndex, movedTag);
                                    AddMainLogEntry($"タグ '{droppedTag}' の移動を元に戻しました: {targetIndex} -> {sourceIndex}");
                                },
                                Description = $"タグ '{droppedTag}' を移動: {sourceIndex} -> {targetIndex}"
                            };

                            action.DoAction();
                            _undoStack.Push(action);
                            _redoStack.Clear();
                            UpdateUIAfterTagsChange();
                        }
                    }
                }
            }
        }

        private void TagListView_PreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (!_isDragging && _draggedItem != null)
            {
                string tag = _draggedItem.Content as string;
                if (tag != null)
                {
                    if (_selectedTags.Contains(tag))
                    {
                        _selectedTags.Remove(tag);
                        AddDebugLogEntry($"タグ '{tag}' の選択を解除しました。");
                    }
                    else
                    {
                        _selectedTags.Add(tag);
                        AddDebugLogEntry($"タグ '{tag}' を選択しました。");
                    }
                    UpdateUIAfterTagSelectionChange();
                }
            }
            _draggedItem = null;
            _startPoint = null;
            _isDragging = false;
        }

        private void AllTagsListView_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            AddDebugLogEntry("AllTagsListView_PreviewMouseLeftButtonDown");
            // _startPoint = e.GetPosition(null);
            // var item = (e.OriginalSource as FrameworkElement)?.DataContext;
            // if (item != null)
            // {
            //     _draggedItem = (sender as ListView)?.ItemContainerGenerator.ContainerFromItem(item) as ListViewItem;
            // }
        }

        private void AllTagsListView_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            AddDebugLogEntry("AllTagsListView_PreviewMouseMove");
            // if (_startPoint == null || _draggedItem == null) return;

            // Point currentPosition = e.GetPosition(null);
            // Vector diff = currentPosition - _startPoint.Value;

            // if (e.LeftButton == MouseButtonState.Pressed &&
            //     (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance + 2 ||
            //      Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance + 2))
            // {
            //     DragDrop.DoDragDrop(_draggedItem, _draggedItem.DataContext, DragDropEffects.Copy);
            //     _startPoint = null;
            //     _draggedItem = null;
            // }
        }

        private static T FindAncestor<T>(DependencyObject current) where T : DependencyObject
        {
            do
            {
                if (current is T)
                {
                    return (T)current;
                }
                current = VisualTreeHelper.GetParent(current);
            }
            while (current != null);
            return null;
        }

        /*
        ここまでドラッグアンドドロップ関連メソッド
        ここからVLM関連メソッド
        */

        private async void VLMModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (VLMModelComboBox.SelectedItem is string selectedModel)
            {
                var modelInfo = _vlmModels.First(m => m.Name == selectedModel);
                UpdateThresholds(modelInfo.GeneralThreshold, DefaultCharacterThreshold);
                await LoadVLMModel(selectedModel);

                // 設定を保存
                SaveSettings();
            }
        }

        private void UpdateThresholds(double generalThreshold, double characterThreshold)
        {
            GeneralThresholdSlider.Value = generalThreshold;
            CharacterThresholdSlider.Value = characterThreshold;
        }

        private void GeneralThresholdSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // スライダーの値が変更されたときの処理
            // 必要に応じて、この値をVLMPredictorに渡す
        }

        private void CharacterThresholdSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // スライダーの値が変更されたときの処理
            // 必要に応じて、この値をVLMPredictorに渡す
        }

        private async void UseGPUCheckBox_Checked(object sender, RoutedEventArgs e)
        {
            if (VLMConcurrencySlider != null)
            {
                VLMConcurrencySlider.Value = Environment.ProcessorCount - 2;
            }

            if (_isInitializeSuccess)
            {
                await LoadVLMModel(VLMModelComboBox.SelectedItem as string, UseGPUCheckBox.IsChecked ?? false);
            }
        }

        private void InitializeVLMPredictor()
        {
            AddDebugLogEntry("InitializeVLMPredictor");
            _vlmPredictor = new VLMPredictor();
            _vlmPredictor.LogUpdated += UpdateVLMLog;
        }

        private async Task LoadVLMModel(string modelName, bool useGpu = true)
        {
            if (_isLoadingVLMModel) { return; }
            _isLoadingVLMModel = true;

            try
            {
                AddMainLogEntry($"VLMモデル '{modelName}' の読み込みを開始します。");
                await _vlmPredictor.LoadModel(modelName, useGpu);
                if (_vlmPredictor.IsGpuLoaded)
                {
                    AddMainLogEntry("GPUを使用します");
                }
                else
                {
                    if (UseGPUCheckBox.IsChecked == true) 
                    { 
                        AddMainLogEntry("GPUの適用に失敗しました。CPUにフォールバックします。");
                        UseGPUCheckBox.IsChecked = false; 
                    }
                    AddMainLogEntry("CPUを使用します");
                }
                AddMainLogEntry($"VLMモデル '{modelName}' の読み込みが完了しました。");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"VLMモデルの読み込みに失敗しました: {ex.Message}");
                AddMainLogEntry($"VLMモデルの読み込みに失敗しました: {ex.Message}");
            }
            finally
            {
                _isLoadingVLMModel = false;

                if (!_isInitializeSuccess)
                {
                    _vlmPredictor.Dispose(); // 初期化途中のモデルを解放
                    AddMainLogEntry("VLMモデルをオフロードしました");
                }
            }
        }

        // VLM推論(単体)を実行するボタンのクリックイベントハンドラ
        private async void VLMPredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isAsyncProcessing) { return; }
            _isAsyncProcessing = true;

            try
            {
                // ボタンを無効化して、処理中であることを示す
                VLMPredictButton.IsEnabled = false;

                await _vlmPredictor.LoadModel(VLMModelComboBox.SelectedItem as string, UseGPUCheckBox.IsChecked ?? false);

                // キャンセルトークンソースを作成
                _cts = new CancellationTokenSource();
                
                // 選択された画像を取得
                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                if (selectedImage == null)
                {
                    AddMainLogEntry("画像が選択されていません。");
                    _isAsyncProcessing = false;
                    return;
                }

                // 非同期でPredictVLMTagsを呼び出す
                var predictedTags = await PredictVLMTagsAsync(selectedImage, _cts.Token);
                
                if (selectedImage != null && predictedTags.Any())
                {
                    // 既存のタグと重複しないタグを抽出
                    var newTags = predictedTags.Except(selectedImage.Tags).ToList();
                    
                    if (newTags.Any())
                    {
                        // 新しいタグを追加するアクションを作成
                        var action = new TagGroupAction
                        {
                            Image = selectedImage,
                            TagInfos = newTags.Select(tag => new TagPositionInfo { Tag = tag, Position = selectedImage.Tags.Count }).ToList(),
                            IsAdd = true,
                            DoAction = () =>
                            {
                                foreach (var tagInfo in newTags)
                                {
                                    selectedImage.Tags.Add(tagInfo);
                                }
                                AddMainLogEntry($"VLM推論により{newTags.Count}個の新しいタグを追加しました");
                            },
                            UndoAction = () =>
                            {
                                for (int i = 0; i < newTags.Count; i++)
                                {
                                    selectedImage.Tags.RemoveAt(selectedImage.Tags.Count - 1);
                                }
                                AddMainLogEntry($"VLM推論により追加された{newTags.Count}個のタグを削除しました");
                            },
                            Description = $"VLM推論により{newTags.Count}個のタグを追加"
                        };

                        // アクションを実行し、Undoスタックに追加
                        action.DoAction();
                        _undoStack.Push(action);
                        _redoStack.Clear();
                        UpdateUIAfterTagsChange();
                    }
                    else
                    {
                        AddMainLogEntry("VLM推論により新しいタグは見つかりませんでした");
                    }
                }
            }
            catch (Exception ex)
            {
                // エラーメッセージをログに記録
                AddMainLogEntry($"VLM推論中にエラーが発生しました: {ex.Message}");
                MessageBox.Show($"エラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                // 処理が完了したらボタンを再度有効化
                VLMPredictButton.IsEnabled = true;

                _cts = null;
                _vlmPredictor.Dispose();

                _isAsyncProcessing = false;
            }
        }

        private async Task<BitmapImage> ConvertToImage(BitmapSource bitmapSource)
        {
            if (bitmapSource == null) return null;

            var bitmapImage = new BitmapImage();
            var bitmapEncoder = new PngBitmapEncoder();
            bitmapEncoder.Frames.Add(BitmapFrame.Create(bitmapSource));

            using (var stream = new MemoryStream())
            {
                bitmapEncoder.Save(stream);
                stream.Seek(0, SeekOrigin.Begin);

                bitmapImage.BeginInit();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.StreamSource = stream;
                bitmapImage.EndInit();
                bitmapImage.Freeze(); // UIスレッド以外でも使用可能にする
            }

            return bitmapImage;
        }

        private async void PredictSelectedRegion_Click(object sender, RoutedEventArgs e)
        {
            if (_isAsyncProcessing) return;
            _isAsyncProcessing = true;

            try
            {
                if (SelectionRectangle.Visibility != Visibility.Visible)
                {
                    AddMainLogEntry("範囲が選択されていません。");
                    return;
                }

                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                if (selectedImage == null)
                {
                    AddMainLogEntry("画像が選択されていません。");
                    return;
                }

                VLMPredictButton.IsEnabled = false;
                _cts = new CancellationTokenSource();

                await _vlmPredictor.LoadModel(VLMModelComboBox.SelectedItem as string, UseGPUCheckBox.IsChecked ?? false);

                // 選択範囲の画像を取得
                var croppedImage = await GetSelectedRegion();
                if (croppedImage == null)
                {
                    AddMainLogEntry("選択範囲の取得に失敗しました。");
                    return;
                }

                // BitmapSourceからBitmapImageに変換
                var bitmapImage = await ConvertToImage(croppedImage);
                if (bitmapImage == null)
                {
                    AddMainLogEntry("画像の変換に失敗しました。");
                    return;
                }

                // 変換後のBitmapImageを使用してVLM推論
                var predictedTags = await Task.Run(() =>
                {
                    return PredictVLMFromTensor(
                        _vlmPredictor.PrepareTensor(bitmapImage)!,
                        _cts.Token
                    );
                }, _cts.Token);

                if (predictedTags.Any())
                {
                    var newTags = predictedTags.Except(selectedImage.Tags).ToList();
                    if (newTags.Any())
                    {
                        var action = CreateAddTagsAction(selectedImage, newTags);
                        action.DoAction();
                        _undoStack.Push(action);
                        _redoStack.Clear();
                        UpdateUIAfterTagsChange();
                        AddMainLogEntry($"選択範囲から{newTags.Count}個のタグを追加しました。");
                    }
                    else
                    {
                        AddMainLogEntry("選択範囲から新しいタグは見つかりませんでした。");
                    }
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"選択範囲のVLM推論中にエラーが発生しました: {ex.Message}");
            }
            finally
            {
                VLMPredictButton.IsEnabled = true;
                _cts = null;
                _vlmPredictor.Dispose();
                _isAsyncProcessing = false;
            }
        }

        private async void VLMPredictAllButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象画像がありません。");
                _isAsyncProcessing = false;
                return;
            }
            if (ConfirmCheckBox.IsChecked == true)
            {
                var result = MessageBox.Show("すべての画像に対してVLM推論を実行しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result != MessageBoxResult.Yes)
                {
                    AddMainLogEntry("VLM推論がキャンセルされました。");
                    _isAsyncProcessing = false;
                    return;
                }
            }
            if (_isAsyncProcessing) { return; }
            _isAsyncProcessing = true;
            try
            {
                // ボタンを無効化して、処理中であることを示す
                VLMPredictAllButton.IsEnabled = false;

                await _vlmPredictor.LoadModel(VLMModelComboBox.SelectedItem as string, UseGPUCheckBox.IsChecked ?? false);
                
                AddMainLogEntry("すべての画像に対してVLM推論を開始します");
                
                await ProcessVLMPredictAllInAsyncPipeline();
                
                AddMainLogEntry("すべての画像に対するVLM推論が完了しました");
            }
            catch (AggregateException ex) // awaitのキャンセルはAggregateExceptionで返される
            {
                AddMainLogEntry($"VLM推論がキャンセルされました: {ex.Message}");
                MessageBox.Show("推論のキャンセルが完了しました", "キャンセル完了", MessageBoxButton.OK, MessageBoxImage.None);
            }
            catch (InvalidOperationException ex)
            {
                AddMainLogEntry($"VLM推論エラー: {ex.Message}");
                MessageBox.Show(ex.Message, "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"VLM推論中にエラーが発生しました: {ex.Message}");
                MessageBox.Show($"エラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                // 処理が完了したらボタンを再度有効化
                _isAsyncProcessing = false;
                VLMPredictAllButton.IsEnabled = true;

                // 多少冗長でもエラーの後始末で整合性を取るためにUI更新
                UpdateProgressBar(0);
                UpdateUIAfterTagsChange();
                // 処理が完了したら処理速度表示をクリア
                ProcessingSpeed = "";

                _cts = null;
                _vlmPredictor.Dispose();
            }
        }

        private async Task ProcessVLMPredictAllInAsyncPipeline()
        {   
            _cts = new CancellationTokenSource();

            if (UseGPUCheckBox.IsChecked == true && !_vlmPredictor.IsGpuLoaded) {
                AddMainLogEntry("GPUが有効になっていますが、GPUモデルが読み込まれていません。");
                return;
            }
            var usingGPU = UseGPUCheckBox.IsChecked == true && _vlmPredictor.IsGpuLoaded;
            
            var pipelineStages = new List<PipelineStage>{

                // 画像を非同期にロードし、テンソルを準備する
                new PipelineStage(async i => {
                    if (i is null || i!.Item2 is null) return null;
                    DenseTensor<float>? tensor = await Task.Run(() => _vlmPredictor.PrepareTensor(i!.Item2));
                    if (tensor is null) return null;
                    return (i!.Item1, tensor);
                }),
                
                // テンソルを使用してタグを予測する
                new PipelineStage(async i => {
                    if (i is null || i!.Item2 is null) return null;
                    return (i!.Item1, await PredictVLMFromTensor(i!.Item2, _cts.Token));
                }, isGpuStage: usingGPU),

                // 予測されたタグを処理する
                new PipelineStage(async i => {
                    if (i is null || i!.Item2 is null) return null;
                    ProcessPredictedTags(i!.Item1, i!.Item2);
                    return i!.Item1;
                })
            };

            var asyncPipelineService = new AsyncPipelineService(
                cpuConcurrencyLimit: (int)VLMConcurrencySlider.Value, 
                gpuConcurrencyLimit: (int)VLMConcurrencySlider.Value,
                pipelineStages
            );
            asyncPipelineService.LogUpdated += UpdatePipelineLog;

            var totalImages = _imageInfos.Count;
            var stopwatch = Stopwatch.StartNew();
            asyncPipelineService.ProgressUpdated += (counters) => {
                double loadedImagesPerSecond = counters[0].Value / (stopwatch.ElapsedMilliseconds / 1000.0);
                double predictImagesPerSecond = counters[1].Value / (stopwatch.ElapsedMilliseconds / 1000.0);
                double totalImagesPerSecond = counters[2].Value / (stopwatch.ElapsedMilliseconds / 1000.0);
                ProcessingSpeed = $"VLM - Load: {loadedImagesPerSecond:F1} Predict: {predictImagesPerSecond:F1} Complete: {totalImagesPerSecond:F1} 枚/秒";
                Dispatcher.Invoke(() => {
                    UpdateProgressBar(counters[2].Value / (double)totalImages);
                    UpdateUIAfterTagsChange();
                });
            };

            try {
                await asyncPipelineService.ProcessAsync<(ImageInfo, BitmapImage), ImageInfo>(
                    LoadAllBitmapImagesAsync(),
                    _cts);
            } catch (OperationCanceledException) {
                AddMainLogEntry("VLM推論がキャンセルされました");
            } finally {
                AddMainLogEntry("VLM推論が完了しました");
                stopwatch.Stop();
                UpdateUIAfterTagsChange();
                UpdateProgressBar(0);
                ProcessingSpeed = "";

                asyncPipelineService.LogUpdated -= UpdatePipelineLog;
            }
        }

        private async IAsyncEnumerable<(ImageInfo imageInfo, BitmapImage bitmap)> LoadAllBitmapImagesAsync()
        {
            foreach (var imageInfo in _imageInfos) {
                yield return (imageInfo, await Task.Run(() => LoadImageForVLMPrediction(imageInfo.ImagePath)));
            }
            AddMainLogEntry("すべての画像を読み込みました(全画像VLM推論用)");
        }

        private BitmapImage LoadImageForVLMPrediction(string imagePath)
        {
            try
            {
                byte[] imageData;
                using (FileStream fs = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    imageData = new byte[fs.Length];
                    // ファイル読み込みをより堅牢に (オプション)
                    int totalBytesRead = 0;
                    while(totalBytesRead < fs.Length)
                    {
                        int bytesRead = fs.Read(imageData, totalBytesRead, (int)fs.Length - totalBytesRead);
                        if (bytesRead == 0) break; // 予期せぬストリーム終端
                        totalBytesRead += bytesRead;
                    }
                     if (totalBytesRead != fs.Length) {
                         Dispatcher.Invoke(() => AddMainLogEntry($"{DateTime.Now:HH:mm:ss} - Warning: Incomplete read for image {imagePath}"));
                     }
                }

                using (MemoryStream ms = new MemoryStream(imageData))
                {
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    // ★ PreservePixelFormat を削除し、IgnoreColorProfile のみ残す
                    bitmapImage.CreateOptions = BitmapCreateOptions.IgnoreColorProfile; 
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.StreamSource = ms;
                    bitmapImage.EndInit(); // ★ ここでまだエラーが出るか？
                    bitmapImage.Freeze();
                    return bitmapImage;
                }
            }
            catch (NotSupportedException nsex) // 特定の例外をキャッチしてログ記録
            {
                 Dispatcher.Invoke(() => AddMainLogEntry($"{DateTime.Now:HH:mm:ss} - NotSupportedException loading image {imagePath}: {nsex.Message}"));
                 return null; // 問題のある画像はスキップする？
            }
            catch (Exception ex)
            {
                // ★ エラー発生時のファイルパスを確実にログ記録
                Dispatcher.Invoke(() => AddMainLogEntry($"{DateTime.Now:HH:mm:ss} - Exception loading image {imagePath}: {ex.GetType().Name} - {ex.Message}"));
                // デバッグ用にスタックトレースも記録すると役立つ場合がある
                // Dispatcher.Invoke(() => LogEntries.Insert(0, $"Stack Trace: {ex.StackTrace}")); 
                // return null; // スキップする場合
                throw; // エラーで処理を中断する場合
            }
        }

        private void ProcessPredictedTags(ImageInfo imageInfo, List<string> predictedTags)
        {
            var newTags = predictedTags.Except(imageInfo.Tags).ToList();
            if (newTags.Any())
            {
                var action = CreateAddTagsAction(imageInfo, newTags);
                action.DoAction();
                lock (_undoStack)
                {
                _undoStack.Push(action);
                }
            }
        }

        private async Task<List<string>> PredictVLMFromTensor(DenseTensor<float> tensor, CancellationToken cancellationToken)
        {
            (float generalThreshold, float characterThreshold) = await Dispatcher.InvokeAsync(() => 
                ((float)GeneralThresholdSlider.Value, (float)CharacterThresholdSlider.Value)
            );

            var (generalTags, rating, characters, allTags) = Task.Run(() => {
                try {
                    return _vlmPredictor.Predict(
                        tensor,
                        generalThreshold,
                        false,
                        characterThreshold,
                        false
                    );
                } catch (Exception ex) {
                    AddMainLogEntry($"画像の読み込み中にエラーが発生しました: {ex.Message}");
                    _vlmErrorLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} 画像読み込みエラー: {ex.Message}");
                    // tag更新をせずに継続
                    return (string.Empty, new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
                }
            }, cancellationToken).Result;

            // original author に確認のため、古い挙動をコメントで保存
            // // generalTagsが空の場合は空のリストを返す
            // if (string.IsNullOrWhiteSpace(generalTags)) { return new List<string>(); }
            // // generalTagsとcharactersを結合して返す
            // var predictedTags = generalTags.Split(',').Select(t => t.Trim()).ToList();
            // predictedTags.AddRange(characters.Keys);
            // return predictedTags;
            
            var predictedTags = generalTags.Split(',').Select(t => t.Trim()).ToList();
            predictedTags.AddRange(characters.Keys);

            return predictedTags;
        }

        // VLM推論
        private async Task<List<string>> PredictVLMTagsAsync(ImageInfo imageInfo, CancellationToken cancellationToken)
        {
            AddMainLogEntry("VLM推論を開始します(単体版)");
            var predictedTags = await Task.Run(() => { 
                return PredictVLMFromTensor(
                    _vlmPredictor.PrepareTensor(LoadImageForVLMPrediction(imageInfo.ImagePath))!,
                    cancellationToken
                );
            }, cancellationToken);
            await Dispatcher.InvokeAsync(() => AddMainLogEntry($"VLM推論結果: {string.Join(", ", predictedTags)}"));
            return predictedTags;
        }

        // VLMログの更新
        private void UpdateVLMLog(object? sender, string log) {
            Dispatcher.Invoke(() => {
                AddDebugLogEntry("UpdateVLMLog");
                _vlmLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} - {log}");
            });
        }

        // パイプラインのログを更新する
        private void UpdatePipelineLog(object? sender, string log)
        {
            Dispatcher.Invoke(() => {
                AddDebugLogEntry("UpdatePipelineLog");
                _pipelineLogQueue.Enqueue($"{DateTime.Now:HH:mm:ss} - {log}");
            });
        }

        /*
        ここまでVLM関連メソッド
        ここからタグカテゴリ関連メソッド
        */

        private void LoadTagCategories()
        {
            // JSONタグカウントをクリア（新規読み込み時）
            _jsonTagCounts.Clear();
            
            _defaultTagCategories = LoadCategoriesFromFiles(DefaultCategoryFiles);
            _customTagCategories = LoadCategoriesFromFiles(CustomCategoryFiles);

            UpdateTagCategories();
            
            // 辞書タグをキャッシュ（検索高速化のため）
            InitializeDictionaryTagsCache();
        }
        
        private void InitializeDictionaryTagsCache()
        {
            _cachedDictionaryTags = _tagCategories.Values
                .SelectMany(category => category.Tags.Keys)
                .Distinct()
                .OrderBy(tag => tag)
                .ToList();
            AddMainLogEntry($"辞書タグキャッシュを初期化しました: {_cachedDictionaryTags.Count}個のタグ");
        }
        
        private void SetDefaultCategoryOrder()
        {
            // デフォルトのカテゴリ順序を設定（ボタンクリック時のみ実行）
            
            // 既存の順序をクリア
            _prefixOrder.Clear();
            _suffixOrder.Clear();
            
            // 新しいデフォルト順序: Rating, Quality, Character, Copyright, Artist, PersonCounts(先頭) | Model, Meta(末尾)
            var defaultPrefixOrder = new List<string> { "Rating", "Quality", "Character", "Copyright", "Artist", "PersonCounts" };
            var defaultSuffixOrder = new List<string> { "Model", "Meta" };
            
            // 既存のカテゴリのみを追加
            foreach (var category in defaultPrefixOrder)
            {
                if (_tagCategories.ContainsKey(category))
                {
                    _prefixOrder.Add(category);
                }
            }
            
            foreach (var category in defaultSuffixOrder)
            {
                if (_tagCategories.ContainsKey(category))
                {
                    _suffixOrder.Add(category);
                }
            }
            
            // UIを更新
            UpdateTagCategories();
            
            AddMainLogEntry($"デフォルトカテゴリ順序を設定: Prefix({string.Join(", ", _prefixOrder)}), Suffix({string.Join(", ", _suffixOrder)})");
        }

        private Dictionary<string, TagCategory> LoadCategoriesFromFiles(string[] files)
        {
            var categories = new Dictionary<string, TagCategory>();
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

            foreach (var file in files)
            {
                try
                {
                    string fullPath = Path.Combine(baseDirectory, file);
                    if (!File.Exists(fullPath))
                    {
                        AddMainLogEntry($"ファイルが見つかりません: {fullPath}");
                        continue;
                    }

                    string jsonContent = File.ReadAllText(fullPath);
                    var tagDictionary = JsonSerializer.Deserialize<Dictionary<string, int>>(jsonContent);
                    
                    if (tagDictionary == null)
                    {
                        AddMainLogEntry($"{file}の読み込み中にエラーが発生しました: デシリアライズ結果がnullです。");
                        continue;
                    }

                    string categoryName = Path.GetFileNameWithoutExtension(file);
                    
                    // タグ名のアンダースコアをスペースに置換
                    var updatedTags = new Dictionary<string, int>();
                    foreach (var tag in tagDictionary)
                    {
                        string updatedTagName = tag.Key.Replace('_', ' ');
                        updatedTags[updatedTagName] = tag.Value;
                        
                        // JSONファイルのカウント値を保存（Danbooru頻度）
                        _jsonTagCounts[updatedTagName] = tag.Value;
                    }

                    categories[categoryName] = new TagCategory { Tags = updatedTags };

                    AddMainLogEntry($"{categoryName}カテゴリのタグを読み込みました。タグ数: {updatedTags.Count}");
                }
                catch (Exception ex)
                {
                    AddMainLogEntry($"{file}の読み込み中にエラーが発生しました: {ex.Message}");
                }
            }

            return categories;
        }

        private void AddTagToUserAddedCategory(string newTag)
        {
            const string userAddedCategoryName = "UserAdded";

            // 既存のカテゴリをオーバーライドするかどうかを確認
            bool overrideExistingCategories = OverrideExistingCategoriesCheckBox.IsChecked ?? false;

            // オーバーライドしない場合のみ、既存のカテゴリをチェック
            if (!overrideExistingCategories)
            {
                // 新しいタグが他のカテゴリに属しているか確認
                foreach (var category in _tagCategories)
                {
                    if (category.Value?.Tags != null && category.Value.Tags.ContainsKey(newTag))
                    {
                        // 既知のカテゴリのカウントを増やす
                        category.Value.Tags[newTag]++;
                        return;
                    }
                }
            }

            // UserAddedカテゴリが存在しない場合、新しく作成
            if (_userAddedTagCategories == null)
            {
                _userAddedTagCategories = new Dictionary<string, TagCategory>();
            }

            if (!_userAddedTagCategories.ContainsKey(userAddedCategoryName))
            {
                _userAddedTagCategories[userAddedCategoryName] = new TagCategory { Tags = new Dictionary<string, int>() };
            }

            if (_userAddedTagCategories[userAddedCategoryName].Tags == null)
            {
                _userAddedTagCategories[userAddedCategoryName].Tags = new Dictionary<string, int>();
            }

            // タグをUserAddedカテゴリに追加または更新
            if (_userAddedTagCategories[userAddedCategoryName].Tags.ContainsKey(newTag))
            {
                _userAddedTagCategories[userAddedCategoryName].Tags[newTag]++;
                AddMainLogEntry($"タグ '{newTag}' をUserAddedカテゴリに追加しました。");
            }
            else
            {
                _userAddedTagCategories[userAddedCategoryName].Tags[newTag] = 1;
            }

            UpdateTagCategories();
        }

        private void UpdateTagCategories()
        {
            if (_tagCategories == null) { return; }
            _tagCategories.Clear();

            if (_defaultTagCategories != null)
            {
                foreach (var category in _defaultTagCategories)
                {
                    _tagCategories[category.Key] = category.Value;
                }
            }

            if (_useCustomCategories && _customTagCategories != null)
            {
                foreach (var category in _customTagCategories)
                {
                    _tagCategories[category.Key] = category.Value;
                }
            }

            if (_userAddedTagCategories != null)
            {
                foreach (var category in _userAddedTagCategories)
                {
                    _tagCategories[category.Key] = category.Value;
                }
            }

            UpdateTagCategoryListView();
        }

        private void UpdateTagCategoryListView()
        {
            var allCategories = _tagCategories.Keys.ToList();
            if (!allCategories.Contains("Unknown"))
            {
                allCategories.Add("Unknown");
            }

            var orderedCategories = _prefixOrder.Concat(_suffixOrder).ToList();
            var remainingCategories = allCategories.Except(orderedCategories).ToList();
            
            _tagCategoryNames.Clear();
            foreach (var category in _prefixOrder)
            {
                _tagCategoryNames.Add(new CategoryItem { Name = category, OrderType = "Prefix" });
            }
            foreach (var category in remainingCategories)
            {
                _tagCategoryNames.Add(new CategoryItem { Name = category, OrderType = "" });
            }
            foreach (var category in _suffixOrder)
            {
                _tagCategoryNames.Add(new CategoryItem { Name = category, OrderType = "Suffix" });
            }
        }

        private void UseCustomCategoriesCheckBox_Checked(object sender, RoutedEventArgs e)
        {
            _useCustomCategories = UseCustomCategoriesCheckBox.IsChecked ?? false;
            UpdateTagCategories();
            UpdateUIAfterTagsChange();
        }

        private string GetTagCategory(string tag)
        {
            if (_userAddedTagCategories != null)
            {
                foreach (var category in _userAddedTagCategories)
                {
                    if (category.Value.Tags.ContainsKey(tag))
                    {
                        return category.Key;
                    }
                }
            }

            if (_useCustomCategories)
            {
                foreach (var category in _customTagCategories)
                {
                    if (category.Value.Tags.ContainsKey(tag))
                    {
                        return category.Key;
                    }
                }
            }

            foreach (var category in _defaultTagCategories)
            {
                if (category.Value.Tags.ContainsKey(tag))
                {
                    return category.Key;
                }
            }

            return "Unknown";
        }

        private void MoveToPrefix_Click(object sender, RoutedEventArgs e)
        {
            if (TagCategoryListView.SelectedItem is CategoryItem selectedCategory)
            {
                _prefixOrder.Remove(selectedCategory.Name);
                _suffixOrder.Remove(selectedCategory.Name);
                _prefixOrder.Add(selectedCategory.Name);
                UpdateTagCategoryListView();
            }
        }

        private void MoveToSuffix_Click(object sender, RoutedEventArgs e)
        {
            if (TagCategoryListView.SelectedItem is CategoryItem selectedCategory)
            {
                _prefixOrder.Remove(selectedCategory.Name);
                _suffixOrder.Remove(selectedCategory.Name);
                _suffixOrder.Add(selectedCategory.Name);
                UpdateTagCategoryListView();
            }
        }

        private void RemoveFromOrders_Click(object sender, RoutedEventArgs e)
        {
            if (TagCategoryListView.SelectedItem is CategoryItem selectedCategory)
            {
                _prefixOrder.Remove(selectedCategory.Name);
                _suffixOrder.Remove(selectedCategory.Name);
                UpdateTagCategoryListView();
            }
        }

        private void SetDefaultOrder_Click(object sender, RoutedEventArgs e)
        {
            SetDefaultCategoryOrder();
        }

        // 選択された画像のタグをカテゴリ順に並び替え
        private void SortByCategoryButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                SortImageTagsByCategory(selectedImage, ShuffleInCategoriesCheckBox.IsChecked ?? false);
                UpdateUIAfterTagsChange();
                UpdateButtonStates();
            }
        }

        private async void SortByCategoryAllButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isAsyncProcessing) { return; }
            _isAsyncProcessing = true;

            ProgressBar.Visibility = Visibility.Visible;
            ProgressBar.Value = 0;

            _cts = new CancellationTokenSource();

            bool shuffleInCategories = ShuffleInCategoriesCheckBox.IsChecked ?? false;

            try
            {
                await Task.Run(() =>
                {
                    int totalImages = _imageInfos.Count;
                    int batchSize = 100; // バッチサイズを設定
                    var lastUpdateTime = DateTime.Now;

                    for (int i = 0; i < totalImages; i += batchSize)
                    {
                        if (_cts.Token.IsCancellationRequested)
                            break;

                        int end = Math.Min(i + batchSize, totalImages);
                        Parallel.For(i, end, j =>
                        {
                            SortImageTagsByCategory(_imageInfos[j], shuffleInCategories);
                        });

                        if ((DateTime.Now - lastUpdateTime).TotalSeconds >= 1)
                        {
                            Dispatcher.Invoke(() =>
                            {
                                ProgressBar.Value = (end) * 100 / totalImages;
                                UpdateUIAfterTagsChange();
                            });
                            lastUpdateTime = DateTime.Now;
                        }
                    }
                    
                }, _cts.Token);
            }
            catch (OperationCanceledException)
            {
                AddMainLogEntry("タグの並び替えがキャンセルされました。");
            }
            finally
            {
                _isAsyncProcessing = false;
                UpdateProgressBar(0);
                UpdateUIAfterTagsChange();
            }
        }

        private void SortImageTagsByCategory(ImageInfo image, bool shuffleInCategories = false)
        {
            var prefixTags = new List<string>();
            var suffixTags = new List<string>();
            var remainingTags = new List<string>(image.Tags);

            var tagMoves = new List<TagPositionInfo>();

            // Prefix tags
            foreach (var category in _prefixOrder)
            {
                var categoryTags = remainingTags.Where(tag => GetTagCategory(tag) == category).ToList();
                if (shuffleInCategories)
                {
                    categoryTags = categoryTags.OrderBy(x => Guid.NewGuid()).ToList();
                }
                foreach (var tag in categoryTags)
                {
                    int sourceIndex = image.Tags.IndexOf(tag);
                    int targetIndex = prefixTags.Count;
                    tagMoves.Add(new TagPositionInfo { Tag = tag, Position = sourceIndex });
                    tagMoves.Add(new TagPositionInfo { Tag = tag, Position = targetIndex });
                }
                prefixTags.AddRange(categoryTags);
                remainingTags.RemoveAll(tag => categoryTags.Contains(tag));
            }

            // Suffix tags
            foreach (var category in _suffixOrder.AsEnumerable().Reverse())
            {
                var categoryTags = remainingTags.Where(tag => GetTagCategory(tag) == category).ToList();
                if (shuffleInCategories)
                {
                    categoryTags = categoryTags.OrderBy(x => Guid.NewGuid()).ToList();
                }
                foreach (var tag in categoryTags)
                {
                    int sourceIndex = image.Tags.IndexOf(tag);
                    int targetIndex = prefixTags.Count + remainingTags.Count;
                    tagMoves.Add(new TagPositionInfo { Tag = tag, Position = sourceIndex });
                    tagMoves.Add(new TagPositionInfo { Tag = tag, Position = targetIndex });
                }
                suffixTags.InsertRange(0, categoryTags);
                remainingTags.RemoveAll(tag => categoryTags.Contains(tag));
            }

            // if (ShuffleInCategoriesCheckBox.IsChecked == true)
            // {
            //     remainingTags = remainingTags.OrderBy(x => Guid.NewGuid()).ToList();
            // }

            var newTags = prefixTags.Concat(remainingTags).Concat(suffixTags).ToList();

            var action = new TagGroupAction
            {
                Image = image,
                TagInfos = tagMoves,
                IsAdd = false,
                DoAction = () =>
                {
                    image.Tags = newTags;
                    Dispatcher.Invoke(() =>
                    {
                        AddMainLogEntry($"画像 '{image.ImagePath}' のタグをカテゴリ順に並び替えました");
                    });
                },
                UndoAction = () =>
                {
                    image.Tags = new List<string>(image.Tags);
                    Dispatcher.Invoke(() =>
                    {
                        AddMainLogEntry($"画像 '{image.ImagePath}' のタグの並び替えを元に戻しました");
                    });
                },
                Description = $"画像 '{image.ImagePath}' のタグをカテゴリ順に並び替え"
            };

            action.DoAction();
            _undoStack.Push(action);
            _redoStack.Clear();
        }

        /*
        ここまでタグカテゴリ関連メソッド
        */

        private async void ShuffleTagsButton_Click(object sender, RoutedEventArgs e)
        {
            var categories = _tagCategories.Keys.ToList();
            if (!categories.Contains("Unknown"))
            {
                categories.Add("Unknown");  // Unknownカテゴリを追加
            }

            var window = new TagShuffleWindow(
                async (startPos, endPos, startCategory, endCategory, applyToAll) =>
                {
                    var progress = new Progress<double>(value => UpdateProgressBar(value));
                    await ShuffleTagsAsync(startPos, endPos, startCategory, endCategory, applyToAll, progress);
                },
                categories  // 修正したカテゴリリストを渡す
            );
            
            window.Owner = this;
            window.ShowDialog();
        }

        private async Task ShuffleTagsAsync(
            int startPos, 
            int endPos, 
            string startCategory, 
            string endCategory, 
            bool applyToAll, 
            IProgress<double> progress)
        {
            if (_isAsyncProcessing) { return; }
            _isAsyncProcessing = true;

            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            var images = applyToAll ? _imageInfos : new List<ImageInfo> { selectedImage };
            int processed = 0;

            foreach (var image in images)
            {
                var tags = image.Tags.ToList();

                AddMainLogEntry($"画像 '{image.ImagePath}' のタグをシャッフルします。");
                AddMainLogEntry($"tags: {string.Join(", ", tags)}");
                
                // カテゴリに基づく開始位置の調整
                if (!string.IsNullOrEmpty(startCategory))
                {
                    if (startCategory == "Unknown")
                    {
                        // Unknownの場合：どのカテゴリにも属していないタグを探す
                        var categoryLastPos = tags
                            .Select((tag, index) => new { Tag = tag, Index = index })
                            .Where(x => !_tagCategories.Any(cat => cat.Value?.Tags?.ContainsKey(x.Tag) == true))
                            .LastOrDefault()?.Index ?? -1;
                        
                        if (categoryLastPos >= 0)
                            startPos = Math.Max(startPos, categoryLastPos + 1);
                    }
                    else if (_tagCategories.ContainsKey(startCategory))
                    {
                        var categoryTags = _tagCategories[startCategory].Tags.Keys.ToHashSet();
                        var categoryLastPos = tags
                            .Select((tag, index) => new { Tag = tag, Index = index })
                            .Where(x => categoryTags.Contains(x.Tag))
                            .LastOrDefault()?.Index ?? -1;
                        
                        if (categoryLastPos >= 0)
                            startPos = Math.Max(startPos, categoryLastPos + 1);
                    }
                }

                // カテゴリに基づく終了位置の調整
                if (!string.IsNullOrEmpty(endCategory))
                {
                    if (endCategory == "Unknown")
                    {
                        // Unknownの場合：どのカテゴリにも属していないタグを探す
                        var categoryFirstPos = tags
                            .Select((tag, index) => new { Tag = tag, Index = index })
                            .Where(x => !_tagCategories.Any(cat => cat.Value?.Tags?.ContainsKey(x.Tag) == true))
                            .FirstOrDefault()?.Index ?? tags.Count;
                        
                        if (categoryFirstPos < tags.Count)
                            endPos = Math.Min(endPos, categoryFirstPos - 1);
                    }
                    else if (_tagCategories.ContainsKey(endCategory))
                    {
                        var categoryTags = _tagCategories[endCategory].Tags.Keys.ToHashSet();
                        var categoryFirstPos = tags
                            .Select((tag, index) => new { Tag = tag, Index = index })
                            .Where(x => categoryTags.Contains(x.Tag))
                            .FirstOrDefault()?.Index ?? tags.Count;
                        
                        if (categoryFirstPos < tags.Count)
                            endPos = Math.Min(endPos, categoryFirstPos - 1);
                    }
                }

                // 実際のシャッフル範囲の決定
                int actualStartPos = Math.Min(startPos, tags.Count);
                int actualEndPos = Math.Min(endPos, tags.Count);
                
                if (actualStartPos < actualEndPos)
                {
                    // 指定範囲のタグをシャッフル
                    var range = tags.GetRange(actualStartPos, actualEndPos - actualStartPos);
                    var shuffled = range.OrderBy(x => Guid.NewGuid()).ToList();
                    tags.RemoveRange(actualStartPos, actualEndPos - actualStartPos);
                    tags.InsertRange(actualStartPos, shuffled);
                    
                    // タグを更新
                    image.Tags = tags;
                }

                processed++;
                UpdateProgressBar((double)processed / images.Count);
            }

            // UI更新
            await Dispatcher.InvokeAsync(() => UpdateUIAfterTagsChange());

            UpdateProgressBar(0);
            _isAsyncProcessing = false;
        }

        private async Task<BitmapSource> GetSelectedRegion()
        {
            var image = SelectedImage.Source as BitmapSource;
            if (image == null || SelectionRectangle.Visibility != Visibility.Visible) return null;

            try
            {
                // 画像とCanvas/Imageのサイズ比を計算
                double scaleX = image.PixelWidth / SelectedImage.ActualWidth;
                double scaleY = image.PixelHeight / SelectedImage.ActualHeight;

                // 選択範囲の座標を取得
                double x = Canvas.GetLeft(SelectionRectangle);
                double y = Canvas.GetTop(SelectionRectangle);

                // 画像の実際の表示位置とサイズを計算
                double imageX = (SelectedImage.ActualWidth - image.PixelWidth / scaleX) / 2;
                double imageY = (SelectedImage.ActualHeight - image.PixelHeight / scaleY) / 2;

                // 選択範囲を画像の座標系に変換
                int pixelX = (int)Math.Max(0, (x - imageX) * scaleX);
                int pixelY = (int)Math.Max(0, (y - imageY) * scaleY);
                int pixelWidth = (int)Math.Min(image.PixelWidth - pixelX, SelectionRectangle.Width * scaleX);
                int pixelHeight = (int)Math.Min(image.PixelHeight - pixelY, SelectionRectangle.Height * scaleY);

                // 範囲が有効かチェック
                if (pixelWidth <= 0 || pixelHeight <= 0)
                {
                    AddMainLogEntry("選択範囲が画像の有効な領域にありません。");
                    return null;
                }

                // 選択範囲を切り出し
                var croppedBitmap = new CroppedBitmap(
                    image,
                    new Int32Rect(pixelX, pixelY, pixelWidth, pixelHeight)
                );

                return croppedBitmap;
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"選択範囲の切り出しに失敗しました: {ex.Message}");
                return null;
            }
        }

        private async void FillTransparencyButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            // RGB値の取得と検証
            if (!byte.TryParse(FillRedTextBox.Text, out byte r) ||
                !byte.TryParse(FillGreenTextBox.Text, out byte g) ||
                !byte.TryParse(FillBlueTextBox.Text, out byte b))
            {
                MessageBox.Show("RGB値は0-255の範囲で入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 透過塗りつぶしの確認ダイアログ
            var result = MessageBox.Show(
                $"フィルタされた画像の透過部分を RGB({r},{g},{b})で塗りつぶしますか？",
                "確認",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question,
                MessageBoxResult.No
            );
            if (result != MessageBoxResult.Yes) return;

            try
            {
                // UI要素の無効化とプログレスバーの初期化
                FillTransparencyButton.IsEnabled = false;
                UpdateProgressBar(0);

                // 対象画像の決定（全画像または選択画像）
                bool applyToAll = ApplyToAllImagesCheckBox.IsChecked ?? false;
                List<ImageInfo> imagesToProcess = new List<ImageInfo>();
                ImageInfo currentImage = null;
                if (applyToAll)
                {
                    imagesToProcess = _imageInfos.ToList();
                }
                else
                {
                    currentImage = ImageListBox.SelectedItem as ImageInfo;
                    if (currentImage == null)
                    {
                        AddMainLogEntry("画像が選択されていません。");
                        return;
                    }
                    imagesToProcess.Add(currentImage);
                }

                AddMainLogEntry("処理開始: 画像表示をクリア");
                SelectedImage.Source = null;

                // 進捗報告用のProgress<T>オブジェクト
                var progress = new Progress<double>(value =>
                {
                    UpdateProgressBar(value / 100);
                });

                // ユーザー指定の塗りつぶし色及び「ランダム色反転」オプションの取得
                var fillColor = System.Drawing.Color.FromArgb(r, g, b);
                bool randomInvert = RandomColorInvertCheckBox.IsChecked ?? false;

                // TransparencyProcessor 内部で画像ごとに乱数処理を実施
                int processedCount = await TransparencyProcessor.ProcessImagesAsync(
                    imagesToProcess,
                    fillColor,
                    _webPHandler,
                    AddMainLogEntry,
                    progress,
                    enableRandomColorInversion: randomInvert
                );

                AddMainLogEntry($"{processedCount} 個の画像の透過部分を塗りつぶしました。");

                // 画像の再読み込みと表示の更新
                if (!applyToAll && currentImage != null)
                {
                    SelectedImage.Source = LoadImage(currentImage.ImagePath);
                }
                else if (applyToAll && ImageListBox.SelectedItem is ImageInfo selectedImage)
                {
                    SelectedImage.Source = LoadImage(selectedImage.ImagePath);
                }

                UpdateSelectedImage(currentImage);
                UpdateUIAfterImageInfosChange();
                AddMainLogEntry($"{processedCount}個の画像の透過部分を塗りつぶしました。");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"処理中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"透過部分の塗りつぶし中にエラーが発生: {ex.Message}");
            }
            finally
            {
                // UI要素の再有効化とプログレスバーのリセット
                FillTransparencyButton.IsEnabled = true;
                UpdateProgressBar(0);
            }
        }

        private async void ResizeButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません。");
                return;
            }

            // サイズのバリデーション
            if (!int.TryParse(ResizeWidthTextBox.Text, out int width) || width <= 0 ||
                !int.TryParse(ResizeHeightTextBox.Text, out int height) || height <= 0)
            {
                MessageBox.Show("サイズは1以上の整数で入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // パラメータの取得
            var mode = (ResizeModeComboBox.SelectedItem as ComboBoxItem)?.Content.ToString() ?? "縮小のみ";
            var format = (OutputFormatComboBox.SelectedItem as ComboBoxItem)?.Content.ToString() ?? "WebP";
            var resample = (ResampleModeComboBox.SelectedItem as ComboBoxItem)?.Content.ToString() ?? "Lanczos2";
            bool applyToAll = ResizeApplyToAllCheckBox.IsChecked ?? false;
            var randomScaleVariation = double.TryParse(RandomScaleVariationTextBox.Text, out double variation) ? variation : 0;
            var randomScaleStep = double.TryParse(RandomScaleStepTextBox.Text, out double step) ? step : 0.05;

            // 処理対象の決定
            List<ImageInfo> imagesToProcess = new List<ImageInfo>();
            if (applyToAll)
            {
                imagesToProcess = _imageInfos.ToList();
            }
            else
            {
                var currentImage = ImageListBox.SelectedItem as ImageInfo;
                if (currentImage == null)
                {
                    AddMainLogEntry("画像が選択されていません。");
                    return;
                }
                imagesToProcess.Add(currentImage);
            }

            // 確認ダイアログ
            var message = $"以下の設定でリサイズを実行しますか？\n\n" +
                         $"サイズ: {width}x{height}px\n" +
                         $"モード: {mode}\n" +
                         $"出力形式: {format}\n" +
                         $"リサンプル: {resample}\n" +
                         $"対象: {(applyToAll ? "すべての画像" : "選択中の画像")}";

            var result = MessageBox.Show(
                message,
                "確認",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question,
                MessageBoxResult.No
            );

            if (result != MessageBoxResult.Yes) return;

            try
            {
                // UI要素を無効化
                ResizeButton.IsEnabled = false;
                UpdateProgressBar(0);

                // 進捗報告用のProgress<T>オブジェクトを作成
                var progress = new Progress<double>(value =>
                {
                    UpdateProgressBar(value / 100);
                });

                // リサイズ処理の実行
                var resizeParams = new ResizeParameters
                {
                    TargetWidth = width,
                    TargetHeight = height,
                    Mode = mode,
                    OutputFormat = format,
                    ResampleMode = resample
                };

                int processedCount = await ImageProcessor.ResizeImagesAsync(
                    imagesToProcess,
                    resizeParams,
                    _webPHandler,
                    AddMainLogEntry,
                    progress,
                    randomScaleVariation,
                    randomScaleStep
                );

                AddMainLogEntry($"{processedCount}個の画像をリサイズしました。");
                
                // 画像の再読み込みと表示の更新
                if (!applyToAll && ImageListBox.SelectedItem is ImageInfo selectedImage)
                {
                    SelectedImage.Source = LoadImage(selectedImage.ImagePath);
                }
                UpdateCentralDisplay();
                UpdateUIAfterImageInfosChange();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"処理中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"リサイズ中にエラーが発生: {ex.Message}");
            }
            finally
            {
                // UI要素を再有効化
                ResizeButton.IsEnabled = true;
                UpdateProgressBar(0);
            }
        }

        // 例：サイドボタンエリアのイベントハンドラ
        private void CreateDaughterDatasetButton_Click(object sender, RoutedEventArgs e)
        {
            // _imageInfos は MainWindow 内の全画像リストとします
            var daughterWindow = new DaughterDatasetWindow(_imageInfos);
            daughterWindow.Owner = this;

            // MainWindow 側のGPU並列度（例：VLMConcurrencySliderの値）をそのまま流用
            if (VLMConcurrencySlider != null)
            {
                daughterWindow.CPUConcurrencyLimit = (int)VLMConcurrencySlider.Value;
            }

            if (daughterWindow.ShowDialog() == true)
            {
                AddMainLogEntry("娘データセットの作成と保存が完了しました。");
            }
        }

        // LoadAllBitmapImagesAsync の修正（同時実行数をさらに制限するテスト）
        private async Task LoadAllBitmapImagesAsync(List<string> imagePaths)
        {
            // ★ 同時実行数をさらに減らしてテスト (例: 2)
            int maxConcurrency = 2; 
            // 必要なら MainWindow にログ追加用のメソッドを用意
            // AddLogToMainWindow($"Using max concurrency: {maxConcurrency}"); 
            var semaphore = new SemaphoreSlim(maxConcurrency);
            var loadedBitmaps = new List<BitmapImage>(); // 結果を収集する場合

            var tasks = imagePaths.Select(async imagePath =>
            {
                await semaphore.WaitAsync(); 
                BitmapImage bitmap = null; // 初期化
                try
                {
                    // Task.Run 内で LoadImageForVLMPrediction を呼び出す
                    bitmap = await Task.Run(() => LoadImageForVLMPrediction(imagePath));
                }
                // LoadImageForVLMPrediction が null を返す可能性がある場合や、
                // Task 内で別の例外が発生する場合に備える (オプション)
                // catch (Exception taskEx) {
                //     AddLogToMainWindow($"Task Exception for {imagePath}: {taskEx.Message}");
                // }
                finally
                {
                    semaphore.Release(); 
                }
                return bitmap; // 成功した場合は BitmapImage、失敗(null返却)なら null
            }).ToList();

            // すべてのタスク完了を待機
            var results = await Task.WhenAll(tasks); 

            // null でない結果のみを処理
            foreach (var bmp in results.Where(b => b != null))
            {
                // 正常に読み込めた BitmapImage に対する処理
                // (例: UIコレクションに追加。Dispatcher.Invoke を忘れずに)
                // Dispatcher.Invoke(() => YourBitmapCollection.Add(bmp)); 
            }
            // AddLogToMainWindow("Finished loading all images.");
        }
        
        #region キーボード操作支援機能
        
        private void AddTagToHistory(string tag)
        {
            // 履歴に追加
            _recentAddedTags.Remove(tag); // 既存を削除
            _recentAddedTags.AddFirst(tag); // 先頭に追加
            
            // 最大数を超えたら古いものを削除
            while (_recentAddedTags.Count > MaxRecentTags)
            {
                _recentAddedTags.RemoveLast();
            }
            
            // 頻度カウント
            if (_tagFrequency.ContainsKey(tag))
                _tagFrequency[tag]++;
            else
                _tagFrequency[tag] = 1;
        }
        
        private void AddRecentTag(int index)
        {
            if (index == 0 && _recentAddedTags.Count > 0)
            {
                // Ctrl+0: 直前のタグを再追加
                var tag = _recentAddedTags.First.Value;
                AddTagToCurrentImage(tag);
            }
            else if (index > 0 && index <= _recentAddedTags.Count)
            {
                // Ctrl+1-9: 履歴から選択
                var tag = _recentAddedTags.ElementAt(index - 1);
                AddTagToCurrentImage(tag);
            }
        }
        
        private void AddTagToCurrentImage(string tag)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null && !selectedImage.Tags.Contains(tag))
            {
                selectedImage.Tags.Add(tag);
                AddTagToHistory(tag);
                UpdateUIAfterTagsChange();
                AddMainLogEntry($"タグ「{tag}」を追加しました");
            }
        }
        
        private void ShowTagHistory()
        {
            // 履歴を検索結果として表示
            if (_recentAddedTags.Count > 0)
            {
                SearchedTagsListView.ItemsSource = _recentAddedTags.Select(tag => new SearchTagInfo 
                { 
                    Tag = tag, 
                    Count = GetTagCount(tag) 
                }).ToList();
                _isShowingTagHistory = true;
                SearchTextBox.Text = "[履歴]";
                SearchTextBox.Focus();
                AddMainLogEntry($"タグ履歴を表示: {_recentAddedTags.Count}件");
            }
            else
            {
                AddMainLogEntry("タグ履歴がありません");
            }
        }
        
        private void AddTagAndMoveNext()
        {
            AddTextboxinputButton_Click(null, null);
            MoveToNextImage();
        }
        
        private void AddTagAndMovePrevious()
        {
            AddTextboxinputButton_Click(null, null);
            MoveToPreviousImage();
        }
        
        private void MoveToNextImage()
        {
            if (ImageListBox.SelectedIndex < ImageListBox.Items.Count - 1)
            {
                ImageListBox.SelectedIndex++;
                ImageListBox.ScrollIntoView(ImageListBox.SelectedItem);
            }
        }
        
        private void MoveToPreviousImage()
        {
            if (ImageListBox.SelectedIndex > 0)
            {
                ImageListBox.SelectedIndex--;
                ImageListBox.ScrollIntoView(ImageListBox.SelectedItem);
            }
        }
        
        private void MoveToFirstImage()
        {
            if (ImageListBox.Items.Count > 0)
            {
                ImageListBox.SelectedIndex = 0;
                ImageListBox.ScrollIntoView(ImageListBox.SelectedItem);
            }
        }
        
        private void MoveToLastImage()
        {
            if (ImageListBox.Items.Count > 0)
            {
                ImageListBox.SelectedIndex = ImageListBox.Items.Count - 1;
                ImageListBox.ScrollIntoView(ImageListBox.SelectedItem);
            }
        }
        
        #endregion
        
        #region 高度なフィルタリング機能
        
        private void AddTagFilterConditionButton_Click(object sender, RoutedEventArgs e)
        {
            foreach (var selectedTag in _selectedTags)
            {
                // 既に存在しない場合のみ追加
                if (!_filterConditions.Any(c => c.Tag == selectedTag && c.TargetType == FilterTargetType.Tag))
                {
                    _filterConditions.Add(new FilterCondition
                    {
                        Tag = selectedTag,
                        ConditionType = FilterConditionType.Contains,
                        LogicType = FilterLogicType.And,
                        TargetType = FilterTargetType.Tag
                    });
                }
            }
            
            if (_selectedTags.Count > 0)
            {
                AddMainLogEntry($"{_selectedTags.Count}個のタグをフィルタ条件に追加しました");
            }
            else
            {
                AddMainLogEntry("追加するタグが選択されていません");
            }
        }
        
        private void AddCategoryFilterConditionButton_Click(object sender, RoutedEventArgs e)
        {
            if (CategoryFilterComboBox.SelectedItem is ComboBoxItem selectedItem)
            {
                string categoryName = selectedItem.Content.ToString();
                
                // 既に存在しない場合のみ追加
                if (!_filterConditions.Any(c => c.Tag == categoryName && c.TargetType == FilterTargetType.Category))
                {
                    _filterConditions.Add(new FilterCondition
                    {
                        Tag = categoryName,
                        ConditionType = FilterConditionType.HasCategory,
                        LogicType = FilterLogicType.And,
                        TargetType = FilterTargetType.Category
                    });
                    
                    AddMainLogEntry($"カテゴリ「{categoryName}」をフィルタ条件に追加しました");
                }
                else
                {
                    AddMainLogEntry($"カテゴリ「{categoryName}」は既に条件に追加されています");
                }
            }
        }
        
        private void ClearFilterConditionsButton_Click(object sender, RoutedEventArgs e)
        {
            _filterConditions.Clear();
            AddMainLogEntry("フィルタ条件をクリアしました");
            FilterResultTextBlock.Text = "";
        }
        
        private void RemoveFilterConditionButton_Click(object sender, RoutedEventArgs e)
        {
            if (e.Source is Button button && button.CommandParameter is FilterCondition condition)
            {
                _filterConditions.Remove(condition);
                AddMainLogEntry($"フィルタ条件「{condition.Tag}」を削除しました");
            }
        }
        
        private void ConditionTypeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is ComboBox comboBox && comboBox.Tag is FilterCondition condition)
            {
                condition.ConditionType = (FilterConditionType)comboBox.SelectedIndex;
            }
        }
        
        // ComboBox初期化時のSelectedIndexを設定するためのイベントハンドラー
        private void ConditionTypeComboBox_Loaded(object sender, RoutedEventArgs e)
        {
            if (sender is ComboBox comboBox && comboBox.Tag is FilterCondition condition)
            {
                comboBox.SelectedIndex = (int)condition.ConditionType;
                
                // カテゴリ条件の場合は、タグ用の選択肢を無効化
                if (condition.TargetType == FilterTargetType.Category)
                {
                    if (comboBox.Items[0] is ComboBoxItem item0) item0.IsEnabled = false; // "含む"
                    if (comboBox.Items[1] is ComboBoxItem item1) item1.IsEnabled = false; // "含まない"
                }
                else // タグ条件の場合
                {
                    if (comboBox.Items[2] is ComboBoxItem item2) item2.IsEnabled = false; // "カテゴリあり"
                    if (comboBox.Items[3] is ComboBoxItem item3) item3.IsEnabled = false; // "カテゴリなし"
                }
            }
        }
        
        private void LogicTypeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is ComboBox comboBox && comboBox.Tag is FilterCondition condition)
            {
                condition.LogicType = (FilterLogicType)comboBox.SelectedIndex;
            }
        }
        
        private void ApplyAdvancedFilterButton_Click(object sender, RoutedEventArgs e)
        {
            if (_originalImageInfos == null || _originalImageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません");
                return;
            }
            
            if (_filterConditions.Count == 0)
            {
                AddMainLogEntry("フィルタ条件が設定されていません");
                return;
            }
            
            var filteredImages = ApplyAdvancedFilter(_originalImageInfos);
            _imageInfos = filteredImages.ToList();
            
            FilterResultTextBlock.Text = $"{_imageInfos.Count}/{_originalImageInfos.Count} 件";
            AddMainLogEntry($"高度フィルタを適用しました: {_imageInfos.Count}件が条件に一致");
            
            UpdateUIAfterImageInfosChange();
        }
        
        private void ResetAdvancedFilterButton_Click(object sender, RoutedEventArgs e)
        {
            if (_originalImageInfos != null)
            {
                _imageInfos = new List<ImageInfo>(_originalImageInfos);
                FilterResultTextBlock.Text = "";
                AddMainLogEntry("フィルタを解除しました");
                UpdateUIAfterImageInfosChange();
            }
        }
        
        private IEnumerable<ImageInfo> ApplyAdvancedFilter(IEnumerable<ImageInfo> images)
        {
            if (_filterConditions.Count == 0)
                return images;
            
            return images.Where(image =>
            {
                var andConditions = _filterConditions.Where(c => c.LogicType == FilterLogicType.And).ToList();
                var orConditions = _filterConditions.Where(c => c.LogicType == FilterLogicType.Or).ToList();
                
                bool andResult = true;
                bool orResult = orConditions.Count == 0; // OR条件がない場合はtrue
                
                // AND条件の評価（すべて満たす必要がある）
                foreach (var condition in andConditions)
                {
                    bool conditionMet = EvaluateFilterCondition(image, condition);
                    
                    if (!conditionMet)
                    {
                        andResult = false;
                        break;
                    }
                }
                
                // OR条件の評価（いずれかを満たせばよい）
                foreach (var condition in orConditions)
                {
                    bool conditionMet = EvaluateFilterCondition(image, condition);
                    
                    if (conditionMet)
                    {
                        orResult = true;
                        break;
                    }
                }
                
                return andResult && orResult;
            });
        }
        
        private bool EvaluateFilterCondition(ImageInfo image, FilterCondition condition)
        {
            switch (condition.TargetType)
            {
                case FilterTargetType.Tag:
                    bool hasTag = image.Tags.Contains(condition.Tag);
                    switch (condition.ConditionType)
                    {
                        case FilterConditionType.Contains:
                            return hasTag;
                        case FilterConditionType.NotContains:
                            return !hasTag;
                        default:
                            return false;
                    }
                
                case FilterTargetType.Category:
                    bool hasCategoryTag = HasCategoryTag(image, condition.Tag);
                    switch (condition.ConditionType)
                    {
                        case FilterConditionType.HasCategory:
                            return hasCategoryTag;
                        case FilterConditionType.NoCategory:
                            return !hasCategoryTag;
                        default:
                            return false;
                    }
                
                default:
                    return false;
            }
        }
        
        private bool HasCategoryTag(ImageInfo image, string categoryName)
        {
            // 画像のタグをそれぞれ調べて、指定されたカテゴリのタグが含まれているかチェック
            foreach (var tag in image.Tags)
            {
                string tagCategory = GetTagCategory(tag);
                if (string.Equals(tagCategory, categoryName, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        private void GenerateCaptionButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null)
            {
                AddMainLogEntry("画像が選択されていません");
                return;
            }

            _ = GenerateCaptionAsync(selectedImage);
        }

        private async Task GenerateCaptionAsync(ImageInfo imageInfo)
        {
            if (imageInfo == null) return;

            try
            {
                AddMainLogEntry("キャプション生成を開始します...");
                GenerateCaptionButton.IsEnabled = false;
                
                // Python環境のセットアップを確認
                bool envReady = await EnsurePythonEnvironment();
                if (!envReady)
                {
                    AddMainLogEntry("Python環境のセットアップに失敗しました。キャプション生成を中止します。");
                    return;
                }
                GenerateCaptionButton.Content = new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    Children = {
                        new System.Windows.Controls.Image { Source = new BitmapImage(new Uri("pack://application:,,,/icon/vlm.png")), Width = 16, Height = 16, Margin = new Thickness(0, 0, 5, 0) },
                        new TextBlock { Text = "生成中..." }
                    }
                };

                string pythonPath = GetPythonPath();
                string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "caption_generator.py");
                string imagePath = imageInfo.ImagePath;
                string tags = string.Join(",", imageInfo.Tags);

                // タグをカテゴリ別に分類
                var tagList = imageInfo.Tags;
                var categorizedTags = new Dictionary<string, List<string>>
                {
                    ["character"] = new List<string>(),
                    ["copyright"] = new List<string>(),
                    ["artist"] = new List<string>(),
                    ["general"] = new List<string>(),
                    ["rating"] = new List<string>(),
                    ["quality"] = new List<string>(),
                    ["meta"] = new List<string>(),
                    ["model"] = new List<string>()
                };

                foreach (var tag in tagList)
                {
                    string category = GetTagCategory(tag).ToLower();
                    
                    // カテゴリ名をPython側の期待する形式にマッピング
                    if (category == "character" || category == "characters")
                        categorizedTags["character"].Add(tag);
                    else if (category == "copyright" || category == "copyrights" || category == "series")
                        categorizedTags["copyright"].Add(tag);
                    else if (category == "artist" || category == "artists")
                        categorizedTags["artist"].Add(tag);
                    else if (category == "rating" || category == "ratings")
                        categorizedTags["rating"].Add(tag);
                    else if (category == "quality")
                        categorizedTags["quality"].Add(tag);
                    else if (category == "meta")
                        categorizedTags["meta"].Add(tag);
                    else if (category == "model" || category == "models")
                        categorizedTags["model"].Add(tag);
                    else
                        categorizedTags["general"].Add(tag);
                }

                // カテゴリ情報をJSON形式で作成
                AddPythonLogEntry($"単発推論 - カテゴリ分類結果: character={categorizedTags["character"].Count}, copyright={categorizedTags["copyright"].Count}, general={categorizedTags["general"].Count}");
                var tagData = new
                {
                    all_tags = tagList.ToList(),
                    categorized = categorizedTags
                };
                string categorizedJson = JsonSerializer.Serialize(tagData);
                
                // JSONデータをBase64エンコードして安全に引数として渡す
                byte[] jsonBytes = System.Text.Encoding.UTF8.GetBytes(categorizedJson);
                string base64Json = Convert.ToBase64String(jsonBytes);

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = $"\"{scriptPath}\" --image \"{imagePath}\" --tags \"{tags}\" --categorized-json-base64 \"{base64Json}\" --save",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };

                using (Process process = new Process { StartInfo = startInfo })
                {
                    process.Start();
                    AddPythonLogEntry($"GLM-4.1V キャプション生成を開始: {Path.GetFileName(imagePath)}");

                    Task<string> outputTask = ReadStreamAsync(process.StandardOutput, "STDOUT");
                    Task<string> errorTask = ReadStreamAsync(process.StandardError, "STDERR");

                    await process.WaitForExitAsync();

                    string output = await outputTask;
                    string error = await errorTask;

                    if (process.ExitCode == 0)
                    {
                        string finalCaption = ExtractFinalCaption(output);
                        if (!string.IsNullOrEmpty(finalCaption))
                        {
                            imageInfo.Caption = finalCaption;
                            
                            // UIの更新を確実に行う
                            Dispatcher.Invoke(() => {
                                // 現在選択中の画像の場合、CaptionTextBoxも直接更新
                                var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                                if (selectedImage == imageInfo)
                                {
                                    CaptionTextBox.Text = finalCaption;
                                }
                                
                                // JSONファイルを更新または作成
                                string jsonFilePath = Path.ChangeExtension(imageInfo.ImagePath, ".json");
                                string tagString = string.Join(",", imageInfo.Tags);
                                if (File.Exists(jsonFilePath))
                                {
                                    UpdateJsonFile(jsonFilePath, tagString, finalCaption);
                                }
                                else
                                {
                                    CreateJsonFile(jsonFilePath, tagString, finalCaption);
                                }
                            });
                            
                            AddMainLogEntry($"キャプションを生成しました: {Path.GetFileName(imagePath)}");
                            AddPythonLogEntry($"キャプション生成成功: {finalCaption.Substring(0, Math.Min(50, finalCaption.Length))}...");
                        }
                        else
                        {
                            AddMainLogEntry("キャプション生成に失敗しました（出力が空です）");
                            AddPythonLogEntry("キャプション生成失敗: 出力が空です");
                        }
                    }
                    else
                    {
                        AddMainLogEntry($"キャプション生成に失敗しました: {error}");
                        AddPythonLogEntry($"プロセス終了コード: {process.ExitCode}");
                        if (!string.IsNullOrEmpty(error))
                        {
                            AddPythonLogEntry($"エラー詳細: {error}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"キャプション生成でエラーが発生しました: {ex.Message}");
                AddPythonLogEntry($"キャプション生成例外エラー: {ex.Message}");
            }
            finally
            {
                GenerateCaptionButton.IsEnabled = true;
                GenerateCaptionButton.Content = new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    Children = {
                        new System.Windows.Controls.Image { Source = new BitmapImage(new Uri("pack://application:,,,/icon/vlm.png")), Width = 16, Height = 16, Margin = new Thickness(0, 0, 5, 0) },
                        new TextBlock { Text = "Caption生成" }
                    }
                };
            }
        }

        private async Task<string> ReadStreamAsync(StreamReader reader, string streamType = "")
        {
            var result = new System.Text.StringBuilder();
            string line;
            while ((line = await reader.ReadLineAsync()) != null)
            {
                if (line.StartsWith("STREAM:"))
                {
                    string streamChar = line.Substring(7);
                    await Dispatcher.InvokeAsync(() => {
                        var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                        if (selectedImage != null && selectedImage.Caption.Length < 1000)
                        {
                            selectedImage.Caption += streamChar;
                        }
                    });
                }
                else if (streamType == "STDERR" && !string.IsNullOrWhiteSpace(line))
                {
                    // Pythonの標準エラー出力をPythonログに表示
                    AddPythonLogEntry($"Python: {line}");
                }
                else if (streamType == "STDOUT" && !string.IsNullOrWhiteSpace(line))
                {
                    // Pythonの標準出力をPythonログに表示
                    if (line.StartsWith("FINAL:"))
                    {
                        // FINALの内容は後で抽出されるので、ログには表示しない
                    }
                    else if (!line.StartsWith("STREAM:"))
                    {
                        // ストリーミング出力以外をPythonログに表示
                        AddPythonLogEntry($"Python: {line}");
                    }
                }
                result.AppendLine(line);
            }
            return result.ToString();
        }

        private string ExtractFinalCaption(string output)
        {
            var lines = output.Split('\n');
            foreach (var line in lines)
            {
                if (line.StartsWith("FINAL:"))
                {
                    return line.Substring(6).Trim();
                }
            }
            return "";
        }

        private string GetPythonPath()
        {
            string appDir = AppDomain.CurrentDomain.BaseDirectory;
            string venvPath = Path.Combine(appDir, ".venv", "Scripts", "python.exe");
            
            if (File.Exists(venvPath))
            {
                return venvPath;
            }
            
            return "python";
        }

        private async Task<bool> EnsurePythonEnvironment()
        {
            string appDir = AppDomain.CurrentDomain.BaseDirectory;
            string venvDir = Path.Combine(appDir, ".venv");
            string pythonExe = Path.Combine(venvDir, "Scripts", "python.exe");
            string requirementsPath = Path.Combine(appDir, "requirements.txt");
            string setupCompleteFile = Path.Combine(venvDir, ".setup_complete");

            if (File.Exists(setupCompleteFile) && File.Exists(pythonExe))
            {
                return true;
            }

            try
            {
                AddMainLogEntry("Python環境をセットアップしています...");
                AddPythonLogEntry("Python環境セットアップを開始します");
                
                if (!File.Exists(requirementsPath))
                {
                    AddMainLogEntry("requirements.txtが見つかりません");
                    AddPythonLogEntry("requirements.txtが見つかりません");
                    return false;
                }

                if (!Directory.Exists(venvDir) || !File.Exists(pythonExe))
                {
                    AddMainLogEntry("Python仮想環境を作成しています...");
                    AddPythonLogEntry($"仮想環境を作成中: {venvDir}");
                    
                    ProcessStartInfo createVenvInfo = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"-m venv \"{venvDir}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = appDir
                    };

                    using (Process createVenv = new Process { StartInfo = createVenvInfo })
                    {
                        createVenv.Start();
                        AddPythonLogEntry("python -m venv を実行中...");
                        
                        await createVenv.WaitForExitAsync();
                        
                        if (createVenv.ExitCode != 0)
                        {
                            string error = await createVenv.StandardError.ReadToEndAsync();
                            AddMainLogEntry($"Python仮想環境の作成に失敗しました: {error}");
                            AddPythonLogEntry($"仮想環境作成エラー: {error}");
                            return false;
                        }
                        else
                        {
                            AddPythonLogEntry("仮想環境の作成が完了しました");
                        }
                    }
                }

                if (!File.Exists(pythonExe))
                {
                    AddMainLogEntry("Python実行ファイルが見つかりません");
                    AddPythonLogEntry("Python実行ファイルが見つかりません");
                    return false;
                }

                AddMainLogEntry("依存関係をインストールしています（時間がかかる場合があります）...");
                AddPythonLogEntry("Python環境セットアップを段階的に実行します...");

                // ステップ1: pipのアップグレード
                AddPythonLogEntry("ステップ 1/3: pipをアップグレードしています...");
                bool pipUpgradeSuccess = await RunPipCommand(pythonExe, "-m pip install --upgrade pip", "pipアップグレード");
                if (!pipUpgradeSuccess)
                {
                    AddPythonLogEntry("警告: pipのアップグレードに失敗しましたが、続行します");
                }

                // ステップ2: 基本パッケージのインストール
                AddPythonLogEntry("ステップ 2/3: 基本パッケージをインストールしています...");
                string[] basicPackages = {
                    "numpy>=1.21.0",
                    "Pillow>=9.0.0",
                    "setuptools>=65.0",
                    "wheel>=0.38.0"
                };

                foreach (string package in basicPackages)
                {
                    bool success = await RunPipCommand(pythonExe, $"-m pip install \"{package}\"", $"基本パッケージ {package}");
                    if (!success)
                    {
                        AddMainLogEntry($"基本パッケージのインストールに失敗しました: {package}");
                        AddPythonLogEntry($"エラー: {package} のインストールに失敗");
                        return false;
                    }
                }

                // ステップ3: 残りの依存関係をインストール
                AddPythonLogEntry("ステップ 3/3: GLM-4.1V関連パッケージをインストールしています...");
                bool mainInstallSuccess = await RunPipCommand(pythonExe, $"-m pip install -r \"{requirementsPath}\"", "メイン依存関係");
                
                if (mainInstallSuccess)
                {
                    File.WriteAllText(setupCompleteFile, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
                    AddMainLogEntry("Python環境のセットアップが完了しました");
                    AddPythonLogEntry("依存関係のインストールが正常に完了しました");
                    AddPythonLogEntry("GLM-4.1V キャプション生成の準備完了");
                    return true;
                }
                else
                {
                    AddMainLogEntry("メイン依存関係のインストールに失敗しました");
                    return false;
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"Python環境のセットアップでエラーが発生しました: {ex.Message}");
                AddPythonLogEntry($"セットアップ例外エラー: {ex.Message}");
                return false;
            }
        }

        private async Task<bool> RunPipCommand(string pythonExe, string pipArgs, string operationName)
        {
            try
            {
                ProcessStartInfo pipInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = pipArgs,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };

                using (Process pipProcess = new Process { StartInfo = pipInfo })
                {
                    pipProcess.Start();
                    AddPythonLogEntry($"実行中: {pipArgs}");

                    // リアルタイムで出力を読み取り
                    Task outputTask = Task.Run(async () =>
                    {
                        while (!pipProcess.StandardOutput.EndOfStream)
                        {
                            string line = await pipProcess.StandardOutput.ReadLineAsync();
                            if (!string.IsNullOrWhiteSpace(line))
                            {
                                AddPythonLogEntry($"pip: {line}");
                            }
                        }
                    });

                    Task errorTask = Task.Run(async () =>
                    {
                        while (!pipProcess.StandardError.EndOfStream)
                        {
                            string line = await pipProcess.StandardError.ReadLineAsync();
                            if (!string.IsNullOrWhiteSpace(line))
                            {
                                AddPythonLogEntry($"pip ERROR: {line}");
                            }
                        }
                    });

                    await pipProcess.WaitForExitAsync();
                    await Task.WhenAll(outputTask, errorTask);

                    if (pipProcess.ExitCode == 0)
                    {
                        AddPythonLogEntry($"✓ {operationName} が正常に完了しました");
                        return true;
                    }
                    else
                    {
                        AddPythonLogEntry($"✗ {operationName} が失敗しました (終了コード: {pipProcess.ExitCode})");
                        return false;
                    }
                }
            }
            catch (Exception ex)
            {
                AddPythonLogEntry($"✗ {operationName} 実行中に例外が発生: {ex.Message}");
                return false;
            }
        }

        private void ClearCaptionButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null)
            {
                AddMainLogEntry("画像が選択されていません");
                return;
            }

            selectedImage.Caption = "";
            AddMainLogEntry("キャプションをクリアしました");
        }

        private void CaptionTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (_isUpdatingSelection) return; // 画像切り替え中は処理しない
            
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                selectedImage.Caption = CaptionTextBox.Text ?? string.Empty;
            }
        }

        private void GenerateAllCaptionsButton_Click(object sender, RoutedEventArgs e)
        {
            if (_imageInfos == null || _imageInfos.Count == 0)
            {
                AddMainLogEntry("対象の画像がありません");
                return;
            }

            if (_isContinuousCaptionGeneration)
            {
                AddMainLogEntry("既に連続キャプション生成が実行中です");
                return;
            }

            bool skipExisting = SkipExistingCaptionsCheckBox.IsChecked ?? false;
            int targetCount = skipExisting ? _imageInfos.Count(img => string.IsNullOrEmpty(img.Caption)) : _imageInfos.Count;
            
            if (targetCount == 0)
            {
                AddMainLogEntry("対象の画像がありません（既にキャプションが存在します）");
                return;
            }

            string message = skipExisting 
                ? $"対象{targetCount}枚の画像にキャプションを生成しますか？\n\n※既にキャプションがある画像はスキップされます"
                : $"全{_imageInfos.Count}枚の画像にキャプションを生成しますか？\n\n※既にキャプションがある画像も上書きされます";

            var result = MessageBox.Show(message, "確認", MessageBoxButton.YesNo, MessageBoxImage.Question);

            if (result == MessageBoxResult.Yes)
            {
                _ = StartContinuousCaptionGenerationAsync();
            }
        }

        private void StopCaptionGenerationButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isContinuousCaptionGeneration && _captionCancellationTokenSource != null)
            {
                _captionCancellationTokenSource.Cancel();
                AddMainLogEntry("キャプション生成の停止要求を送信しました");
            }
        }

        private async Task StartContinuousCaptionGenerationAsync()
        {
            _isContinuousCaptionGeneration = true;
            _currentCaptionIndex = 0;
            _captionCancellationTokenSource = new CancellationTokenSource();

            // UI状態を更新
            GenerateCaptionButton.IsEnabled = false;
            GenerateAllCaptionsButton.IsEnabled = false;
            StopCaptionGenerationButton.IsEnabled = true;
            ImageListBox.IsEnabled = false; // ユーザーによる画像切り替えを無効化
            
            // 進捗バーを表示して初期化
            ProgressBar.Visibility = Visibility.Visible;
            UpdateProgressBar(0);

            try
            {
                bool skipExisting = SkipExistingCaptionsCheckBox.IsChecked ?? false;
                int targetCount = skipExisting ? _imageInfos.Count(img => string.IsNullOrEmpty(img.Caption)) : _imageInfos.Count;
                
                AddMainLogEntry($"連続キャプション生成を開始します（対象: {targetCount}枚、スキップ: {skipExisting}）");

                // 永続Pythonセッションを開始
                bool sessionStarted = await StartPersistentPythonSessionAsync();
                if (!sessionStarted)
                {
                    AddMainLogEntry("永続Pythonセッションの開始に失敗しました。連続生成を中止します。");
                    return;
                }

                int processedCount = 0;
                int skippedCount = 0;

                for (int i = 0; i < _imageInfos.Count; i++)
                {
                    if (_captionCancellationTokenSource.Token.IsCancellationRequested)
                    {
                        AddMainLogEntry($"キャプション生成が停止されました（処理済み: {processedCount}枚、スキップ: {skippedCount}枚）");
                        break;
                    }

                    _currentCaptionIndex = i;
                    var imageInfo = _imageInfos[i];

                    // スキップ処理
                    if (skipExisting && !string.IsNullOrEmpty(imageInfo.Caption))
                    {
                        skippedCount++;
                        AddMainLogEntry($"スキップ: {Path.GetFileName(imageInfo.ImagePath)} (既存キャプションあり)");
                        
                        // スキップ時も進捗を更新
                        double skipProgress = (double)(processedCount + skippedCount) / _imageInfos.Count;
                        UpdateProgressBar(skipProgress);
                        continue;
                    }

                    // 現在の画像を選択状態に更新
                    ImageListBox.SelectedItem = imageInfo;
                    ImageListBox.ScrollIntoView(imageInfo);

                    AddMainLogEntry($"キャプション生成中: {Path.GetFileName(imageInfo.ImagePath)} ({processedCount + 1}/{targetCount})");

                    // 永続セッションを使用してキャプション生成
                    await GenerateCaptionWithPersistentSessionAsync(imageInfo);
                    processedCount++;

                    // 進捗を更新（全体画像数に対する進捗として表示）
                    double progress = (double)(processedCount + skippedCount) / _imageInfos.Count;
                    UpdateProgressBar(progress);

                    if (_captionCancellationTokenSource.Token.IsCancellationRequested)
                        break;
                }

                if (!_captionCancellationTokenSource.Token.IsCancellationRequested)
                {
                    AddMainLogEntry($"連続キャプション生成が完了しました（処理済み: {processedCount}枚、スキップ: {skippedCount}枚）");
                }
            }
            catch (OperationCanceledException)
            {
                AddMainLogEntry("キャプション生成がキャンセルされました");
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"連続キャプション生成中にエラーが発生: {ex.Message}");
            }
            finally
            {
                // モデルアンロードを実行（VRAMを解放）
                await UnloadModelAsync();
                
                // UI状態をリセット
                _isContinuousCaptionGeneration = false;
                GenerateCaptionButton.IsEnabled = true;
                GenerateAllCaptionsButton.IsEnabled = true;
                StopCaptionGenerationButton.IsEnabled = false;
                ImageListBox.IsEnabled = true; // 画像切り替えを再有効化
                
                // 進捗バーを非表示にして初期化
                ProgressBar.Visibility = Visibility.Hidden;
                UpdateProgressBar(0);
                
                _captionCancellationTokenSource?.Dispose();
                _captionCancellationTokenSource = null;
            }
        }

        private async Task<bool> StartPersistentPythonSessionAsync()
        {
            try
            {
                string pythonPath = GetPythonPath();
                string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "caption_generator.py");

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = $"\"{scriptPath}\" --interactive",
                    UseShellExecute = false,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };

                _persistentPythonProcess = new Process { StartInfo = startInfo };
                _persistentPythonProcess.Start();

                _pythonInput = _persistentPythonProcess.StandardInput;
                _pythonOutput = _persistentPythonProcess.StandardOutput;
                _pythonError = _persistentPythonProcess.StandardError;

                // エラー出力を継続的に監視
                _ = Task.Run(async () =>
                {
                    try
                    {
                        while (!_persistentPythonProcess.HasExited)
                        {
                            string line = await _pythonError.ReadLineAsync();
                            if (!string.IsNullOrEmpty(line))
                            {
                                AddPythonLogEntry($"Python stderr: {line}");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        AddPythonLogEntry($"エラー出力監視の例外: {ex.Message}");
                    }
                });

                AddPythonLogEntry("永続Pythonセッションを開始しました");
                
                // READYメッセージを待機
                string readyMessage = await _pythonOutput.ReadLineAsync();
                if (readyMessage == "READY")
                {
                    AddPythonLogEntry("Pythonセッションが準備完了しました");
                    return true;
                }
                else
                {
                    AddMainLogEntry($"Pythonセッションの初期化に失敗: {readyMessage}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"永続Pythonセッションの開始に失敗: {ex.Message}");
                return false;
            }
        }

        private async Task<string> ExecutePythonCommandAsync(string imagePath, string tags)
        {
            AddPythonLogEntry("=== ExecutePythonCommandAsync 開始 ===");
            AddPythonLogEntry($"受信したタグ: {tags}");
            
            if (_persistentPythonProcess == null || _persistentPythonProcess.HasExited)
            {
                AddMainLogEntry("Pythonセッションが無効です。再開始します。");
                bool started = await StartPersistentPythonSessionAsync();
                if (!started) return null;
            }

            try
            {
                // 連続モード用のログ出力（プロンプト表示は初回のみ）
                if (_currentCaptionIndex == 0 || _currentCaptionIndex == -1)
                {
                    string prompt = @"Please describe this image in 2-3 concise sentences. Focus on the main subject and key visual elements.

Character: {character}
Copyright: {copyright}

Provide your answer wrapped in <answer></answer> tags:";
                    AddPythonLogEntry($"使用するプロンプト:\n{prompt}");
                }
                
                AddPythonLogEntry($"対象画像: {Path.GetFileName(imagePath)}");
                AddPythonLogEntry($"タグ: {tags}");

                // タグをカテゴリ別に分類
                var tagList = tags.Split(',').Select(t => t.Trim()).Where(t => !string.IsNullOrEmpty(t)).ToList();
                var categorizedTags = new Dictionary<string, List<string>>
                {
                    ["character"] = new List<string>(),
                    ["copyright"] = new List<string>(),
                    ["artist"] = new List<string>(),
                    ["general"] = new List<string>(),
                    ["rating"] = new List<string>(),
                    ["quality"] = new List<string>(),
                    ["meta"] = new List<string>(),
                    ["model"] = new List<string>()
                };

                foreach (var tag in tagList)
                {
                    string category = GetTagCategory(tag).ToLower();
                    
                    // カテゴリ名をPython側の期待する形式にマッピング
                    if (category == "character" || category == "characters")
                        categorizedTags["character"].Add(tag);
                    else if (category == "copyright" || category == "copyrights" || category == "series")
                        categorizedTags["copyright"].Add(tag);
                    else if (category == "artist" || category == "artists")
                        categorizedTags["artist"].Add(tag);
                    else if (category == "rating" || category == "ratings")
                        categorizedTags["rating"].Add(tag);
                    else if (category == "quality")
                        categorizedTags["quality"].Add(tag);
                    else if (category == "meta")
                        categorizedTags["meta"].Add(tag);
                    else if (category == "model" || category == "models")
                        categorizedTags["model"].Add(tag);
                    else
                        categorizedTags["general"].Add(tag);
                }

                // カテゴリ情報をJSON形式で作成
                AddPythonLogEntry($"カテゴリ分類結果: character={categorizedTags["character"].Count}, copyright={categorizedTags["copyright"].Count}, general={categorizedTags["general"].Count}");
                
                // デバッグ用：各カテゴリの具体的なタグを表示
                foreach (var kvp in categorizedTags)
                {
                    if (kvp.Value.Count > 0)
                    {
                        var sampleTags = string.Join(", ", kvp.Value.Take(3));
                        AddPythonLogEntry($"  {kvp.Key}カテゴリ: {sampleTags}...");
                    }
                }
                var tagData = new
                {
                    all_tags = tagList,
                    categorized = categorizedTags
                };
                string tagsJson = JsonSerializer.Serialize(tagData);

                // コマンドを送信（JSON形式で送信）
                string command = $"PROCESS_JSON|{imagePath}|{tagsJson}";
                AddPythonLogEntry($"送信コマンド: {command.Substring(0, Math.Min(100, command.Length))}...");
                await _pythonInput.WriteLineAsync(command);
                await _pythonInput.FlushAsync();

                AddPythonLogEntry($"GLM-4.1V キャプション生成を開始: {Path.GetFileName(imagePath)}");

                // 結果を継続的に読み取り（ストリーミング対応）
                string finalCaption = "";
                StringBuilder streamingOutput = new StringBuilder();
                
                // タイムアウト付きで応答を待機
                var cancellationToken = new CancellationTokenSource(TimeSpan.FromSeconds(120)); // 120秒タイムアウト
                bool gotFinalResponse = false;
                
                while (!cancellationToken.Token.IsCancellationRequested)
                {
                    string line = null;
                    try 
                    {
                        var readTask = _pythonOutput.ReadLineAsync();
                        if (await Task.WhenAny(readTask, Task.Delay(-1, cancellationToken.Token)) == readTask)
                        {
                            line = await readTask;
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        AddPythonLogEntry("Python応答タイムアウト");
                        break;
                    }
                    
                    if (line == null) 
                    {
                        // ストリームが閉じられた場合
                        AddPythonLogEntry("Pythonストリームが閉じられました");
                        break;
                    }
                    
                    if (string.IsNullOrEmpty(line)) 
                    {
                        // 空行は無視して続行
                        continue;
                    }

                    if (line.StartsWith("STREAM:"))
                    {
                        // ストリーミング文字を内部的に蓄積（ログには出力しない）
                        string char_ = line.Substring(7);
                        streamingOutput.Append(char_);
                        // ログ出力を削除してクラッタリングを防止
                    }
                    else if (line.StartsWith("FINAL:"))
                    {
                        finalCaption = line.Substring(6);
                        AddPythonLogEntry($"キャプション生成完了: {finalCaption}");
                        // SAVEDまたはSAVE_FAILEDを待つため、breakしない
                    }
                    else if (line.StartsWith("ERROR:"))
                    {
                        AddPythonLogEntry($"Python エラー: {line.Substring(6)}");
                        return null;
                    }
                    else if (line == "SAVED")
                    {
                        AddPythonLogEntry("JSONファイルに保存完了");
                        if (!string.IsNullOrEmpty(finalCaption))
                        {
                            gotFinalResponse = true;
                            break; // FINALとSAVEDの両方を受信したら終了
                        }
                    }
                    else if (line == "SAVE_FAILED")
                    {
                        AddPythonLogEntry("JSONファイルの保存に失敗");
                        if (!string.IsNullOrEmpty(finalCaption))
                        {
                            gotFinalResponse = true;
                            break; // FINALとSAVE_FAILEDの両方を受信したら終了
                        }
                    }
                    else
                    {
                        // その他の出力をログに表示
                        AddPythonLogEntry($"Python出力: {line}");
                    }
                }
                
                return finalCaption;
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"Python通信エラー: {ex.Message}");
                AddPythonLogEntry($"通信エラー詳細: {ex}");
                return null;
            }
        }

        private async Task GenerateCaptionWithPersistentSessionAsync(ImageInfo imageInfo)
        {
            try
            {
                string tags = string.Join(",", imageInfo.Tags);
                string finalCaption = await ExecutePythonCommandAsync(imageInfo.ImagePath, tags);

                if (!string.IsNullOrEmpty(finalCaption))
                {
                    imageInfo.Caption = finalCaption;

                    // UIとJSONファイルの更新
                    var selectedImage = ImageListBox.SelectedItem as ImageInfo;
                    if (selectedImage == imageInfo)
                    {
                        CaptionTextBox.Text = finalCaption;
                    }

                    string jsonFilePath = Path.ChangeExtension(imageInfo.ImagePath, ".json");
                    string tagString = string.Join(",", imageInfo.Tags);
                    if (File.Exists(jsonFilePath))
                    {
                        UpdateJsonFile(jsonFilePath, tagString, finalCaption);
                    }
                    else
                    {
                        CreateJsonFile(jsonFilePath, tagString, finalCaption);
                    }

                    AddMainLogEntry($"キャプションを生成しました: {Path.GetFileName(imageInfo.ImagePath)}");
                }
                else
                {
                    AddMainLogEntry($"キャプション生成に失敗: {Path.GetFileName(imageInfo.ImagePath)} - 有効なキャプションが生成されませんでした");
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"キャプション生成エラー ({Path.GetFileName(imageInfo.ImagePath)}): {ex.Message}");
                AddPythonLogEntry($"キャプション生成エラー: {ex}");
            }
        }

        private async Task UnloadModelAsync()
        {
            try
            {
                if (_persistentPythonProcess != null && !_persistentPythonProcess.HasExited)
                {
                    AddPythonLogEntry("モデルをアンロード中...");
                    
                    // アンロードコマンドを送信
                    await _pythonInput.WriteLineAsync("UNLOAD");
                    await _pythonInput.FlushAsync();
                    
                    // アンロード完了応答を待機（タイムアウト付き）
                    var timeout = TimeSpan.FromSeconds(10);
                    var cts = new CancellationTokenSource(timeout);
                    
                    try
                    {
                        while (!cts.Token.IsCancellationRequested)
                        {
                            string response = await _pythonOutput.ReadLineAsync();
                            if (response == "MODEL_UNLOADED")
                            {
                                AddPythonLogEntry("モデルのアンロードが完了しました");
                                break;
                            }
                            else if (response == "UNLOAD_FAILED")
                            {
                                AddPythonLogEntry("モデルのアンロードに失敗しました");
                                break;
                            }
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        AddPythonLogEntry("モデルアンロードのタイムアウト");
                    }
                }
            }
            catch (Exception ex)
            {
                AddPythonLogEntry($"モデルアンロードエラー: {ex.Message}");
            }
        }

        private void StopPersistentPythonSession()
        {
            try
            {
                if (_persistentPythonProcess != null && !_persistentPythonProcess.HasExited)
                {
                    // モデルアンロードコマンドを送信してからプロセス終了
                    _pythonInput?.WriteLine("UNLOAD");
                    _pythonInput?.Flush();
                    
                    // アンロード応答を少し待機
                    System.Threading.Thread.Sleep(1000);
                    
                    // 終了コマンドを送信
                    _pythonInput?.WriteLine("EXIT");
                    _pythonInput?.Flush();

                    // プロセス終了を待機（タイムアウト付き）
                    if (!_persistentPythonProcess.WaitForExit(5000))
                    {
                        _persistentPythonProcess.Kill();
                    }

                    AddPythonLogEntry("永続Pythonセッションを終了しました");
                }
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"Pythonセッション終了エラー: {ex.Message}");
            }
            finally
            {
                _pythonInput?.Dispose();
                _pythonOutput?.Dispose();
                _pythonError?.Dispose();
                _persistentPythonProcess?.Dispose();

                _pythonInput = null;
                _pythonOutput = null;
                _pythonError = null;
                _persistentPythonProcess = null;
            }
        }
        
        #endregion
    }

    // フィルタ条件クラス
    public class FilterCondition : INotifyPropertyChanged
    {
        private string _tag;
        private FilterConditionType _conditionType;
        private FilterLogicType _logicType;
        private FilterTargetType _targetType;

        public string Tag
        {
            get => _tag;
            set { _tag = value; OnPropertyChanged(); }
        }

        public FilterConditionType ConditionType
        {
            get => _conditionType;
            set { _conditionType = value; OnPropertyChanged(); }
        }

        public FilterLogicType LogicType
        {
            get => _logicType;
            set { _logicType = value; OnPropertyChanged(); }
        }

        public FilterTargetType TargetType
        {
            get => _targetType;
            set { _targetType = value; OnPropertyChanged(); }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public enum FilterConditionType
    {
        Contains,        // 含む
        NotContains,     // 含まない
        HasCategory,     // カテゴリのタグがある
        NoCategory       // カテゴリのタグがない
    }

    public enum FilterTargetType
    {
        Tag,        // 個別タグ
        Category    // カテゴリ
    }

    public enum FilterLogicType
    {
        And, // AND条件
        Or   // OR条件
    }
}
