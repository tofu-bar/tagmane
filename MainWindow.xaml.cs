using Microsoft.WindowsAPICodePack.Dialogs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Collections.ObjectModel;
using System.IO;

namespace tagmane
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private FileExplorer _fileExplorer;
        private List<ImageInfo> _imageInfos;
        private Dictionary<string, int> _allTags;
        private bool _isUpdatingSelection = false;
        private HashSet<string> _selectedTags = new HashSet<string>();
        private HashSet<string> _currentImageTags = new HashSet<string>();
        private Stack<TagAction> _undoStack = new Stack<TagAction>();
        private Stack<TagAction> _redoStack = new Stack<TagAction>();
        private ObservableCollection<string> _debugLogEntries;
        private ObservableCollection<string> _logEntries;
        private ObservableCollection<ActionLogItem> _actionLogItems;
        private const int MaxLogEntries = 20; // 100から20に変更
        private Point? _startPoint;
        private ListViewItem _draggedItem;
        private bool _isDragging = false;
        private VLMPredictor _vlmPredictor;

        public MainWindow()
        {
            try
            {
                InitializeComponent();
                _fileExplorer = new FileExplorer();
                _allTags = new Dictionary<string, int>();
                _logEntries = new ObservableCollection<string>();
                _debugLogEntries = new ObservableCollection<string>();
                _actionLogItems = new ObservableCollection<ActionLogItem>();
                ActionListView.ItemsSource = _actionLogItems;
                
                // デバッグ用のメッセージを追加
                MessageBox.Show("MainWindowが初期化されました。");
                
                // ウィンドウを表示
                this.Show();

                InitializeVLMPredictor();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"MainWindowの初期化中にエラーが発生しました: {ex.Message}\n\nStackTrace: {ex.StackTrace}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private async void InitializeVLMPredictor()
        {
            _vlmPredictor = new VLMPredictor();
            _vlmPredictor.LogUpdated += UpdateVLMLog;
            LoadVLMModel();
        }

        private async void LoadVLMModel()
        {
            try
            {
                AddMainLogEntry("VLMモデルの読み込みを開始します。");
                await _vlmPredictor.LoadModel("SmilingWolf/wd-swinv2-tagger-v3");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"VLMモデルの読み込みに失敗しました: {ex.Message}");
            }
        }

        private async void PredictVLMTags()
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null)
            {
                AddMainLogEntry("画像が選択されていません。");
                return;
            }

            AddMainLogEntry("VLM推論を開始します");

            try
            {
                var (generalTags, rating, characters, allTags) = _vlmPredictor.Predict(
                    new BitmapImage(new Uri(selectedImage.ImagePath)),
                    0.35f, // generalThresh
                    false, // generalMcutEnabled
                    0.85f, // characterThresh
                    false  // characterMcutEnabled
                );

                // 結果を表示または処理する
                UpdateVLMTagsDisplay(generalTags, rating, characters, allTags);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"VLM推論中にエラーが発生しました: {ex.Message}");
            }
        }

        private void UpdateVLMTagsDisplay(string generalTags, Dictionary<string, float> rating, Dictionary<string, float> characters, Dictionary<string, float> allTags)
        {
            // UIを更新して結果を表示する
            // 例: ListBoxやTextBlockに結果を表示する
            AddMainLogEntry($"VLM推論結果: {generalTags}");
        }

        // VLM推論を実行するボタンのクリックイベントハンドラ
        private async void VLMPredictButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // ボタンを無効化して、処理中であることを示す
                VLMPredictButton.IsEnabled = false;
                
                // 非同期でPredictVLMTagsを呼び出す
                await PredictVLMTagsAsync();
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
            }
        }

        // PredictVLMTagsメソッドを非同期に変更
        private async Task PredictVLMTagsAsync()
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage == null)
            {
                AddMainLogEntry("画像が選択されていません。");
                return;
            }

            AddMainLogEntry("VLM推論を開始します");

            try
            {
                var (generalTags, rating, characters, allTags) = _vlmPredictor.Predict(
                    new BitmapImage(new Uri(selectedImage.ImagePath)),
                    0.35f, // generalThresh
                    false, // generalMcutEnabled
                    0.85f, // characterThresh
                    false  // characterMcutEnabled
                );

                // 結果を表示または処理する
                UpdateVLMTagsDisplay(generalTags, rating, characters, allTags);
            }
            catch (Exception ex)
            {
                AddMainLogEntry($"VLM推論中にエラーが発生しました: {ex.Message}");
                throw;
            }

            AddMainLogEntry("VLM推論が完了しました。");
        }

        // VLMログの更新
        private void UpdateVLMLog(object sender, string log)
        {
            Dispatcher.Invoke(() =>
            {
                VLMLogTextBox.Text = log;
            });
        }

        // 左ペイン: フォルダ選択と画像リスト表示
        private void SelectFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "フォルダを選択してください"
            };

            if (dialog.ShowDialog() == CommonFileDialogResult.Ok)
            {
                _imageInfos = _fileExplorer.GetImageInfos(dialog.FileName);
                ImageListBox.ItemsSource = _imageInfos;
                UpdateAllTags();
                
                // Undo/Redoスタックをクリア
                _undoStack.Clear();
                _redoStack.Clear();
                
                // デバッグ用のメッセージを追加
                AddMainLogEntry($"{_imageInfos.Count}個の画像が見つかりました。");
                AddMainLogEntry($"フォルダを選択しました: {dialog.FileName}");
                AddMainLogEntry("Undo/Redoスタックをクリアしました。");
            }
        }    

        // 中央ペイン: 選択された画像の表示と関連テキストの表示
        private void ImageListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isUpdatingSelection) return;

            if (ImageListBox.SelectedItem is ImageInfo selectedImage)
            {
                try
                {
                    _isUpdatingSelection = true;
                    SelectedImage.Source = new BitmapImage(new Uri(selectedImage.ImagePath));
                    AssociatedText.Text = selectedImage.AssociatedText;
                    _currentImageTags = new HashSet<string>(selectedImage.Tags);
                    UpdateTagListView();
                    UpdateAllTagsListView();
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
        }

        // 右ペイン1: 現在の画像のタグリスト表示と選択
        // タグリストビューの更新
        private void UpdateTagListView()
        {
            AddDebugLogEntry("UpdateTagListView");
            _isUpdatingSelection = true;
            try
            {
                var currentTags = _currentImageTags.ToList();
                TagListView.ItemsSource = currentTags;

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
            if (_isUpdatingSelection) return;

            _isUpdatingSelection = true;
            try
            {
                foreach (string removedTag in e.RemovedItems)
                {
                    _selectedTags.Remove(removedTag);
                }

                foreach (string addedTag in e.AddedItems)
                {
                    _selectedTags.Add(addedTag);
                }

                UpdateAllTagsListView();
                UpdateSelectedTagsListBox();
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        // 右ペイン2: 全タグリストの表示、選択、ソート
        // 全タグリストビューの更新
        private void UpdateAllTagsListView()
        {
            AddDebugLogEntry("UpdateAllTagsListView");
            var sortedTags = _allTags
                .Select(kvp => new
                {
                    Tag = kvp.Key,
                    Count = kvp.Value,
                    IsSelected = _selectedTags.Contains(kvp.Key),
                    IsCurrentImageTag = _currentImageTags.Contains(kvp.Key)
                })
                .OrderByDescending(item => item.IsSelected)
                .ThenByDescending(item => item.Count)
                .ThenBy(item => item.Tag)
                .ToList();

            AllTagsListView.SelectionChanged -= AllTagsListView_SelectionChanged;
            AllTagsListView.ItemsSource = sortedTags;
            // 選択状態を更新
            AllTagsListView.SelectedItems.Clear();
            var selectedItems = AllTagsListView.Items.Cast<dynamic>()
                .Where(item => _selectedTags.Contains(item.Tag))
                .ToList();
            foreach (var item in selectedItems)
            {
                AllTagsListView.SelectedItems.Add(item);
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

                UpdateAllTagsListView();
                UpdateTagListView();  // 個別タグリストを更新
                UpdateSelectedTagsListBox();
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        // このメソッドは、AllTagsListViewの選択状態を更新します。
        private void UpdateAllTagsSelection(IEnumerable<string> selectedTags)
        {
            var selectedTagsSet = new HashSet<string>(selectedTags);
            AllTagsListView.SelectedItems.Clear();
            foreach (var item in AllTagsListView.Items)
            {
                if (selectedTagsSet.Contains(((dynamic)item).Tag))
                {
                    AllTagsListView.SelectedItems.Add(item);
                }
            }
        }

        // 右ペイン3: 選択されたタグの表示
        private void UpdateSelectedTagsListBox()
        {
            SelectedTagsListBox.ItemsSource = _selectedTags.ToList();
        }

        // 共通: タグの選択状態更新
        private void UpdateTagListViewSelection(IEnumerable<string> selectedTags)
        {
            TagListView.SelectedItems.Clear();
            foreach (var tag in TagListView.Items)
            {
                if (selectedTags.Contains(tag as string))
                {
                    TagListView.SelectedItems.Add(tag);
                }
            }
        }

        // 全タ��の更新
        private void UpdateAllTags()
        {
            _allTags.Clear();
            foreach (var imageInfo in _imageInfos)
            {
                foreach (var tag in imageInfo.Tags)
                {
                    if (_allTags.ContainsKey(tag))
                    {
                        _allTags[tag]++;
                    }
                    else
                    {
                        _allTags[tag] = 1;
                    }
                }
            }

            UpdateCurrentTags();
            UpdateTagListView();
            UpdateAllTagsListView();
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
        // ボタンのクリックイベント
        // 元に戻す
        private void UndoButton_Click(object sender, RoutedEventArgs e)
        {
            if (_undoStack.Count > 0)
            {
                var action = _undoStack.Pop();
                action.UndoAction();
                _redoStack.Push(action);
                UpdateAllTags();
                UpdateButtonStates();
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
                UpdateAllTags();
                UpdateButtonStates();
                AddActionLogItem("やり直し", action.Description);
            }
        }

        // タグの追加
        private void AddTagButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                var selectedTags = SelectedTagsListBox.Items.Cast<string>().ToList();
                var addedTags = new List<string>();

                foreach (var tag in selectedTags)
                {
                    if (!selectedImage.Tags.Contains(tag))
                    {
                        var action = new TagAction
                        {
                            Image = selectedImage,
                            Tag = tag,
                            IsAdd = true,
                            DoAction = () =>
                            {
                                selectedImage.Tags.Add(tag);
                                AddMainLogEntry($"タグ '{tag}' を追加しました");
                            },
                            UndoAction = () =>
                            {
                                selectedImage.Tags.Remove(tag);
                                AddMainLogEntry($"タグ '{tag}' の追加を取り消しました");
                            },
                            Description = $"タグ '{tag}' を追加"
                        };

                        action.DoAction();
                        _undoStack.Push(action);
                        addedTags.Add(tag);
                    }
                }

                if (addedTags.Count > 0)
                {
                    UpdateAllTags();
                    UpdateButtonStates();
                    _redoStack.Clear();
                }
            }
        }

        private void RemoveTagButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedImage = ImageListBox.SelectedItem as ImageInfo;
            if (selectedImage != null)
            {
                var selectedTags = TagListView.SelectedItems.Cast<string>().ToList(); // ImageTagsListBoxをTagListViewに変更
                var removedTags = new List<string>();

                foreach (var tag in selectedTags)
                {
                    var action = new TagAction
                    {
                        Image = selectedImage,
                        Tag = tag,
                        IsAdd = false,
                        DoAction = () =>
                        {
                            selectedImage.Tags.Remove(tag);
                            AddMainLogEntry($"タグ '{tag}' を削除しました");
                            UpdateTagListView();
                            UpdateAllTags();
                        },
                        UndoAction = () =>
                        {
                            selectedImage.Tags.Add(tag);
                            AddMainLogEntry($"タグ '{tag}' の削除を取り消しました");
                            UpdateTagListView();
                            UpdateAllTags();
                        },
                        Description = $"タグ '{tag}' を削除"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    removedTags.Add(tag);
                }

                if (removedTags.Count > 0)
                {
                    AddMainLogEntry($"{removedTags.Count}個のタグを削除しました。");
                    UpdateAllTags();
                    UpdateButtonStates();
                    _redoStack.Clear();
                }
            }
        }

        private void RemoveAllTagsButton_Click(object sender, RoutedEventArgs e)
        {
            var result = MessageBox.Show("選択したタグをすべての画像から削除しますか？", "確認", MessageBoxButton.YesNo, MessageBoxImage.Warning);
            if (result == MessageBoxResult.Yes)
            {
                var selectedTags = AllTagsListView.SelectedItems.Cast<dynamic>().Select(item => item.Tag as string).ToList();
                var removedTags = new Dictionary<ImageInfo, List<string>>();

                foreach (var imageInfo in _imageInfos)
                {
                    var tagsToRemove = imageInfo.Tags.Intersect(selectedTags).ToList();
                    if (tagsToRemove.Count > 0)
                    {
                        removedTags[imageInfo] = tagsToRemove;
                        foreach (var tag in tagsToRemove)
                        {
                            imageInfo.Tags.Remove(tag);
                        }
                    }
                }
                AddMainLogEntry($"{removedTags.Count}個のタグを削除しました。");

                if (removedTags.Count > 0)
                {
                    var action = new TagAction
                    {
                        DoAction = () =>
                        {
                            foreach (var kvp in removedTags)
                            {
                                foreach (var tag in kvp.Value)
                                {
                                    kvp.Key.Tags.Remove(tag);
                                }
                            }
                            AddMainLogEntry($"{removedTags.Count}個のタグを削除しました。");
                            UpdateTagListView();
                            UpdateAllTags();
                        },
                        UndoAction = () =>
                        {
                            foreach (var kvp in removedTags)
                            {
                                kvp.Key.Tags.AddRange(kvp.Value);
                            }
                            AddMainLogEntry($"{removedTags.Count}個のタグを復元しました。");
                            UpdateTagListView();
                            UpdateAllTags();
                        }
                    };
                    _undoStack.Push(action);
                    _redoStack.Clear();
                    UpdateTagListView();
                    UpdateAllTags();
                    UpdateButtonStates();
                }
            }
        }

        private void UpdateButtonStates()
        {
            UndoButton.IsEnabled = _undoStack.Count > 0;
            RedoButton.IsEnabled = _redoStack.Count > 0;
        }

        // デバッグログを追加するメソッド
        private void AddDebugLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            _debugLogEntries.Insert(0, logMessage);
            while (_debugLogEntries.Count > MaxLogEntries)
            {
                _debugLogEntries.RemoveAt(_debugLogEntries.Count - 1);
            }
            DebugLogTextBox.Text = string.Join(Environment.NewLine, _debugLogEntries);
        }

        // ログを追加するメソッド
        private void AddMainLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            _logEntries.Insert(0, logMessage);
            while (_logEntries.Count > MaxLogEntries)
            {
                _logEntries.RemoveAt(_logEntries.Count - 1);
            }
            // TextBoxに直接ログを追加
            MainLogTextBox.Text = string.Join(Environment.NewLine, _logEntries);
        }

        // アクションログを追加するメソッド
        private void AddActionLogItem(string actionType, string description)
        {
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

        // TagActionクラスを修正
        private class TagAction
        {
            public ImageInfo Image { get; set; }
            public string Tag { get; set; }
            public bool IsAdd { get; set; }
            public Action DoAction { get; set; }
            public Action UndoAction { get; set; }
            public string Description { get; set; }
        }

        private void SaveTagsButton_Click(object sender, RoutedEventArgs e)
        {
            foreach (var imageInfo in _imageInfos)
            {
                SaveTagsToFile(imageInfo);
            }
            MessageBox.Show("すべての画像のタグを保存しまた。", "保存完了", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void SaveTagsToFile(ImageInfo imageInfo)
        {
            string textFilePath = System.IO.Path.ChangeExtension(imageInfo.ImagePath, ".txt");
            string tagString = string.Join(", ", imageInfo.Tags);
            
            try
            {
                File.WriteAllText(textFilePath, tagString);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"ファイルの保存中にエラーが発生しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                AddMainLogEntry($"タグの保存に失敗: {System.IO.Path.GetFileName(imageInfo.ImagePath)} - {ex.Message}");
            }
        }

        // VLMでタグを作成
        private void MakeTagsWithVLM_Click(object sender, RoutedEventArgs e)
        {
            AddDebugLogEntry("MakeTagsWithVLM_Click");

            // 警告を表示
            MessageBox.Show("VLMでタグを作成します。\nこの操作は時間がかかります。", "VLMタグ作成", MessageBoxButton.OK, MessageBoxImage.Warning);

            // VLMを使用してタグを作成するロジックを実装
            MessageBox.Show("VLMでタグを作成しました。", "VLMタグ作成", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private ListBoxItem dragLbi;
        private int? dragIndex;
        private Point? startPos;
        private DataObject dragData;

        private void CleanDragDropData()
        {
            dragLbi = null;
            dragIndex = null;
            startPos = null;
            dragData = null;
        }

        private void TagListView_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            AddDebugLogEntry("TagListView_PreviewMouseLeftButtonDown");
            _startPoint = e.GetPosition(null);
            var item = (e.OriginalSource as FrameworkElement)?.DataContext;
            if (item != null)
            {
                _draggedItem = (sender as ListView)?.ItemContainerGenerator.ContainerFromItem(item) as ListViewItem;
            }
        }

        private void TagListView_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            AddDebugLogEntry("TagListView_PreviewMouseMove");
            if (_startPoint == null || _draggedItem == null) return;

            Point currentPosition = e.GetPosition(null);
            Vector diff = currentPosition - _startPoint.Value;

            if (e.LeftButton == MouseButtonState.Pressed &&
                (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance + 2 ||
                 Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance + 2))
            {
                DragDrop.DoDragDrop(_draggedItem, _draggedItem.DataContext, DragDropEffects.Move);
                _startPoint = null;
                _draggedItem = null;
            }
        }

        private void AllTagsListView_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            AddDebugLogEntry("AllTagsListView_PreviewMouseLeftButtonDown");
            _startPoint = e.GetPosition(null);
            var item = (e.OriginalSource as FrameworkElement)?.DataContext;
            if (item != null)
            {
                _draggedItem = (sender as ListView)?.ItemContainerGenerator.ContainerFromItem(item) as ListViewItem;
            }
        }

        private void AllTagsListView_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            AddDebugLogEntry("AllTagsListView_PreviewMouseMove");
            if (_startPoint == null || _draggedItem == null) return;

            Point currentPosition = e.GetPosition(null);
            Vector diff = currentPosition - _startPoint.Value;

            if (e.LeftButton == MouseButtonState.Pressed &&
                (Math.Abs(diff.X) > SystemParameters.MinimumHorizontalDragDistance + 2 ||
                 Math.Abs(diff.Y) > SystemParameters.MinimumVerticalDragDistance + 2))
            {
                DragDrop.DoDragDrop(_draggedItem, _draggedItem.DataContext, DragDropEffects.Copy);
                _startPoint = null;
                _draggedItem = null;
            }
        }
    }
}