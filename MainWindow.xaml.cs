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
        private ObservableCollection<string> _logEntries;
        private ObservableCollection<ActionLogItem> _actionLogItems;
        private const int MaxLogEntries = 20; // 100から20に変更

        public MainWindow()
        {
            try
            {
                InitializeComponent();
                _fileExplorer = new FileExplorer();
                _allTags = new Dictionary<string, int>();
                _logEntries = new ObservableCollection<string>();
                _actionLogItems = new ObservableCollection<ActionLogItem>();
                ActionListView.ItemsSource = _actionLogItems;
                
                // デバッグ用のメッセージを追加
                MessageBox.Show("MainWindowが初期化されました。");
                
                // ウィンドウを表示
                this.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"MainWindowの初期化中にエラーが発生しました: {ex.Message}\n\nStackTrace: {ex.StackTrace}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
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
                
                // デバッグ用のメッセージを追加
                MessageBox.Show($"{_imageInfos.Count}個の画像が見つかりました。");
                AddLogEntry($"フォルダを選択しました: {dialog.FileName}");
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
                    AddLogEntry($"画像を選択しました: {System.IO.Path.GetFileName(selectedImage.ImagePath)}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"画像の読み込み中にエラーが発生しました: {ex.Message}");
                }
                finally
                {
                    _isUpdatingSelection = false;
                }
            }
        }

        // 右ペイン1: 現在の画像のタグリスト表示と選択
        private void UpdateTagListView()
        {
            _isUpdatingSelection = true;
            try
            {
                var currentTags = _currentImageTags.ToList();
                TagListView.ItemsSource = currentTags;

                // ItemsSourceを設定した後、UIが更新されるまで少し待つ
                TagListView.UpdateLayout();

                foreach (var tag in currentTags)
                {
                    var item = TagListView.ItemContainerGenerator.ContainerFromItem(tag) as ListViewItem;
                    if (item != null)
                    {
                        item.IsSelected = _selectedTags.Contains(tag);
                    }
                }
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        private void TagListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
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

            UpdateAllTagsListView();
        }

        private void UpdateAllTagsListView()
        {
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

            AllTagsListView.ItemsSource = sortedTags;

            // 選択状態を更新
            AllTagsListView.SelectedItems.Clear();
            foreach (var item in AllTagsListView.Items)
            {
                if (_selectedTags.Contains(((dynamic)item).Tag))
                {
                    AllTagsListView.SelectedItems.Add(item);
                }
            }
        }

        private void AllTagsListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
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
                                AddLogEntry($"タグ '{tag}' を追加しました");
                            },
                            UndoAction = () =>
                            {
                                selectedImage.Tags.Remove(tag);
                                AddLogEntry($"タグ '{tag}' の追加を取り消しました");
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
                            AddLogEntry($"タグ '{tag}' を削除しました");
                        },
                        UndoAction = () =>
                        {
                            selectedImage.Tags.Add(tag);
                            AddLogEntry($"タグ '{tag}' の削除を取り消しました");
                        },
                        Description = $"タグ '{tag}' を削除"
                    };

                    action.DoAction();
                    _undoStack.Push(action);
                    removedTags.Add(tag);
                }

                if (removedTags.Count > 0)
                {
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
                            UpdateTagListView();
                            UpdateAllTags();
                        },
                        UndoAction = () =>
                        {
                            foreach (var kvp in removedTags)
                            {
                                kvp.Key.Tags.AddRange(kvp.Value);
                            }
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

        // ログを追加するメソッド
        private void AddLogEntry(string message)
        {
            string logMessage = $"{DateTime.Now:HH:mm:ss} - {message}";
            _logEntries.Insert(0, logMessage);
            while (_logEntries.Count > MaxLogEntries)
            {
                _logEntries.RemoveAt(_logEntries.Count - 1);
            }
            // TextBoxに直接ログを追加
            LogTextBox.Text = string.Join(Environment.NewLine, _logEntries);
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
    }
}
