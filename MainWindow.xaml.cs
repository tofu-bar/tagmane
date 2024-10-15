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

        public MainWindow()
        {
            try
            {
                InitializeComponent();
                _fileExplorer = new FileExplorer();
                _allTags = new Dictionary<string, int>();
                
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
                    UpdateTagListBox();
                    UpdateAllTagsListView();
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
        private void UpdateTagListBox()
        {
            _isUpdatingSelection = true;
            try
            {
                var currentTags = _currentImageTags.ToList();
                TagListBox.ItemsSource = currentTags;

                for (int i = 0; i < currentTags.Count; i++)
                {
                    var item = TagListBox.ItemContainerGenerator.ContainerFromIndex(i) as ListBoxItem;
                    if (item != null)
                    {
                        item.IsSelected = _selectedTags.Contains(currentTags[i]);
                    }
                }
            }
            finally
            {
                _isUpdatingSelection = false;
            }
        }

        private void TagListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
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
                UpdateTagListBox();  // この行を追加
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

        // 共通: タグの選択状態を更新
        private void UpdateTagListBoxSelection(IEnumerable<string> selectedTags)
        {
            TagListBox.SelectedItems.Clear();
            foreach (var tag in TagListBox.Items)
            {
                if (selectedTags.Contains(tag as string))
                {
                    TagListBox.SelectedItems.Add(tag);
                }
            }
        }
    }
}
