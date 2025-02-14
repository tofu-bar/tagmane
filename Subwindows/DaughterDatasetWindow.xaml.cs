using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Microsoft.WindowsAPICodePack.Dialogs; // フォルダ選択ダイアログを利用する場合

namespace tagmane.Subwindows
{
    public partial class DaughterDatasetWindow : Window
    {
        // 画像情報の一覧（MainWindow から渡す）
        private List<ImageInfo> _imageInfos;

        public DaughterDatasetWindow(List<ImageInfo> imageInfos)
        {
            InitializeComponent();
            _imageInfos = imageInfos;
        }

        private void BrowseButton_Click(object sender, RoutedEventArgs e)
        {
            // フォルダ選択ダイアログ（Windows API Code Pack などを利用）
            var dlg = new CommonOpenFileDialog
            {
                IsFolderPicker = true
            };

            if (dlg.ShowDialog() == CommonFileDialogResult.Ok)
            {
                TargetDirectoryTextBox.Text = dlg.FileName;
            }
        }

        private void OKButton_Click(object sender, RoutedEventArgs e)
        {
            // サンプリング率のパース
            if (!double.TryParse(SamplingRatioTextBox.Text, out double samplingRatio) || samplingRatio <= 0 || samplingRatio >= 1)
            {
                MessageBox.Show("正しいサンプリング率（0～1の間）を入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // クラスタリング手法の選択
            string clusteringMethod = ((ComboBoxItem)ClusteringMethodComboBox.SelectedItem).Content.ToString();

            // 保存先ディレクトリの確認
            string targetDirectory = TargetDirectoryTextBox.Text;
            if (string.IsNullOrWhiteSpace(targetDirectory) || !Directory.Exists(targetDirectory))
            {
                MessageBox.Show("有効な保存先ディレクトリを指定してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // ★ ここでクラスタリングおよびサンプリング処理を実施 ★
            // サンプル実装：※実際のアルゴリズムは後程詳細に実装してください。
            List<ImageInfo> sampledImages = PerformClusteringAndSampling(_imageInfos, clusteringMethod, samplingRatio);

            // 画像の複製処理
            foreach (var image in sampledImages)
            {
                try
                {
                    // 画像ファイルのコピー
                    string imageDestPath = Path.Combine(targetDirectory, Path.GetFileName(image.ImagePath));
                    File.Copy(image.ImagePath, imageDestPath, overwrite: true);

                    // タグファイルのコピー
                    // ここでは、画像ファイルと同じフォルダにある、拡張子 .txt のタグファイルをコピーする例です。
                    string imageDirectory = Path.GetDirectoryName(image.ImagePath);
                    string tagFileName = Path.GetFileNameWithoutExtension(image.ImagePath) + ".txt";
                    string tagSourcePath = Path.Combine(imageDirectory, tagFileName);
                    if (File.Exists(tagSourcePath))
                    {
                        string tagDestPath = Path.Combine(targetDirectory, tagFileName);
                        File.Copy(tagSourcePath, tagDestPath, overwrite: true);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"画像またはタグファイルのコピーに失敗しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }
            }

            MessageBox.Show("娘データセットの作成と保存が完了しました。", "完了", MessageBoxButton.OK, MessageBoxImage.Information);
            DialogResult = true;
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }

        // 疑似的なクラスタリングとサンプリング処理
        // TODO: 実際のアルゴリズム（例：k-means や階層クラスタリング）を実装する
        private List<ImageInfo> PerformClusteringAndSampling(List<ImageInfo> images, string method, double ratio)
        {
            // 以下は very simple な実装例：全体からランダムにサンプリングする
            int sampleCount = (int)(images.Count * ratio);
            Random rnd = new Random();
            return images.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();
        }
    }
} 