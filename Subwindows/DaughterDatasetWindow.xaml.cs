using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Microsoft.WindowsAPICodePack.Dialogs; // フォルダ選択ダイアログを利用する場合

namespace tagmane.Subwindows
{
    public partial class DaughterDatasetWindow : Window
    {
        // MainWindow 側のGPU並列度の値を流用するためのプロパティ
        public int CPUConcurrencyLimit { get; set; } = Environment.ProcessorCount;

        // 画像情報の一覧（MainWindow から渡す）
        private List<ImageInfo> _imageInfos;

        // キャンセル用の CancellationTokenSource
        private CancellationTokenSource _cts;

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

        // デバッグメッセージをUI上に追加するヘルパーメソッド
        private void AppendDebugMessage(string message)
        {
            Dispatcher.Invoke(() =>
            {
                DebugTextBox.AppendText($"{DateTime.Now:HH:mm:ss} - {message}\n");
                DebugTextBox.ScrollToEnd();
            });
        }

        // OKボタン：長時間処理を非同期実行（キャンセル対応）
        private async void OKButton_Click(object sender, RoutedEventArgs e)
        {
            if (!double.TryParse(SamplingRatioTextBox.Text, out double samplingRatio) || samplingRatio <= 0 || samplingRatio >= 1)
            {
                MessageBox.Show("正しいサンプリング率（0～1の間）を入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 選択されたクラスタリング手法
            string clusteringMethod = ((ComboBoxItem)ClusteringMethodComboBox.SelectedItem).Content.ToString();

            // 保存先ディレクトリの確認
            string targetDirectory = TargetDirectoryTextBox.Text;
            if (string.IsNullOrWhiteSpace(targetDirectory) || !Directory.Exists(targetDirectory))
            {
                MessageBox.Show("有効な保存先ディレクトリを指定してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 進捗バー表示、OKボタン無効、Cancelボタンは有効にしてキャンセルを許可
            ProcessingProgressBar.Visibility = Visibility.Visible;
            OKButton.IsEnabled = false;
            CancelButton.IsEnabled = true;
            DebugTextBox.Clear();

            // 新たな CancellationTokenSource を作成
            _cts = new CancellationTokenSource();

            try
            {
                await Task.Run(() =>
                {
                    AppendDebugMessage("クラスタリング処理開始");
                    // キャンセル対応版のクラスタリング・サンプリング処理
                    List<ImageInfo> sampledImages = PerformClusteringAndSampling(_imageInfos, clusteringMethod, samplingRatio, _cts.Token);

                    // もしキャンセルにより null が返ったなら、早期終了する
                    if (sampledImages == null)
                    {
                        AppendDebugMessage("クラスタリング処理がキャンセルされたため、娘データセットの作成を中断します。");
                        return;
                    }

                    AppendDebugMessage($"クラスタリング・サンプリング完了（サンプル枚数: {sampledImages.Count}）");

                    AppendDebugMessage("ファイルコピー処理開始");
                    int counter = 0;
                    foreach (var image in sampledImages)
                    {
                        _cts.Token.ThrowIfCancellationRequested();

                        string fileName = Path.GetFileName(image.ImagePath);
                        AppendDebugMessage($"画像 [{fileName}] のコピー開始");

                        // 画像ファイルのコピー
                        string imageDestPath = Path.Combine(targetDirectory, fileName);
                        File.Copy(image.ImagePath, imageDestPath, overwrite: true);

                        // タグファイル（画像と同じベースネーム .txt）のコピー
                        string imageDirectory = Path.GetDirectoryName(image.ImagePath);
                        string tagFileName = Path.GetFileNameWithoutExtension(image.ImagePath) + ".txt";
                        string tagSourcePath = Path.Combine(imageDirectory, tagFileName);
                        if (File.Exists(tagSourcePath))
                        {
                            string tagDestPath = Path.Combine(targetDirectory, tagFileName);
                            File.Copy(tagSourcePath, tagDestPath, overwrite: true);
                            AppendDebugMessage($"画像 [{fileName}] に関連するタグファイル [{tagFileName}] のコピー完了");
                        }
                        else
                        {
                            AppendDebugMessage($"画像 [{fileName}] にタグファイルは存在しません");
                        }
                        counter++;
                        AppendDebugMessage($"画像 [{fileName}] のコピー完了（{counter}/{sampledImages.Count}）");
                    }
                    AppendDebugMessage("すべてのファイルコピー処理完了");
                }, _cts.Token);

                // 処理完了後 UIスレッドでメッセージ表示
                Dispatcher.Invoke(() =>
                {
                    MessageBox.Show("娘データセットの作成と保存が完了しました。", "完了", MessageBoxButton.OK, MessageBoxImage.Information);
                    DialogResult = true;
                });
            }
            catch (OperationCanceledException)
            {
                Dispatcher.Invoke(() =>
                {
                    MessageBox.Show("処理がキャンセルされました。", "キャンセル", MessageBoxButton.OK, MessageBoxImage.Warning);
                    DialogResult = false;
                });
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    MessageBox.Show($"画像またはタグファイルのコピーに失敗しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                });
            }
            finally
            {
                // 終了後、進捗バー非表示、ボタン有効化
                Dispatcher.Invoke(() =>
                {
                    ProcessingProgressBar.Visibility = Visibility.Collapsed;
                    OKButton.IsEnabled = true;
                    CancelButton.IsEnabled = true;
                });
                _cts.Dispose();
                _cts = null;
            }
        }

        // Cancelボタンのクリック：キャンセル要求を送信
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            if (_cts != null && !_cts.IsCancellationRequested)
            {
                _cts.Cancel();
                AppendDebugMessage("キャンセル要求が送信されました。");
            }
        }

        /// <summary>
        /// クラスタリングおよびサンプリング処理を実施します。
        /// キャンセルが要求された場合は、以降の処理（OrderByやサンプリング処理）をスキップします。
        /// </summary>
        /// <param name="images">入力画像リスト</param>
        /// <param name="method">クラスタリング手法</param>
        /// <param name="ratio">サンプリング率</param>
        /// <param name="ct">キャンセル用トークン</param>
        /// <returns>処理結果の画像リスト（キャンセルの場合は null）</returns>
        private List<ImageInfo> PerformClusteringAndSampling(List<ImageInfo> images, string method, double ratio, CancellationToken ct)
        {
            Random rnd = new Random();
            int sampleCount = (int)(images.Count * ratio);

            switch (method)
            {
                case "Random":
                    return images.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();

                case "Hierarchical":
                    // 階層クラスタリングを実施し、デンドログラム順の画像リストを取得
                    List<ImageInfo> dendroOrdering = PerformHierarchicalClusteringDendrogram(images, ct);

                    if (dendroOrdering == null) { return null; }
                    
                    // その順番からランダムにサンプル抽出
                    return dendroOrdering.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();

                case "K-Means":
                    MessageBox.Show("K-Means は未実装です。ランダムサンプリングを代わりに実施します。",
                                    "情報", MessageBoxButton.OK, MessageBoxImage.Information);
                    return images.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();

                default:
                    // 万が一、他の文字列が渡された場合のフォールバック処理
                    return images.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();
            }
        }

        /// <summary>
        /// 階層クラスタリングによりデンドログラム順の画像リストを作成します。
        /// 内部の全ペアの距離計算部分をCPU並列処理により高速化し、
        /// マージ処理中はindeterminateであった進捗バーをdeterminateに切り替え、進捗割合を表示します。
        /// キャンセル要求時は例外を再スローせず、null を返します。
        /// </summary>
        /// <param name="images">画像リスト</param>
        /// <param name="ct">キャンセル用トークン</param>
        /// <returns>クラスタリング後の画像リスト（キャンセルされた場合は null）</returns>
        private List<ImageInfo> PerformHierarchicalClusteringDendrogram(List<ImageInfo> images, CancellationToken ct)
        {
            List<ClusterNode> nodes = images.Select(img => new ClusterNode(img)).ToList();
            int initialCount = images.Count;
            double threshold = 0.5;
            int cpuConcurrencyLimit = CPUConcurrencyLimit;
            int mergeIteration = 0;

            // UI更新：進捗バーをdeterminateに設定
            Dispatcher.BeginInvoke(new Action(() =>
            {
                ProcessingProgressBar.IsIndeterminate = false;
                ProcessingProgressBar.Minimum = 0;
                ProcessingProgressBar.Maximum = 100;
            }));

            while (true)
            {
                ct.ThrowIfCancellationRequested();

                double globalMinDistance = double.MaxValue;
                int globalIndex1 = -1;
                int globalIndex2 = -1;
                object lockObj = new object();

                ParallelOptions parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = cpuConcurrencyLimit,
                    CancellationToken = ct
                };

                try
                {
                    Parallel.For(0, nodes.Count,
                        parallelOptions,
                        () => (minDistance: double.MaxValue, index1: -1, index2: -1),
                        (i, state, localMin) =>
                        {
                            for (int j = i + 1; j < nodes.Count; j++)
                            {
                                double d = DistanceBetweenNodes(nodes[i], nodes[j]);
                                if (d < threshold && d < localMin.minDistance)
                                {
                                    localMin = (d, i, j);
                                }
                            }
                            return localMin;
                        },
                        localResult =>
                        {
                            lock (lockObj)
                            {
                                if (localResult.minDistance < globalMinDistance)
                                {
                                    globalMinDistance = localResult.minDistance;
                                    globalIndex1 = localResult.index1;
                                    globalIndex2 = localResult.index2;
                                }
                            }
                        });
                }
                catch (OperationCanceledException)
                {
                    AppendDebugMessage("キャンセルが検出されました（クラスタリング処理中断）。");
                    return null;
                }

                if (globalIndex1 != -1 && globalIndex2 != -1)
                {
                    int firstIndex = Math.Min(globalIndex1, globalIndex2);
                    int secondIndex = Math.Max(globalIndex1, globalIndex2);
                    ClusterNode mergedNode = new ClusterNode(nodes[firstIndex], nodes[secondIndex], globalMinDistance);
                    nodes.RemoveAt(secondIndex);
                    nodes.RemoveAt(firstIndex);
                    nodes.Add(mergedNode);
                    mergeIteration++;

                    double progressPercent = ((double)(initialCount - nodes.Count) / (initialCount - 1)) * 100.0;
                    Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ProcessingProgressBar.Value = progressPercent;
                    }));
                }
                else
                {
                    break;
                }
            }

            Dispatcher.BeginInvoke(new Action(() =>
            {
                ProcessingProgressBar.Value = 100;
            }));

            List<ImageInfo> ordering = new List<ImageInfo>();
            foreach (var node in nodes)
            {
                ordering.AddRange(node.GetLeafImages());
            }
            return ordering;
        }

        /// <summary>
        /// ノード間の距離を、所属する画像間の最小Jaccard距離で定義します。
        /// </summary>
        private double DistanceBetweenNodes(ClusterNode node1, ClusterNode node2)
        {
            double minDistance = double.MaxValue;
            foreach (var img1 in node1.Images)
            {
                foreach (var img2 in node2.Images)
                {
                    double d = JaccardDistance(img1, img2);
                    if (d < minDistance)
                        minDistance = d;
                }
            }
            return minDistance;
        }

        /// <summary>
        /// 2つの画像間のJaccard距離を、ImageInfoにキャッシュされたTagSetを用いて計算します。
        /// タグの共通部分と和集合から類似度を求め、1から減じた値を距離とします。
        /// </summary>
        private double JaccardDistance(ImageInfo a, ImageInfo b)
        {
            var setA = a.TagSet;
            var setB = b.TagSet;

            // unionとintersectionを求める
            var union = new HashSet<string>(setA);
            union.UnionWith(setB);
            if (union.Count == 0)
                return 0.0;

            int intersectionCount = setA.Intersect(setB).Count();
            double similarity = (double)intersectionCount / union.Count;
            return 1.0 - similarity;
        }

        /// <summary>
        /// クラスタノードの定義：クラスタリングの各ノード（葉も内部ノードも）を表現します。
        /// </summary>
        private class ClusterNode
        {
            public ClusterNode Left { get; set; }
            public ClusterNode Right { get; set; }
            public List<ImageInfo> Images { get; set; }
            public double Distance { get; set; }  // マージ時の距離

            // 葉ノードとしてのコンストラクタ
            public ClusterNode(ImageInfo image)
            {
                Images = new List<ImageInfo> { image };
            }

            // 内部ノードとしてのコンストラクタ（左右ノードのマージ）
            public ClusterNode(ClusterNode left, ClusterNode right, double distance)
            {
                Left = left;
                Right = right;
                Distance = distance;
                Images = new List<ImageInfo>();
                Images.AddRange(left.Images);
                Images.AddRange(right.Images);
            }

            /// <summary>
            /// in-orderで葉ノードにある画像リストを取得します。
            /// </summary>
            public List<ImageInfo> GetLeafImages()
            {
                List<ImageInfo> list = new List<ImageInfo>();
                if (Left == null && Right == null)
                {
                    return Images;
                }
                if (Left != null)
                    list.AddRange(Left.GetLeafImages());
                if (Right != null)
                    list.AddRange(Right.GetLeafImages());
                return list;
            }
        }

        // ウィンドウが閉じられた場合もキャンセル要求を送信
        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            if (_cts != null && !_cts.IsCancellationRequested)
            {
                _cts.Cancel();
            }
            base.OnClosing(e);
        }
    }
} 