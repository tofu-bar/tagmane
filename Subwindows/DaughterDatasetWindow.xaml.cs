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

        // 重み付けオプション（true にするとタグ共通部分に (1 - 出現頻度) の重みを付ける）
        public bool UseWeightedTags { get; set; } = false;

        // 全画像情報に基づくタグ重み（key: タグ, value: 1 - 出現頻度）
        private Dictionary<string, double> tagWeights = new Dictionary<string, double>();

        public DaughterDatasetWindow(List<ImageInfo> imageInfos)
        {
            InitializeComponent();
            _imageInfos = imageInfos;
            if (_imageInfos != null && _imageInfos.Count > 0)
            {
                ComputeTagWeights();
            }
        }

        /// <summary>
        /// 各タグの重み（= 1 - (そのタグの出現回数 / 全画像数)）を計算します。
        /// </summary>
        private void ComputeTagWeights()
        {
            var tagCounts = new Dictionary<string, int>();
            int totalCount = _imageInfos.Count;
            foreach (var image in _imageInfos)
            {
                foreach (var tag in image.TagSet)
                {
                    if (tagCounts.ContainsKey(tag))
                        tagCounts[tag]++;
                    else
                        tagCounts[tag] = 1;
                }
            }
            foreach (var kvp in tagCounts)
            {
                tagWeights[kvp.Key] = 1.0 - ((double)kvp.Value / totalCount);
            }
        }

        /// <summary>
        /// 指定されたタグ集合の重み付き合計を返します。
        /// </summary>
        private double GetWeightedSum(HashSet<string> tagSet)
        {
            return tagSet.Sum(tag => tagWeights.ContainsKey(tag) ? tagWeights[tag] : 1.0);
        }

        /// <summary>
        /// 2つのタグ集合の重み付き共通部分の和を返します。
        /// </summary>
        private double GetWeightedIntersection(HashSet<string> setA, HashSet<string> setB)
        {
            return setA.Intersect(setB).Sum(tag => tagWeights.ContainsKey(tag) ? tagWeights[tag] : 1.0);
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
            // 画像情報が存在しない場合は、エラーメッセージを表示して処理を中断
            if (_imageInfos == null)
            {
                AppendDebugMessage("画像情報が存在しません。");
                return;
            }

            // ここから従来の処理を続行します
            if (!double.TryParse(SamplingRatioSlider.Value.ToString(), out double samplingRatio) ||
                 samplingRatio <= 0 || samplingRatio >= 1)
            {
                AppendDebugMessage("正しいサンプリング率（0～1の間）を入力してください。");
                return;
            }

            // 距離計算方法の選択によって、利用する関数を設定
            string distanceMetric = ((ComboBoxItem)DistanceMetricComboBox.SelectedItem).Content.ToString();
            Func<ImageInfo, ImageInfo, double> distanceFunc;
            switch (distanceMetric)
            {
                case "Jaccard":
                    distanceFunc = JaccardDistance;
                    break;
                case "Dice":
                    distanceFunc = DiceDistance;
                    break;
                case "Simpson":
                    distanceFunc = SimpsonDistance;
                    break;
                default:
                    distanceFunc = JaccardDistance;
                    break;
            }

            // サンプリング方法の選択を取得
            string samplingMethod = ((ComboBoxItem)SamplingMethodComboBox.SelectedItem).Content.ToString();
            
            AppendDebugMessage("距離計算方法: " + distanceMetric);
            AppendDebugMessage("サンプリング方法: " + samplingMethod);

            // サンプリングロジックを、選択されたサンプリング方法に応じた delegate にまとめる
            Func<List<ImageInfo>, int, CancellationToken, List<ImageInfo>> samplingLogic;
            switch (samplingMethod)
            {
                case "Random":
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        Random rnd = new Random();
                        return imgs.OrderBy(x => rnd.Next()).Take(cnt).ToList();
                    };
                    break;
                case "Hierarchical-Fixed":
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        List<ImageInfo> ordering = PerformHierarchicalClusteringDendrogram(imgs, token, distanceFunc);
                        if (ordering == null) { throw new OperationCanceledException(); }
                        return FixedIntervalSampling(ordering, cnt);
                    };
                    break;
                case "Hierarchical-Farthest":
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        List<ImageInfo> ordering = PerformHierarchicalClusteringDendrogram(imgs, token, distanceFunc);
                        if (ordering == null) { throw new OperationCanceledException(); }
                        return FarthestPointSampling(ordering, cnt, distanceFunc);
                    };
                    break;
                case "OptimizedDiversity":
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        return OptimizeDiversitySampling(imgs, cnt, distanceFunc);
                    };
                    break;
                case "KMeans":
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        return KMeansTagSampling(imgs, samplingRatio);
                    };
                    break;
                default:
                    samplingLogic = (imgs, cnt, token) =>
                    {
                        Random rnd = new Random();
                        return imgs.OrderBy(x => rnd.Next()).Take(cnt).ToList();
                    };
                    break;
            }

            // 保存先ディレクトリの確認
            string targetDirectory = TargetDirectoryTextBox.Text;
            if (string.IsNullOrWhiteSpace(targetDirectory) || !Directory.Exists(targetDirectory))
            {
                AppendDebugMessage("有効な保存先ディレクトリを指定してください。");
                return;
            }

            ProcessingProgressBar.Visibility = Visibility.Visible;
            OKButton.IsEnabled = false;
            CancelButton.IsEnabled = true;
            DebugTextBox.Clear();

            _cts = new CancellationTokenSource();

            try
            {
                AppendDebugMessage("サンプリング開始");

                // サンプリングする画像数を計算
                int sampleCount = (int)(_imageInfos.Count * double.Parse(SamplingRatioSlider.Value.ToString()));

                // サンプリング処理を非同期実行
                List<ImageInfo> sampledImages = await Task.Run(() =>
                {
                    return samplingLogic(_imageInfos, sampleCount, _cts.Token);
                }, _cts.Token);

                AppendDebugMessage("サンプリング完了（サンプル枚数: " + sampledImages.Count + "）");
                AppendDebugMessage("ファイルコピー処理開始");
                int counter = 0;
                foreach (var image in sampledImages)
                {
                    _cts.Token.ThrowIfCancellationRequested();

                    string fileName = Path.GetFileName(image.ImagePath);
                    AppendDebugMessage("画像 [" + fileName + "] のコピー開始");

                    string imageDestPath = Path.Combine(targetDirectory, fileName);
                    imageDestPath = GetUniquePath(targetDirectory, image.ImagePath);
                    File.Copy(image.ImagePath, imageDestPath, overwrite: true);

                    string imageDirectory = Path.GetDirectoryName(image.ImagePath);
                    string tagFileName = Path.GetFileNameWithoutExtension(image.ImagePath) + ".txt";
                    string tagSourcePath = Path.Combine(imageDirectory, tagFileName);
                    if (File.Exists(tagSourcePath))
                    {
                        string tagDestPath = Path.Combine(targetDirectory, tagFileName);
                        tagDestPath = GetUniquePath(targetDirectory, tagSourcePath);
                        File.Copy(tagSourcePath, tagDestPath, overwrite: true);
                        AppendDebugMessage("画像 [" + fileName + "] に関連するタグファイル [" + tagFileName + "] のコピー完了");
                    }
                    else
                    {
                        AppendDebugMessage("画像 [" + fileName + "] にタグファイルは存在しません");
                    }
                    counter++;
                    AppendDebugMessage("画像 [" + fileName + "] のコピー完了（" + counter + "/" + sampledImages.Count + "）");
                }
                AppendDebugMessage("すべてのファイルコピー処理完了");

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
                    AppendDebugMessage("処理がキャンセルされました。");
                    DialogResult = false;
                });
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    AppendDebugMessage("画像またはタグファイルのコピーに失敗しました: " + ex.Message);
                });
            }
            finally
            {
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
            else
            {
                // 実行中の処理がない場合は、ウィンドウを閉じる
                this.DialogResult = false; // ダイアログウィンドウならこの設定を行うと、呼び出し元にキャンセルを通知できます
                this.Close();
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
                    // 階層クラスタリングを実施し、デンドログラム順の画像リストを取得（Jaccard 距離を用いる）
                    List<ImageInfo> dendroOrdering = PerformHierarchicalClusteringDendrogram(images, ct, JaccardDistance);
                    if (dendroOrdering == null) { return null; }
                    
                    // 均等に分布するようにサンプルを抽出する (FixedIntervalSampling を使用)
                    return FixedIntervalSampling(dendroOrdering, sampleCount);

                case "OptimizedDiversity":
                    // Jaccard 距離の総和を最大化する（多様性が高い）サンプル群を抽出する
                    return OptimizeDiversitySampling(images, sampleCount, JaccardDistance);

                default:
                    // 万が一、他の文字列が渡された場合のフォールバック処理
                    return images.OrderBy(x => rnd.Next()).Take(sampleCount).ToList();
            }
        }

        /// <summary>
        /// 階層クラスタリングによりデンドログラム順の画像リストを作成します。
        /// 内部の距離計算は distanceFunc を用いて行います。キャンセル要求時は null を返します。
        /// </summary>
        /// <param name="images">画像リスト</param>
        /// <param name="ct">キャンセル用トークン</param>
        /// <param name="distanceFunc">距離計算用の関数</param>
        /// <returns>クラスタリング後の画像リスト（キャンセルされた場合は null）</returns>
        private List<ImageInfo> PerformHierarchicalClusteringDendrogram(List<ImageInfo> images, CancellationToken ct, Func<ImageInfo, ImageInfo, double> distanceFunc)
        {
            List<ClusterNode> nodes = images.Select(img => new ClusterNode(img)).ToList();
            int initialCount = images.Count;
            double threshold = 0.5;
            int cpuConcurrencyLimit = CPUConcurrencyLimit;

            Dispatcher.BeginInvoke(new Action(() =>
            {
                ProcessingProgressBar.IsIndeterminate = false;
                ProcessingProgressBar.Minimum = 0;
                ProcessingProgressBar.Maximum = 100;
            }));

            AppendDebugMessage("クラスタリング開始");

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
                                double d = DistanceBetweenNodes(nodes[i], nodes[j], distanceFunc);
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
        /// ノード間の距離を、所属する画像間の最小距離で定義します。
        /// </summary>
        private double DistanceBetweenNodes(ClusterNode node1, ClusterNode node2, Func<ImageInfo, ImageInfo, double> distanceFunc)
        {
            double minDistance = double.MaxValue;
            foreach (var img1 in node1.Images)
            {
                foreach (var img2 in node2.Images)
                {
                    double d = distanceFunc(img1, img2);
                    if (d < minDistance)
                        minDistance = d;
                }
            }
            return minDistance;
        }

        /// <summary>
        /// タグ集合に対して Jaccard 距離を計算します。重み付けオプションが有効な場合は、
        /// 共通部分、和集合共に各タグの重みの和として計算します。
        /// </summary>
        private double JaccardDistance(ImageInfo a, ImageInfo b)
        {
            var setA = a.TagSet;
            var setB = b.TagSet;
            var union = new HashSet<string>(setA);
            union.UnionWith(setB);
            if (union.Count == 0)
                return 0.0;

            double intersectionValue, unionValue;

            if (UseWeightedTags)
            {
                intersectionValue = GetWeightedIntersection(setA, setB);
                unionValue = union.Sum(tag => tagWeights.ContainsKey(tag) ? tagWeights[tag] : 1.0);
            }
            else
            {
                intersectionValue = setA.Intersect(setB).Count();
                unionValue = union.Count;
            }

            double similarity = intersectionValue / unionValue;
            return 1.0 - similarity;
        }

        /// <summary>
        /// Dice係数に基づく距離を計算します。（距離 = 1 - Dice係数）
        /// 重み付けオプションが有効な場合は、各集合の大きさも重みの和として算出します。
        /// </summary>
        private double DiceDistance(ImageInfo a, ImageInfo b)
        {
            var setA = a.TagSet;
            var setB = b.TagSet;
            double intersectionValue, sumA, sumB;

            if (UseWeightedTags)
            {
                intersectionValue = GetWeightedIntersection(setA, setB);
                sumA = GetWeightedSum(setA);
                sumB = GetWeightedSum(setB);
            }
            else
            {
                intersectionValue = setA.Intersect(setB).Count();
                sumA = setA.Count;
                sumB = setB.Count;
            }
            double similarity = (2.0 * intersectionValue) / (sumA + sumB);
            return 1.0 - similarity;
        }

        /// <summary>
        /// Simpson係数（Overlap coefficient）に基づく距離を計算します。（距離 = 1 - Simpson係数）
        /// 重み付けオプションが有効な場合は、各集合の合計も重みの和として算出します。
        /// </summary>
        private double SimpsonDistance(ImageInfo a, ImageInfo b)
        {
            var setA = a.TagSet;
            var setB = b.TagSet;
            double intersectionValue, weightedA, weightedB;

            if (UseWeightedTags)
            {
                intersectionValue = GetWeightedIntersection(setA, setB);
                weightedA = GetWeightedSum(setA);
                weightedB = GetWeightedSum(setB);
            }
            else
            {
                intersectionValue = setA.Intersect(setB).Count();
                weightedA = setA.Count;
                weightedB = setB.Count;
            }

            double minTotal = Math.Min(weightedA, weightedB);
            if (minTotal == 0)
                return 0.0;
            double similarity = intersectionValue / minTotal;
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

        /// <summary>
        /// Jaccard距離に基づく「最大多様性」サンプリングを実施します。
        /// すなわち、選択されたサンプル集合内の全ペア距離の総和が最大になるような部分集合を Greedy に近似して選びます。
        /// </summary>
        /// <param name="images">候補画像リスト</param>
        /// <param name="sampleCount">選択する画像数</param>
        /// <param name="distanceFunc">距離計算用関数</param>
        /// <returns>多様性が高い画像群</returns>
        private List<ImageInfo> OptimizeDiversitySampling(List<ImageInfo> images, int sampleCount, Func<ImageInfo, ImageInfo, double> distanceFunc)
        {
            AppendDebugMessage("OptimizeDiversitySampling 開始");

            // 早期リターンケースの統一処理
            if (images.Count <= sampleCount || sampleCount == 1)
            {
                Dispatcher.Invoke(() => { ProcessingProgressBar.Value = 100; });
                return sampleCount == 1 
                    ? new List<ImageInfo> { images[0] }
                    : new List<ImageInfo>(images);
            }

            List<ImageInfo> selected = new List<ImageInfo>();
            
            // 初期選択部分（0-20%）は変更なし
            ParallelOptions parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = CPUConcurrencyLimit };
            double maxDistance = -1;
            ImageInfo first = null, second = null;
            int totalComparisons = images.Count * (images.Count - 1) / 2;
            int currentComparison = 0;
            object lockObj = new object();

            AppendDebugMessage("最遠ペア探索開始 並列度 = " + CPUConcurrencyLimit);

            Dispatcher.BeginInvoke(new Action(() =>
            {
                ProcessingProgressBar.IsIndeterminate = false;
                ProcessingProgressBar.Minimum = 0;
                ProcessingProgressBar.Maximum = 100;
            }));

            // 最遠ペア探索を並列化（進捗は0～20%の範囲で更新）
            Parallel.For(0, images.Count, parallelOptions, i =>
            {
                double localMaxDistance = -1;
                ImageInfo localFirst = null, localSecond = null;
                for (int j = i + 1; j < images.Count; j++)
                {
                    double d = distanceFunc(images[i], images[j]);
                    if (d > localMaxDistance)
                    {
                        localMaxDistance = d;
                        localFirst = images[i];
                        localSecond = images[j];
                    }

                    // 進捗更新を追加
                    int comp = Interlocked.Increment(ref currentComparison);
                    if (comp % 100 == 0)  // パフォーマンスのため100回に1回更新
                    {
                        double progressValue = 20.0 * comp / totalComparisons;
                        Dispatcher.Invoke(() => { ProcessingProgressBar.Value = progressValue; });
                    }
                }

                lock (lockObj)
                {
                    if (localMaxDistance > maxDistance)
                    {
                        maxDistance = localMaxDistance;
                        first = localFirst;
                        second = localSecond;
                    }
                }
            });

            if (first == null || second == null)
            {
                Dispatcher.Invoke(() => { ProcessingProgressBar.Value = 100; });
                return images.Take(sampleCount).ToList();
            }
            selected.Add(first);
            selected.Add(second);

            // Greedyアプローチの進捗計算用の変数
            int remainingSelections = sampleCount - 2; // 残り選択回数
            int currentSelection = 0;

            // Greedy アプローチで順次候補を追加（20-100%の範囲で進捗を表示）
            while (selected.Count < sampleCount)
            {
                ImageInfo bestCandidate = null;
                double bestIncrease = -1;
                object lockCandidate = new object();

                Parallel.ForEach(images, parallelOptions, candidate =>
                {
                    if (selected.Contains(candidate))
                        return;
                    double sumDistances = 0;
                    foreach (var sel in selected)
                    {
                        sumDistances += distanceFunc(candidate, sel);
                    }
                    lock (lockCandidate)
                    {
                        if (sumDistances > bestIncrease)
                        {
                            bestIncrease = sumDistances;
                            bestCandidate = candidate;
                        }
                    }
                });

                if (bestCandidate == null)
                    break;

                selected.Add(bestCandidate);
                currentSelection++;

                // 進捗を20-100%の範囲で更新
                Dispatcher.Invoke(() =>
                {
                    double progressValue = 20.0 + (80.0 * currentSelection / remainingSelections);
                    ProcessingProgressBar.Value = Math.Min(progressValue, 100);
                });
            }

            Dispatcher.Invoke(() => { ProcessingProgressBar.Value = 100; });
            return selected;
        }

        /// <summary>
        /// 距離計算に基づく最遠点サンプリングを実施します。
        /// 既に選択された画像集合からの最小距離が最大となる画像を逐次選び、全体のばらつきを高めます。
        /// </summary>
        /// <param name="images">候補画像リスト（例えば、デンドログラム順）</param>
        /// <param name="sampleCount">選択する画像数</param>
        /// <param name="distanceFunc">距離計算用関数</param>
        /// <returns>最遠点サンプリングにより選ばれた画像リスト</returns>
        private List<ImageInfo> FarthestPointSampling(List<ImageInfo> images, int sampleCount, Func<ImageInfo, ImageInfo, double> distanceFunc)
        {
            if (images.Count <= sampleCount)
            {
                return new List<ImageInfo>(images);
            }

            List<ImageInfo> selected = new List<ImageInfo>();
            // 初期候補は先頭の画像（他の初期化方法も検討可能）
            selected.Add(images[0]);

            while (selected.Count < sampleCount)
            {
                ImageInfo candidateCandidate = null;
                double candidateMinDistance = -1;
                foreach (var candidate in images)
                {
                    if (selected.Contains(candidate))
                        continue;
                    double minDistance = double.MaxValue;
                    foreach (var sel in selected)
                    {
                        double d = distanceFunc(candidate, sel);
                        if (d < minDistance)
                            minDistance = d;
                    }
                    if (minDistance > candidateMinDistance)
                    {
                        candidateMinDistance = minDistance;
                        candidateCandidate = candidate;
                    }
                }
                if (candidateCandidate == null)
                    break;
                selected.Add(candidateCandidate);
            }
            return selected;
        }

        /// <summary>
        /// 指定されたリストから、均等間隔にサンプルを抽出します。
        /// 項目数が cnt 未満の場合は、元のリスト全体を返します。
        /// </summary>
        /// <param name="ordering">サンプリング元の画像リスト</param>
        /// <param name="cnt">抽出するサンプル数</param>
        /// <returns>均等間隔に抽出された画像リスト</returns>
        private List<ImageInfo> FixedIntervalSampling(List<ImageInfo> ordering, int cnt)
        {
            int totalCount = ordering.Count;
            if (cnt >= totalCount)
            {
                return ordering;
            }

            List<ImageInfo> sampled = new List<ImageInfo>();
            if (cnt == 1)
            {
                sampled.Add(ordering[0]);
            }
            else
            {
                double step = (double)(totalCount - 1) / (cnt - 1);
                for (int i = 0; i < cnt; i++)
                {
                    int index = (int)Math.Round(i * step);
                    sampled.Add(ordering[index]);
                }
            }
            return sampled;
        }

        /// <summary>
        /// 指定されたターゲットディレクトリに、ソースファイルのオリジナル名称の先頭に
        /// コピー元ディレクトリ名を必ず付与し、同一名称が存在する場合は連番のサフィックスを追加して、一意なパスを返します。
        /// </summary>
        /// <param name="targetDirectory">コピー先ディレクトリ</param>
        /// <param name="sourceFilePath">コピー元のファイルパス</param>
        /// <returns>一意なファイルパス</returns>
        private string GetUniquePath(string targetDirectory, string sourceFilePath)
        {
            string originalName = Path.GetFileName(sourceFilePath);
            string folderName = Path.GetFileName(Path.GetDirectoryName(sourceFilePath));
            
            // 常にディレクトリ名をプレフィックスとして付与する
            string candidate = Path.Combine(targetDirectory, $"{folderName}_{originalName}");
            
            if (!File.Exists(candidate))
                return candidate;
            
            // 衝突が発生した場合は連番のサフィックスを追加して一意な名前にする
            string filenameWithoutExt = Path.GetFileNameWithoutExtension(candidate);
            string ext = Path.GetExtension(candidate);
            int count = 1;
            string newCandidate = candidate;
            while (File.Exists(newCandidate))
            {
                newCandidate = Path.Combine(targetDirectory, $"{filenameWithoutExt}_{count}{ext}");
                count++;
            }
            return newCandidate;
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

        /// <summary>
        /// 選択された画像とその対応するタグファイル（画像と同名のtxtファイル）を、
        /// 対象ディレクトリへ一意の名前でコピーします。
        /// </summary>
        /// <param name="imagesToCopy">コピー対象の画像情報リスト</param>
        /// <param name="targetDirectory">保存先ディレクトリ</param>
        private void CopyFilesWithUniqueNames(List<ImageInfo> imagesToCopy, string targetDirectory)
        {
            foreach (var imageInfo in imagesToCopy)
            {
                try
                {
                    // 画像ファイルのコピー
                    string sourceImagePath = imageInfo.ImagePath;
                    string destImagePath = GetUniquePath(targetDirectory, sourceImagePath);
                    File.Copy(sourceImagePath, destImagePath);

                    // タグファイルのパスを、画像と同名の .txt として取得
                    string tagFilePath = Path.ChangeExtension(imageInfo.ImagePath, ".txt");
                    if (File.Exists(tagFilePath))
                    {
                        string destTagFile = GetUniquePath(targetDirectory, tagFilePath);
                        File.Copy(tagFilePath, destTagFile);
                    }
                }
                catch (Exception ex)
                {
                    // 例外発生時はデバッグエリアへメッセージを表示
                    Dispatcher.Invoke(() =>
                    {
                        AppendDebugMessage("画像またはタグファイルのコピーに失敗しました: " + ex.Message);
                    });
                }
            }
        }

        private List<ImageInfo> KMeansTagSampling(List<ImageInfo> images, double samplingRatio)
        {
            // 全画像のタグ情報から、全タグのインデックスを作成
            Dictionary<string, int> tagIndex = new Dictionary<string, int>();
            foreach (var img in images)
            {
                if (img.Tags == null)
                    continue;
                foreach (var tag in img.Tags)
                {
                    if (!tagIndex.ContainsKey(tag))
                    {
                        tagIndex.Add(tag, tagIndex.Count);
                    }
                }
            }
            
            // クラスタ数の決定: 画像数 × サンプリング率 (最低1クラスタ)
            int k = Math.Max(1, (int)Math.Round(images.Count * samplingRatio));
            AppendDebugMessage($"KMeansTagSampling 開始: クラスタ数 = {k}");
            
            if (k >= images.Count)
            {
                Dispatcher.Invoke(() => { ProcessingProgressBar.Value = 100; });
                return images.OrderBy(x => Guid.NewGuid()).ToList();
            }

            Random rand = new Random();
            
            // 初期セントロイドの選択: ランダムに選んだ k 個の画像のベクトルを使用
            List<double[]> centroids = new List<double[]>();
            var initialCentroids = images.OrderBy(x => rand.Next()).Take(k).ToList();
            foreach (var img in initialCentroids)
            {
                centroids.Add(ExtractTagVector(img, tagIndex));
            }
            
            int maxIterations = 20;
            int[] assignments = new int[images.Count];

            Dispatcher.BeginInvoke(new Action(() =>
            {
                ProcessingProgressBar.IsIndeterminate = false;
                ProcessingProgressBar.Minimum = 0;
                ProcessingProgressBar.Maximum = 100;
            }));

            // k-meansクラスタリングの反復処理（クラスタリング部分で全体の80%の進捗を利用）
            for (int iter = 0; iter < maxIterations; iter++)
            {
                bool assignmentChanged = false;
                // 各画像を最も近いセントロイドに割り当てる
                for (int i = 0; i < images.Count; i++)
                {
                    double minDist = double.MaxValue;
                    int cluster = 0;
                    double[] vec = ExtractTagVector(images[i], tagIndex);
                    for (int j = 0; j < k; j++)
                    {
                        double d = EuclideanDistance(vec, centroids[j]);
                        if (d < minDist)
                        {
                            minDist = d;
                            cluster = j;
                        }
                    }
                    if (assignments[i] != cluster)
                    {
                        assignments[i] = cluster;
                        assignmentChanged = true;
                    }
                }
                
                // 各クラスタについてセントロイドを更新する
                List<double[]> newCentroids = new List<double[]>();
                for (int j = 0; j < k; j++)
                {
                    var clusterMembers = images.Where((img, index) => assignments[index] == j).ToList();
                    if (clusterMembers.Count == 0)
                    {
                        newCentroids.Add(centroids[j]);
                    }
                    else
                    {
                        int dims = tagIndex.Count;
                        double[] avg = new double[dims];
                        foreach (var img in clusterMembers)
                        {
                            double[] vec = ExtractTagVector(img, tagIndex);
                            for (int d = 0; d < dims; d++)
                            {
                                avg[d] += vec[d];
                            }
                        }
                        for (int d = 0; d < dims; d++)
                        {
                            avg[d] /= clusterMembers.Count;
                        }
                        newCentroids.Add(avg);
                    }
                }
                
                double totalShift = 0;
                for (int j = 0; j < k; j++)
                {
                    totalShift += EuclideanDistance(centroids[j], newCentroids[j]);
                }
                centroids = newCentroids;

                // 反復毎に進捗バーを更新 (クラスタリング部分で最大80%まで)
                Dispatcher.Invoke(() =>
                {
                    double progressValue = 80.0 * (iter + 1) / maxIterations;
                    ProcessingProgressBar.Value = Math.Min(progressValue, 80);
                });
                
                if (!assignmentChanged || totalShift < 1e-5)
                    break;
            }
            
            // 各クラスタからランダムに1枚を選出し、最終的にサンプルセットを生成（残り20%の進捗を利用）
            List<ImageInfo> sampled = new List<ImageInfo>();
            for (int j = 0; j < k; j++)
            {
                var clusterMembers = images.Where((img, index) => assignments[index] == j).ToList();
                if (clusterMembers.Any())
                {
                    sampled.Add(clusterMembers[rand.Next(clusterMembers.Count)]);
                }
                // クラスタ毎に進捗バーを更新（80%～100%）
                Dispatcher.Invoke(() =>
                {
                    double progressValue = 80.0 + (20.0 * (j + 1) / k);
                    ProcessingProgressBar.Value = Math.Min(progressValue, 100);
                });
            }
            
            Dispatcher.Invoke(() => { ProcessingProgressBar.Value = 100; });
            return sampled;
        }

        /// <summary>
        /// 画像のタグ情報から、全体タグリストに基づいたOne-Hotの配列を生成します。
        /// </summary>
        /// <param name="image">対象のImageInfo</param>
        /// <param name="tagIndex">全タグとそのインデックスのディクショナリ</param>
        /// <returns>生成されたベクトル</returns>
        private double[] ExtractTagVector(ImageInfo image, Dictionary<string, int> tagIndex)
        {
            double[] vector = new double[tagIndex.Count];
            if (image.Tags != null)
            {
                foreach (var tag in image.Tags)
                {
                    if (tagIndex.TryGetValue(tag, out int idx))
                    {
                        vector[idx] = 1.0;
                    }
                }
            }
            return vector;
        }

        private double EuclideanDistance(double[] vec1, double[] vec2)
        {
            double sum = 0;
            for (int i = 0; i < vec1.Length; i++)
            {
                double d = vec1[i] - vec2[i];
                sum += d * d;
            }
            return Math.Sqrt(sum);
        }
    }
} 