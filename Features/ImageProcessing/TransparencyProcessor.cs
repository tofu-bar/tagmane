using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace tagmane.Features.ImageProcessing
{
    public static class TransparencyProcessor
    {
        // 乱数生成用（マルチスレッド環境でも安全な程度に）
        private static readonly Random _rand = new Random();

        /// <summary>
        /// 対象画像群に対して透過部分の塗りつぶし処理を行います。
        /// 対応形式は PNG および WebP です。
        /// </summary>
        /// <param name="images">画像情報のコレクション</param>
        /// <param name="fillColor">基本の塗りつぶし色</param>
        /// <param name="webPHandler">WebP画像用ハンドラー</param>
        /// <param name="logger">ログ出力用デリゲート</param>
        /// <param name="progress">進捗報告用</param>
        /// <param name="enableRandomColorInversion">
        /// trueの場合、各画像ごとに50%の確率で塗りつぶし色を反転します
        /// </param>
        /// <returns>処理した画像の総数</returns>
        public static async Task<int> ProcessImagesAsync(
            IEnumerable<ImageInfo> images, 
            System.Drawing.Color fillColor, 
            WebPHandler webPHandler, 
            Action<string> logger,
            IProgress<double> progress = null,
            bool enableRandomColorInversion = false
        )
        {
            int processedCount = 0;
            int totalCount = images.Count();
            int currentCount = 0;

            foreach (var imageInfo in images)
            {
                currentCount++;
                progress?.Report((double)currentCount / totalCount * 100);

                string ext = Path.GetExtension(imageInfo.ImagePath).ToLower();
                if (ext != ".png" && ext != ".webp")
                    continue;

                logger($"処理中: {imageInfo.ImagePath} ({currentCount}/{totalCount})");
                try
                {
                    await Task.Run(() =>
                    {
                        // 画像ごとに適用する塗りつぶし色を決定
                        System.Drawing.Color effectiveFillColor = fillColor;
                        if (enableRandomColorInversion)
                        {
                            // ロックをかけて乱数取得（シンプルな実装）
                            lock (_rand)
                            {
                                if (_rand.NextDouble() < 0.5)
                                {
                                    effectiveFillColor = System.Drawing.Color.FromArgb(255 - fillColor.R, 255 - fillColor.G, 255 - fillColor.B);
                                    logger($"ランダムに色が反転されました: {fillColor} → {effectiveFillColor} at {imageInfo.ImagePath}");
                                }
                                else
                                {
                                    logger($"ランダム色反転は発生しませんでした: {fillColor} at {imageInfo.ImagePath}");
                                }
                            }
                        }

                        bool hasTransparency = false;

                        // 形式に応じた透過チェック
                        if (ext == ".png")
                        {
                            using (var bitmap = new Bitmap(imageInfo.ImagePath))
                            {
                                hasTransparency = HasTransparency(bitmap);
                            }
                        }
                        else if (ext == ".webp")
                        {
                            // WebPの場合、BitmapSourceを生成して透過チェック
                            var bmpSource = webPHandler.LoadWebPImage(imageInfo.ImagePath);
                            hasTransparency = HasTransparency(bmpSource);

                            if (!hasTransparency)
                            {
                                logger($"透過部分なし: {imageInfo.ImagePath}");
                                return;
                            }

                            logger($"透過部分あり: {imageInfo.ImagePath}");
                            string tempPath = Path.GetTempFileName();
                            string tempPngPath = Path.ChangeExtension(tempPath, ".png");
                            bool success = false;

                            try
                            {
                                // BitmapSourceをPNGとして保存
                                using (var fs = new FileStream(tempPngPath, FileMode.Create))
                                {
                                    BitmapEncoder encoder = new PngBitmapEncoder();
                                    encoder.Frames.Add(BitmapFrame.Create(bmpSource));
                                    encoder.Save(fs);
                                }

                                using (var bitmap = new Bitmap(tempPngPath))
                                {
                                    using (var newBitmap = FillTransparency(bitmap, effectiveFillColor))
                                    {
                                        byte[] encoded = webPHandler.EncodeWebPImage(newBitmap, 75.0f);
                                        File.WriteAllBytes(tempPath, encoded);
                                        success = true;
                                    }
                                }

                                if (success)
                                {
                                    logger($"元ファイル削除開始: {imageInfo.ImagePath}");
                                    File.Delete(imageInfo.ImagePath);
                                    logger("元ファイル削除完了");

                                    logger($"ファイル移動開始: {tempPath} → {imageInfo.ImagePath}");
                                    File.Move(tempPath, imageInfo.ImagePath);
                                    logger("ファイル移動完了");

                                    processedCount++;
                                }
                            }
                            finally
                            {
                                if (File.Exists(tempPngPath)) File.Delete(tempPngPath);
                                if (!success && File.Exists(tempPath)) File.Delete(tempPath);
                            }
                            return;
                        }

                        if (!hasTransparency)
                        {
                            logger($"透過部分なし: {imageInfo.ImagePath}");
                            return;
                        }

                        // PNGの場合の処理
                        logger($"透過部分あり: {imageInfo.ImagePath}");
                        string pngTempPath = Path.GetTempFileName();
                        bool pngSuccess = false;

                        try
                        {
                            using (var bitmap = new Bitmap(imageInfo.ImagePath))
                            {
                                using (var newBitmap = FillTransparency(bitmap, effectiveFillColor))
                                {
                                    newBitmap.Save(pngTempPath, System.Drawing.Imaging.ImageFormat.Png);
                                    pngSuccess = true;
                                }
                            }

                            if (pngSuccess)
                            {
                                logger($"元ファイル削除開始: {imageInfo.ImagePath}");
                                File.Delete(imageInfo.ImagePath);
                                logger("元ファイル削除完了");

                                logger($"ファイル移動開始: {pngTempPath} → {imageInfo.ImagePath}");
                                File.Move(pngTempPath, imageInfo.ImagePath);
                                logger("ファイル移動完了");

                                processedCount++;
                            }
                        }
                        finally
                        {
                            if (!pngSuccess && File.Exists(pngTempPath)) File.Delete(pngTempPath);
                        }
                    });
                }
                catch (Exception ex)
                {
                    logger($"個別処理でエラー: {ex.Message} at {imageInfo.ImagePath}");
                }
            }

            progress?.Report(100);
            return processedCount;
        }

        /// <summary>
        /// 指定したBitmapの透過部分を指定色で塗りつぶした新しいBitmapを生成します。
        /// </summary>
        public static Bitmap FillTransparency(Bitmap original, System.Drawing.Color fillColor)
        {
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);
            using (Graphics g = Graphics.FromImage(newBitmap))
            {
                g.Clear(fillColor);
                g.DrawImage(original, 0, 0, original.Width, original.Height);
            }
            return newBitmap;
        }

        /// <summary>
        /// Bitmapに透過部分があるかどうかをチェックします。
        /// WebPの場合はBitmapSourceから直接チェックします。
        /// </summary>
        public static bool HasTransparency(BitmapSource bitmapSource)
        {
            // ピクセルフォーマットがアルファチャンネルを含むか確認
            if (bitmapSource.Format != PixelFormats.Bgra32 && bitmapSource.Format != PixelFormats.Pbgra32)
                return false;

            int stride = (bitmapSource.PixelWidth * bitmapSource.Format.BitsPerPixel + 7) / 8;
            byte[] pixels = new byte[stride * bitmapSource.PixelHeight];
            bitmapSource.CopyPixels(pixels, stride, 0);

            // アルファチャンネルをチェック (4バイトごとの4番目のバイト)
            for (int i = 3; i < pixels.Length; i += 4)
            {
                if (pixels[i] < 255)
                    return true;
            }
            return false;
        }

        public static bool HasTransparency(Bitmap bitmap)
        {
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    if (bitmap.GetPixel(x, y).A < 255)
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// BitmapSourceをSystem.Drawing.Bitmapに変換するヘルパーメソッドです。
        /// </summary>
        public static Bitmap BitmapFromSource(BitmapSource bitmapsource)
        {
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapsource));
                enc.Save(outStream);
                Bitmap bitmap = new Bitmap(outStream);
                return new Bitmap(bitmap);
            }
        }
    }
} 