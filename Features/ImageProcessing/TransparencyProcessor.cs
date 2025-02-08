using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Media.Imaging;

namespace tagmane.Features.ImageProcessing
{
    public static class TransparencyProcessor
    {
        /// <summary>
        /// 対象画像群に対して透過部分の塗りつぶし処理を行います。
        /// 対応形式は PNG および WebP です。
        /// </summary>
        public static int ProcessImages(IEnumerable<ImageInfo> images, System.Drawing.Color fillColor, Action<string> logger)
        {
            int processedCount = 0;
            foreach (var imageInfo in images)
            {
                string ext = Path.GetExtension(imageInfo.ImagePath).ToLower();
                if (ext != ".png" && ext != ".webp")
                    continue;

                logger($"処理中: {imageInfo.ImagePath}");
                try
                {
                    string tempPath = Path.GetTempFileName();
                    bool success = false;
                    Bitmap bitmap = null;

                    // 形式に応じた読み込み方法
                    if (ext == ".png")
                    {
                        bitmap = new Bitmap(imageInfo.ImagePath);
                    }
                    else if (ext == ".webp")
                    {
                        var webpHandler = new WebPHandler("libwebp.dll");
                        BitmapSource bmpSource = webpHandler.LoadWebPImage(imageInfo.ImagePath);
                        bitmap = BitmapFromSource(bmpSource);
                    }

                    if (bitmap == null)
                    {
                        logger($"画像読み込み失敗: {imageInfo.ImagePath}");
                        continue;
                    }

                    using (bitmap)
                    {
                        if (HasTransparency(bitmap))
                        {
                            logger($"透過部分あり: {imageInfo.ImagePath}");
                            using (var newBitmap = FillTransparency(bitmap, fillColor))
                            {
                                if (ext == ".png")
                                {
                                    logger($"一時ファイルに保存（PNG）: {tempPath}");
                                    newBitmap.Save(tempPath, System.Drawing.Imaging.ImageFormat.Png);
                                }
                                else if (ext == ".webp")
                                {
                                    logger($"WebPエンコード開始: {imageInfo.ImagePath}");
                                    var webpHandler = new WebPHandler("libwebp.dll");
                                    // 例として品質75を使用
                                    byte[] encoded = webpHandler.EncodeWebPImage(newBitmap, 75.0f);
                                    logger($"WebPエンコード完了: {encoded.Length} バイト");
                                    File.WriteAllBytes(tempPath, encoded);
                                }
                                success = true;
                            }
                        }
                        else
                        {
                            logger($"透過部分なし: {imageInfo.ImagePath}");
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
                    else if (File.Exists(tempPath))
                    {
                        File.Delete(tempPath);
                    }
                }
                catch (Exception ex)
                {
                    logger($"個別処理でエラー: {ex.Message} at {imageInfo.ImagePath}");
                }
            }
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
        /// </summary>
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