using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace tagmane.Features.ImageProcessing
{
    public static class ImageProcessor
    {
        public static async Task<int> ResizeImagesAsync(
            IEnumerable<ImageInfo> images,
            ResizeParameters parameters,
            WebPHandler webPHandler,
            Action<string> logger,
            IProgress<double> progress = null)
        {
            int processedCount = 0;
            int totalCount = images.Count();
            int currentCount = 0;

            foreach (var imageInfo in images)
            {
                currentCount++;
                progress?.Report((double)currentCount / totalCount * 100);

                logger($"処理中: {imageInfo.ImagePath} ({currentCount}/{totalCount})");
                try
                {
                    await Task.Run(() =>
                    {
                        string tempPath = Path.GetTempFileName();
                        bool success = false;

                        try
                        {
                            // 拡張子を取得
                            string ext = Path.GetExtension(imageInfo.ImagePath).ToLower();
                            Bitmap originalBitmap;

                            // WebP画像の場合は専用ハンドラーを使用
                            if (ext == ".webp")
                            {
                                var bitmapSource = webPHandler.LoadWebPImage(imageInfo.ImagePath);
                                originalBitmap = BitmapSourceToBitmap(bitmapSource);
                            }
                            else
                            {
                                // 他の形式の場合は通常の方法で読み込み
                                using (var fs = new FileStream(imageInfo.ImagePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                                {
                                    originalBitmap = new Bitmap(fs);
                                }
                            }

                            using (originalBitmap)
                            {
                                var (newWidth, newHeight) = parameters.CalculateNewSize(
                                    originalBitmap.Width,
                                    originalBitmap.Height
                                );

                                // サイズが変わらない場合はスキップ
                                if (newWidth == originalBitmap.Width && newHeight == originalBitmap.Height)
                                {
                                    logger($"リサイズ不要: {imageInfo.ImagePath}");
                                    return;
                                }

                                logger($"リサイズ実行: {originalBitmap.Width}x{originalBitmap.Height} → {newWidth}x{newHeight}");

                                // リサイズの実行
                                using (var newBitmap = parameters.ResampleMode.StartsWith("Lanczos")
                                    ? LanczosResizer.ResizeImage(originalBitmap, newWidth, newHeight, parameters.ResampleMode)
                                    : new Bitmap(newWidth, newHeight))
                                {
                                    if (!parameters.ResampleMode.StartsWith("Lanczos"))
                                    {
                                        using (var g = Graphics.FromImage(newBitmap))
                                        {
                                            g.InterpolationMode = InterpolationMode.Bilinear;
                                            g.DrawImage(originalBitmap, 0, 0, newWidth, newHeight);
                                        }
                                    }

                                    // 形式に応じて保存
                                    switch (parameters.OutputFormat)
                                    {
                                        case "PNG":
                                            newBitmap.Save(tempPath, System.Drawing.Imaging.ImageFormat.Png);
                                            break;
                                        case "JPEG":
                                            newBitmap.Save(tempPath, System.Drawing.Imaging.ImageFormat.Jpeg);
                                            break;
                                        case "WebP":
                                            // WebPの場合は直接WebPHandlerを使用してエンコード
                                            byte[] webpData = webPHandler.EncodeWebPImage(newBitmap, 75.0f);
                                            File.WriteAllBytes(tempPath, webpData);
                                            break;
                                        default:
                                            throw new ArgumentException($"未対応の出力形式: {parameters.OutputFormat}");
                                    }
                                }
                            }

                            // 拡張子の更新
                            string newExt = parameters.OutputFormat.ToLower();
                            if (newExt == "jpeg") newExt = "jpg";
                            string targetPath = Path.ChangeExtension(imageInfo.ImagePath, newExt);

                            // ファイルの置き換え
                            if (File.Exists(targetPath))
                                File.Delete(targetPath);
                            File.Move(tempPath, targetPath);

                            // 元のファイルと異なるパスになった場合は元ファイルを削除
                            if (targetPath != imageInfo.ImagePath && File.Exists(imageInfo.ImagePath))
                                File.Delete(imageInfo.ImagePath);

                            // ImageInfoのパスを更新
                            imageInfo.ImagePath = targetPath;

                            success = true;
                            processedCount++;
                        }
                        finally
                        {
                            if (!success && File.Exists(tempPath))
                                File.Delete(tempPath);
                        }
                    });
                }
                catch (Exception ex)
                {
                    logger($"エラー: {imageInfo.ImagePath} - {ex.Message}");
                }
            }

            return processedCount;
        }

        // BitmapSourceをBitmapに変換するヘルパーメソッド
        private static Bitmap BitmapSourceToBitmap(BitmapSource source)
        {
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new PngBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(source));
                enc.Save(outStream);
                outStream.Seek(0, SeekOrigin.Begin);
                return new Bitmap(outStream);
            }
        }
    }
} 