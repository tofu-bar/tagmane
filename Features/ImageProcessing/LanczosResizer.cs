using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace tagmane.Features.ImageProcessing
{
    public static class LanczosResizer
    {
        private static double Lanczos(double x, int a)
        {
            if (x == 0.0)
                return 1.0;

            if (Math.Abs(x) >= a)
                return 0.0;

            double piX = Math.PI * x;
            double sincX = Math.Sin(piX) / piX;
            double sincXA = Math.Sin(piX / a) / (piX / a);

            return sincX * sincXA;
        }

        public static Bitmap ResizeImage(Bitmap src, int newWidth, int newHeight, string resampleMode)
        {
            // リサンプルモードに応じてLanczosのパラメータを設定
            int a = resampleMode == "Lanczos3" ? 3 : 2;

            // 元画像のサイズを保存
            int srcWidth = src.Width;
            int srcHeight = src.Height;

            // 元画像のピクセルデータを取得
            var srcRect = new Rectangle(0, 0, srcWidth, srcHeight);
            var srcData = src.LockBits(srcRect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var srcPtr = srcData.Scan0;
            var srcStride = srcData.Stride;
            var srcPixels = new byte[srcStride * srcHeight];
            Marshal.Copy(srcPtr, srcPixels, 0, srcPixels.Length);

            // 新しい画像のピクセルデータを準備
            var dest = new Bitmap(newWidth, newHeight);
            var destRect = new Rectangle(0, 0, newWidth, newHeight);
            var destData = dest.LockBits(destRect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            var destPtr = destData.Scan0;
            var destStride = destData.Stride;
            var destPixels = new byte[destStride * newHeight];

            // スケーリング係数を計算
            double scaleX = (double)srcWidth / newWidth;
            double scaleY = (double)srcHeight / newHeight;

            // 並列処理で新しい画像を生成
            Parallel.For(0, newHeight, j =>
            {
                for (int i = 0; i < newWidth; i++)
                {
                    double sourceX = i * scaleX;
                    double sourceY = j * scaleY;

                    // Lanczos補間を使用してピクセル値を計算
                    double r = 0, g = 0, b = 0, alpha = 0;
                    double totalWeight = 0;

                    int startX = Math.Max(0, (int)(sourceX - a));
                    int endX = Math.Min(srcWidth - 1, (int)(sourceX + a));
                    int startY = Math.Max(0, (int)(sourceY - a));
                    int endY = Math.Min(srcHeight - 1, (int)(sourceY + a));

                    for (int y = startY; y <= endY; y++)
                    {
                        for (int x = startX; x <= endX; x++)
                        {
                            double dx = x - sourceX;
                            double dy = y - sourceY;
                            double weight = Lanczos(dx, a) * Lanczos(dy, a);

                            int srcIndex = y * srcStride + x * 4;
                            b += srcPixels[srcIndex] * weight;
                            g += srcPixels[srcIndex + 1] * weight;
                            r += srcPixels[srcIndex + 2] * weight;
                            alpha += srcPixels[srcIndex + 3] * weight;
                            totalWeight += weight;
                        }
                    }

                    // 計算した値を新しい画像に設定
                    int destIndex = j * destStride + i * 4;
                    if (totalWeight != 0)
                    {
                        destPixels[destIndex] = (byte)Math.Min(255, Math.Max(0, b / totalWeight));
                        destPixels[destIndex + 1] = (byte)Math.Min(255, Math.Max(0, g / totalWeight));
                        destPixels[destIndex + 2] = (byte)Math.Min(255, Math.Max(0, r / totalWeight));
                        destPixels[destIndex + 3] = (byte)Math.Min(255, Math.Max(0, alpha / totalWeight));
                    }
                }
            });

            // 結果をビットマップにコピー
            Marshal.Copy(destPixels, 0, destPtr, destPixels.Length);

            // リソースを解放
            src.UnlockBits(srcData);
            dest.UnlockBits(destData);

            return dest;
        }
    }
} 