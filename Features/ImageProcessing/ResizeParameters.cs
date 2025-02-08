using System;

namespace tagmane.Features.ImageProcessing
{
    public class ResizeParameters
    {
        public int TargetWidth { get; set; }
        public int TargetHeight { get; set; }
        public string Mode { get; set; }
        public string OutputFormat { get; set; }
        public string ResampleMode { get; set; }

        /// <summary>
        /// 指定されたサイズと元のサイズから、実際のリサイズ後のサイズを計算します。
        /// </summary>
        public (int width, int height) CalculateNewSize(int originalWidth, int originalHeight)
        {
            double widthRatio = (double)TargetWidth / originalWidth;
            double heightRatio = (double)TargetHeight / originalHeight;
            double ratio;

            switch (Mode)
            {
                case "拡大のみ":
                    if (widthRatio <= 1 && heightRatio <= 1)
                        return (originalWidth, originalHeight);
                    ratio = Math.Max(widthRatio, heightRatio);
                    break;

                case "縮小のみ":
                    if (widthRatio >= 1 && heightRatio >= 1)
                        return (originalWidth, originalHeight);
                    ratio = Math.Min(widthRatio, heightRatio);
                    break;

                case "リサイズ":
                    // 倍率の正負が逆の場合は大きい方に合わせる
                    if ((widthRatio > 1 && heightRatio < 1) || (widthRatio < 1 && heightRatio > 1))
                        ratio = Math.Max(widthRatio, heightRatio);
                    else
                        ratio = Math.Min(widthRatio, heightRatio);
                    break;

                default:
                    throw new ArgumentException($"未対応のリサイズモード: {Mode}");
            }

            return (
                (int)Math.Round(originalWidth * ratio),
                (int)Math.Round(originalHeight * ratio)
            );
        }
    }
} 