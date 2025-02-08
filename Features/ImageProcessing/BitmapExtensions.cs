using System.Drawing;
using System.Drawing.Imaging;

namespace tagmane.Features.ImageProcessing
{
    public static class BitmapExtensions
    {
        public static bool HasAlphaChannel(this Bitmap bitmap)
        {
            return bitmap.PixelFormat == PixelFormat.Format32bppArgb ||
                   bitmap.PixelFormat == PixelFormat.Format32bppPArgb;
        }
    }
} 