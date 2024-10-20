using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace tagmane
{
    public class WebPHandler
    {
        private string _webpDllPath;

        public WebPHandler(string webpDllPath)
        {
            _webpDllPath = webpDllPath;
        }

        public BitmapSource LoadWebPImage(string imagePath)
        {
            if (string.IsNullOrEmpty(_webpDllPath) || !File.Exists(_webpDllPath))
            {
                throw new FileNotFoundException($"WebP.dllが見つかりません。設定で正しいパスを指定してください。現在のパス: {_webpDllPath}");
            }

            byte[] webpData = File.ReadAllBytes(imagePath);

            int width, height;
            IntPtr outputBuffer = IntPtr.Zero;
            try
            {
                // WebP画像の情報を取得
                IntPtr sizeInfo = NativeMethods.WebPGetInfo(webpData, webpData.Length, out width, out height);
                if (sizeInfo == IntPtr.Zero)
                {
                    throw new InvalidOperationException($"WebP画像の情報取得に失敗しました。画像ファイル: {imagePath}");
                }

                Console.WriteLine($"画像サイズ: 幅 {width}, 高さ {height}");

                int stride = width * 4;
                int outputSize = stride * height;

                // 出力バッファを確保
                outputBuffer = Marshal.AllocHGlobal(outputSize);

                // WebP画像をデコード
                IntPtr result = NativeMethods.WebPDecodeBGRAInto(webpData, webpData.Length, outputBuffer, outputSize, stride);
                if (result == IntPtr.Zero)
                {
                    throw new InvalidOperationException($"WebP画像のデコードに失敗しました。画像ファイル: {imagePath}");
                }

                // ピクセルデータをバイト配列にコピー
                byte[] pixelData = new byte[outputSize];
                Marshal.Copy(outputBuffer, pixelData, 0, outputSize);

                return BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, pixelData, stride);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"例外が発生しました: {ex.Message}");
                Console.WriteLine($"スタックトレース: {ex.StackTrace}");
                throw;
            }
            finally
            {
                if (outputBuffer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(outputBuffer);
                }
            }
        }

        private static class NativeMethods
        {
            [DllImport("libwebp.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr WebPGetInfo(
                [In] byte[] data,
                int data_size,
                out int width,
                out int height);

            [DllImport("libwebp.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr WebPDecodeBGRAInto(
                [In] byte[] data,
                int data_size,
                IntPtr output_buffer,
                int output_buffer_size,
                int output_stride);
        }
    }
}
