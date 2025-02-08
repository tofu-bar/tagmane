using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace tagmane.Features.ImageProcessing
{
    public class WebPHandler
    {
        private string _webpDllPath;

        public WebPHandler(string webpDllPath)
        {
            _webpDllPath = webpDllPath;
        }

        /// <summary>
        /// WebP画像を読み込み、BitmapSourceとして返す。
        /// </summary>
        public BitmapSource LoadWebPImage(string imagePath)
        {
            if (string.IsNullOrEmpty(_webpDllPath) || !File.Exists(_webpDllPath))
            {
                throw new FileNotFoundException($"WebP DLLが見つかりません。指定されたパス: {_webpDllPath}");
            }

            byte[] webpData = File.ReadAllBytes(imagePath);

            int width, height;
            IntPtr outputBuffer = IntPtr.Zero;
            try
            {
                IntPtr sizeInfo = NativeMethods.WebPGetInfo(webpData, webpData.Length, out width, out height);
                if (sizeInfo == IntPtr.Zero)
                {
                    throw new InvalidOperationException($"WebP画像の情報取得に失敗しました。画像ファイル: {imagePath}");
                }

                int stride = width * 4;
                int outputSize = stride * height;

                outputBuffer = Marshal.AllocHGlobal(outputSize);

                IntPtr result = NativeMethods.WebPDecodeBGRAInto(webpData, webpData.Length, outputBuffer, outputSize, stride);
                if (result == IntPtr.Zero)
                {
                    throw new InvalidOperationException($"WebP画像のデコードに失敗しました。画像ファイル: {imagePath}");
                }

                byte[] pixelData = new byte[outputSize];
                Marshal.Copy(outputBuffer, pixelData, 0, outputSize);

                return BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, pixelData, stride);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"例外が発生: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
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

        /// <summary>
        /// 指定したBitmapをWebP形式にエンコードし、byte[]として返す。
        /// </summary>
        public byte[] EncodeWebPImage(System.Drawing.Bitmap bitmap, float qualityFactor)
        {
            // BitmapからBGRAバイト配列に変換
            var rect = new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height);
            System.Drawing.Imaging.BitmapData bmpData = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            int stride = bmpData.Stride;
            int bufferSize = stride * bitmap.Height;
            byte[] pixelData = new byte[bufferSize];
            Marshal.Copy(bmpData.Scan0, pixelData, 0, bufferSize);
            bitmap.UnlockBits(bmpData);

            // libwebp.dll の WebPEncodeBGRA を使用してエンコード
            IntPtr outputPtr;
            UIntPtr outputSize = NativeMethods.WebPEncodeBGRA(pixelData, bitmap.Width, bitmap.Height, stride, qualityFactor, out outputPtr);
            int size = (int)outputSize;
            if (size == 0 || outputPtr == IntPtr.Zero)
                throw new InvalidOperationException("WebP画像のエンコードに失敗しました。");

            byte[] result = new byte[size];
            Marshal.Copy(outputPtr, result, 0, size);
            NativeMethods.WebPFree(outputPtr);
            return result;
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

            [DllImport("libwebp.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern UIntPtr WebPEncodeBGRA(
                [In] byte[] bgra,
                int width,
                int height,
                int stride,
                float qualityFactor,
                out IntPtr output);

            [DllImport("libwebp.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern void WebPFree(IntPtr p);
        }
    }
} 