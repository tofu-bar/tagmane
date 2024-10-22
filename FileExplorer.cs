using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace tagmane
{
    public class FileExplorer
    {
        private readonly string[] _imageExtensions = { ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp" };

        public List<ImageInfo> GetImageInfos(string folderPath)
        {
            var imageInfos = new List<ImageInfo>();

            var imageFiles = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
                .Where(file => _imageExtensions.Contains(Path.GetExtension(file).ToLower()));

            foreach (var imagePath in imageFiles)
            {
                var imageInfo = new ImageInfo
                {
                    ImagePath = imagePath,
                    AssociatedText = GetAssociatedText(imagePath)
                };
                // タグの前後のスペースを除去し、空のタグを除外し、エスケープされたかっこを元に戻す
                imageInfo.Tags = imageInfo.AssociatedText
                    .Split(',')
                    .Select(t => t.Trim())
                    .Where(t => !string.IsNullOrWhiteSpace(t))
                    .Select(t => t.Replace("\\(", "(").Replace("\\)", ")"))
                    .ToList();

                imageInfos.Add(imageInfo);
            }

            return imageInfos;
        }

        private string GetAssociatedText(string imagePath)
        {
            var textPath = Path.ChangeExtension(imagePath, ".txt");
            if (File.Exists(textPath))
            {
                return File.ReadAllText(textPath);
            }
            return string.Empty;
        }
    }
}
