using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

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
                    AssociatedText = GetAssociatedText(imagePath),
                    Caption = GetCaptionFromJson(imagePath)
                };
                // タグの前後のスペースを除去し、空のタグを除外し、エスケープされたかっこを元に戻す
                imageInfo.Tags = imageInfo.AssociatedText
                    .Split(',')
                    .Select(t => ProcessTag(t))
                    .Where(t => !string.IsNullOrWhiteSpace(t))
                    .ToList();

                imageInfos.Add(imageInfo);
            }

            return imageInfos;
        }
        
        private string ProcessTag(string tag)
        {
            // 先頭と末尾のスペースを削除
            tag = tag.Trim();
            // アンダースコアをスペースに置換
            tag = tag.Replace('_', ' ');
            // エスケープされたカッコを戻す
            tag = tag.Replace("\\(", "(").Replace("\\)", ")");
            return tag;
        }

        private string GetAssociatedText(string imagePath)
        {
            // まず .txt ファイルをチェック
            var textPath = Path.ChangeExtension(imagePath, ".txt");
            if (File.Exists(textPath))
            {
                return File.ReadAllText(textPath);
            }
            
            // .txt が存在しない場合は .json をチェック
            var jsonPath = Path.ChangeExtension(imagePath, ".json");
            if (File.Exists(jsonPath))
            {
                try
                {
                    var jsonContent = File.ReadAllText(jsonPath);
                    using var document = JsonDocument.Parse(jsonContent);
                    if (document.RootElement.TryGetProperty("tags", out var tagsProperty))
                    {
                        return tagsProperty.GetString() ?? string.Empty;
                    }
                }
                catch (JsonException)
                {
                    // JSON解析エラーの場合は空文字列を返す
                }
                catch (Exception)
                {
                    // その他のエラーの場合も空文字列を返す
                }
            }
            
            return string.Empty;
        }

        private string GetCaptionFromJson(string imagePath)
        {
            var jsonPath = Path.ChangeExtension(imagePath, ".json");
            if (File.Exists(jsonPath))
            {
                try
                {
                    var jsonContent = File.ReadAllText(jsonPath);
                    using var document = JsonDocument.Parse(jsonContent);
                    if (document.RootElement.TryGetProperty("caption", out var captionProperty))
                    {
                        return captionProperty.GetString() ?? string.Empty;
                    }
                }
                catch (JsonException)
                {
                    // JSON解析エラーの場合は空文字列を返す
                }
                catch (Exception)
                {
                    // その他のエラーの場合も空文字列を返す
                }
            }
            
            return string.Empty;
        }
    }
}
