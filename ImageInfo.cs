using System.Collections.Generic;
using System.IO;

namespace tagmane
{
    public class ImageInfo
    {
        public string ImagePath { get; set; }
        public string AssociatedText { get; set; }
        public List<string> Tags { get; set; }

        // タグリストからハッシュセットを初回のみ生成しキャッシュするプロパティ
        private HashSet<string> _tagSet;
        public HashSet<string> TagSet
        {
            get
            {
                if (_tagSet == null)
                {
                    _tagSet = new HashSet<string>(Tags);
                }
                return _tagSet;
            }
        }

        public ImageInfo()
        {
            Tags = new List<string>();
        }

        public override string ToString()
        {
            return System.IO.Path.GetFileName(ImagePath);
        }
    }
}
