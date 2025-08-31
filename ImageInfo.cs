using System.Collections.Generic;
using System.IO;
using System.ComponentModel;

namespace tagmane
{
    public class ImageInfo : INotifyPropertyChanged
    {
        public string ImagePath { get; set; }
        public string AssociatedText { get; set; }
        public List<string> Tags { get; set; }
        private string _caption = "";
        public string Caption 
        { 
            get => _caption;
            set 
            {
                if (_caption != value)
                {
                    _caption = value;
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Caption)));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

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
