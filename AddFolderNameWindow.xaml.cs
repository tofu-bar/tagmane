using System;
using System.Windows;
using System.Windows.Controls;

namespace tagmane
{
    public partial class AddFolderNameWindow : Window
    {
        public int DirectoryLevels => (int)DirectoryLevelsSlider.Value;
        public bool FromEnd => FromEndCheckBox.IsChecked ?? false;
        public bool ParseCommas => ParseCommasCheckBox.IsChecked ?? false;
        public bool ApplyToAll => ApplyToAllCheckBox.IsChecked ?? false;
        public double AddProbability => AddProbabilitySlider.Value / 100.0;

        public AddFolderNameWindow(int maxLevels)
        {
            InitializeComponent();
            DirectoryLevelsSlider.Maximum = maxLevels;
            DirectoryLevelsSlider.Value = maxLevels;
        }

        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            if (DirectoryLevels <= 0)
            {
                MessageBox.Show("ディレクトリレベルは1以上を指定してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            DialogResult = true;
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }
    }
}
