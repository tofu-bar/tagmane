using System;
using System.IO;
using System.Windows;

namespace tagmane
{
    public partial class LicenseAgreementWindow : Window
    {
        public LicenseAgreementWindow()
        {
            InitializeComponent();
            LoadLicenseText();
        }

        private void LoadLicenseText()
        {
            try
            {
                string licenseText = File.ReadAllText("LICENSE.txt");
                LicenseTextBlock.Text = licenseText;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"ライセンスファイルの読み込みに失敗しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                LicenseTextBlock.Text = "ライセンスファイルを読み込めませんでした。";
            }
        }

        private void AgreeButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
        }
    }
}
