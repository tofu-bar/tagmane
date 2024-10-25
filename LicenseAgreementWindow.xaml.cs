using System;
using System.IO;
using System.Windows;

namespace tagmane
{
    public partial class LicenseAgreementWindow : Window
    {
        public string CurrentVersion { get; set; }

        public LicenseAgreementWindow(string currentVersion)
        {
            InitializeComponent();
            CurrentVersion = currentVersion;
            DataContext = this; // バインディングのためにDataContextを設定
            LoadLicenseText();
        }

        private void LoadLicenseText()
        {
            try
            {
                string exePath = System.Reflection.Assembly.GetExecutingAssembly().Location;
                string exeDir = Path.GetDirectoryName(exePath);
                string licenseFilePath = Path.Combine(exeDir, "LICENSE.txt");

                if (File.Exists(licenseFilePath))
                {
                    string licenseText = File.ReadAllText(licenseFilePath);
                    LicenseTextBlock.Text = licenseText;
                }
                else
                {
                    throw new FileNotFoundException("LICENSE.txtファイルが見つかりません。", licenseFilePath);
                }
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
