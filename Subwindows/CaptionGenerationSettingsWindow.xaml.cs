using System;
using System.Windows;

namespace tagmane
{
    public partial class CaptionGenerationSettingsWindow : Window
    {
        public CaptionGenerationSettings Settings { get; private set; }
        
        public CaptionGenerationSettingsWindow(CaptionGenerationSettings currentSettings = null)
        {
            InitializeComponent();
            
            // 現在の設定を適用、なければデフォルト値を使用
            Settings = currentSettings ?? new CaptionGenerationSettings();
            LoadSettings();
        }
        
        private void LoadSettings()
        {
            SystemPromptTextBox.Text = Settings.SystemPrompt;
            UserPromptTextBox.Text = Settings.UserPrompt;
            MaxTokensSlider.Value = Settings.MaxTokens;
            TemperatureSlider.Value = Settings.Temperature;
            TopPSlider.Value = Settings.TopP;
            UseExistingTagsCheckBox.IsChecked = Settings.UseExistingTags;
            UseCharacterTagsCheckBox.IsChecked = Settings.UseCharacterTags;
            UseCopyrightTagsCheckBox.IsChecked = Settings.UseCopyrightTags;
            UseGeneralTagsCheckBox.IsChecked = Settings.UseGeneralTags;
            UseArtistTagsCheckBox.IsChecked = Settings.UseArtistTags;
            UseRatingTagsCheckBox.IsChecked = Settings.UseRatingTags;
            UseQualityTagsCheckBox.IsChecked = Settings.UseQualityTags;
            UseMetaTagsCheckBox.IsChecked = Settings.UseMetaTags;
            UseModelTagsCheckBox.IsChecked = Settings.UseModelTags;
            DetailedDescriptionCheckBox.IsChecked = Settings.DetailedDescription;
            
            UpdateValueLabels();
            UpdateTagOptionsEnabled();
        }
        
        private void SaveSettings()
        {
            Settings.SystemPrompt = SystemPromptTextBox.Text;
            Settings.UserPrompt = UserPromptTextBox.Text;
            Settings.MaxTokens = (int)MaxTokensSlider.Value;
            Settings.Temperature = TemperatureSlider.Value;
            Settings.TopP = TopPSlider.Value;
            Settings.UseExistingTags = UseExistingTagsCheckBox.IsChecked ?? true;
            Settings.UseCharacterTags = UseCharacterTagsCheckBox.IsChecked ?? true;
            Settings.UseCopyrightTags = UseCopyrightTagsCheckBox.IsChecked ?? true;
            Settings.UseGeneralTags = UseGeneralTagsCheckBox.IsChecked ?? true;
            Settings.UseArtistTags = UseArtistTagsCheckBox.IsChecked ?? false;
            Settings.UseRatingTags = UseRatingTagsCheckBox.IsChecked ?? false;
            Settings.UseQualityTags = UseQualityTagsCheckBox.IsChecked ?? false;
            Settings.UseMetaTags = UseMetaTagsCheckBox.IsChecked ?? true;
            Settings.UseModelTags = UseModelTagsCheckBox.IsChecked ?? false;
            Settings.DetailedDescription = DetailedDescriptionCheckBox.IsChecked ?? false;
        }
        
        private void UpdateValueLabels()
        {
            MaxTokensValueText.Text = ((int)MaxTokensSlider.Value).ToString();
            TemperatureValueText.Text = TemperatureSlider.Value.ToString("F1");
            TopPValueText.Text = TopPSlider.Value.ToString("F1");
        }
        
        private void MaxTokensSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (MaxTokensValueText != null)
                MaxTokensValueText.Text = ((int)e.NewValue).ToString();
        }
        
        private void TemperatureSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TemperatureValueText != null)
                TemperatureValueText.Text = e.NewValue.ToString("F1");
        }
        
        private void TopPSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TopPValueText != null)
                TopPValueText.Text = e.NewValue.ToString("F1");
        }
        
        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            Settings = new CaptionGenerationSettings(); // デフォルト値でリセット
            LoadSettings();
        }
        
        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            SaveSettings();
            DialogResult = true;
            Close();
        }
        
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
        
        private void UseExistingTagsCheckBox_CheckedChanged(object sender, RoutedEventArgs e)
        {
            UpdateTagOptionsEnabled();
        }
        
        private void UpdateTagOptionsEnabled()
        {
            if (UseExistingTagsCheckBox == null || UseCharacterTagsCheckBox == null || 
                UseCopyrightTagsCheckBox == null || UseGeneralTagsCheckBox == null ||
                UseArtistTagsCheckBox == null || UseRatingTagsCheckBox == null ||
                UseQualityTagsCheckBox == null || UseMetaTagsCheckBox == null ||
                UseModelTagsCheckBox == null)
                return;
                
            bool isEnabled = UseExistingTagsCheckBox.IsChecked ?? false;
            UseCharacterTagsCheckBox.IsEnabled = isEnabled;
            UseCopyrightTagsCheckBox.IsEnabled = isEnabled;
            UseGeneralTagsCheckBox.IsEnabled = isEnabled;
            UseArtistTagsCheckBox.IsEnabled = isEnabled;
            UseRatingTagsCheckBox.IsEnabled = isEnabled;
            UseQualityTagsCheckBox.IsEnabled = isEnabled;
            UseMetaTagsCheckBox.IsEnabled = isEnabled;
            UseModelTagsCheckBox.IsEnabled = isEnabled;
        }
    }
    

    // キャプション生成設定を保持するクラス
    public class CaptionGenerationSettings
    {
        public string SystemPrompt { get; set; } = "You are a helpful assistant that describes images accurately and concisely.";
        public string UserPrompt { get; set; } = "Please describe this image in detail, focusing on the main subjects, their actions, the setting, and any notable details.";
        public int MaxTokens { get; set; } = 2048;
        public double Temperature { get; set; } = 0.7;
        public double TopP { get; set; } = 0.9;
        public bool UseExistingTags { get; set; } = true;
        public bool UseCharacterTags { get; set; } = true;
        public bool UseCopyrightTags { get; set; } = true;
        public bool UseGeneralTags { get; set; } = true;
        public bool UseArtistTags { get; set; } = false;
        public bool UseRatingTags { get; set; } = false;
        public bool UseQualityTags { get; set; } = false;
        public bool UseMetaTags { get; set; } = true;
        public bool UseModelTags { get; set; } = false;
        public bool DetailedDescription { get; set; } = false;
        
        // 設定のコピーを作成するメソッド
        public CaptionGenerationSettings Clone()
        {
            return new CaptionGenerationSettings
            {
                SystemPrompt = this.SystemPrompt,
                UserPrompt = this.UserPrompt,
                MaxTokens = this.MaxTokens,
                Temperature = this.Temperature,
                TopP = this.TopP,
                UseExistingTags = this.UseExistingTags,
                UseCharacterTags = this.UseCharacterTags,
                UseCopyrightTags = this.UseCopyrightTags,
                UseGeneralTags = this.UseGeneralTags,
                UseArtistTags = this.UseArtistTags,
                UseRatingTags = this.UseRatingTags,
                UseQualityTags = this.UseQualityTags,
                UseMetaTags = this.UseMetaTags,
                UseModelTags = this.UseModelTags,
                DetailedDescription = this.DetailedDescription
            };
        }
    }
}