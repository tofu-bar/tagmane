using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace tagmane
{
    public partial class CaptionGenerationSettingsWindow : Window
    {
        public CaptionGenerationSettings Settings { get; private set; }
        
        public CaptionGenerationSettingsWindow(CaptionGenerationSettings currentSettings = null)
        {
            System.Diagnostics.Debug.WriteLine("=== CaptionGenerationSettingsWindow Constructor called ===");
            InitializeComponent();
            
            // 現在の設定を適用、なければデフォルト値を使用
            Settings = currentSettings ?? new CaptionGenerationSettings();
            
            // まず基本設定をロード
            LoadSettings();
            
            // キャッシュされたGPUリストを読み込む
            LoadGpuListFromCache();
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
            
            // GPU選択の設定はLoadGpuList()で処理済み
            
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
            
            // GPU ID を保存
            Settings.GpuId = Math.Max(0, GpuSelectionComboBox.SelectedIndex);
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

        private void LoadGpuListFromCache()
        {
            System.Diagnostics.Debug.WriteLine("=== LoadGpuListFromCache() called ===");
            
            GpuSelectionComboBox.Items.Clear();
            
            // GpuManagerからGPUリストを取得
            var gpus = GpuManager.AvailableGpus;
            
            if (gpus.Count == 0)
            {
                // GPUリストが空の場合は検出中メッセージを表示
                GpuSelectionComboBox.Items.Add("GPU 0: 検出中...");
                GpuSelectionComboBox.SelectedIndex = 0;
                
                // 非同期でGPU初期化を待ってから再読み込み
                _ = WaitForGpuInitializationAsync();
            }
            else
            {
                // GPUリストをコンボボックスに追加
                foreach (var gpu in gpus)
                {
                    GpuSelectionComboBox.Items.Add(gpu.DisplayName);
                }
                
                // 保存された設定に基づいて選択
                int targetIndex = Math.Min(Settings.GpuId, GpuSelectionComboBox.Items.Count - 1);
                targetIndex = Math.Max(0, targetIndex);
                GpuSelectionComboBox.SelectedIndex = targetIndex;
                
                System.Diagnostics.Debug.WriteLine($"GPU ComboBox loaded with {gpus.Count} cached GPUs");
            }
        }
        
        private async Task WaitForGpuInitializationAsync()
        {
            // GPU初期化を待つ
            while (!GpuManager.IsInitialized)
            {
                await Task.Delay(100);
            }
            
            // 初期化完了後、UIスレッドでコンボボックスを更新
            await Dispatcher.InvokeAsync(() =>
            {
                LoadGpuListFromCache();
            });
        }
        
        private bool _isLoadingGpu = false;
        
        private async Task LoadGpuListAsync()
        {
            System.Diagnostics.Debug.WriteLine("=== LoadGpuList() called ===");
            
            // 重複実行を防ぐ
            if (_isLoadingGpu)
            {
                System.Diagnostics.Debug.WriteLine("LoadGpuList already in progress, skipping...");
                return;
            }
            _isLoadingGpu = true;
            
            try
            {
                System.Diagnostics.Debug.WriteLine($"ComboBox items before clear: {GpuSelectionComboBox.Items.Count}");
                GpuSelectionComboBox.Items.Clear();
                
                // Python スクリプトでGPU情報を取得
                string appDir = AppDomain.CurrentDomain.BaseDirectory;
                string pythonExe = System.IO.Path.Combine(appDir, ".venv", "Scripts", "python.exe");
                string scriptPath = System.IO.Path.Combine(appDir, "caption_generator.py");

                bool gpuDetected = false;

                if (System.IO.File.Exists(pythonExe) && System.IO.File.Exists(scriptPath))
                {
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = pythonExe,
                        Arguments = $"\"{scriptPath}\" --list-gpus",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = appDir
                    };

                    using (var process = new Process { StartInfo = startInfo })
                    {
                        process.Start();
                        string output = await process.StandardOutput.ReadToEndAsync();
                        string errorOutput = await process.StandardError.ReadToEndAsync();
                        await process.WaitForExitAsync();

                        System.Diagnostics.Debug.WriteLine($"GPU detection output: {output}");
                        System.Diagnostics.Debug.WriteLine($"GPU detection error: {errorOutput}");

                        if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                        {
                            var lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                            var gpuEntries = new HashSet<string>(); // 重複を防ぐ
                            
                            System.Diagnostics.Debug.WriteLine($"Total output lines: {lines.Length}");

                            foreach (var line in lines)
                            {
                                var trimmedLine = line.Trim();
                                System.Diagnostics.Debug.WriteLine($"Processing line: '{trimmedLine}'");
                                
                                // "GPU X:" で始まる行のみを処理
                                if (trimmedLine.StartsWith("GPU ") && trimmedLine.Contains(":"))
                                {
                                    System.Diagnostics.Debug.WriteLine($"Adding GPU entry: '{trimmedLine}'");
                                    bool added = gpuEntries.Add(trimmedLine);
                                    System.Diagnostics.Debug.WriteLine($"Entry added to set: {added}");
                                    gpuDetected = true;
                                }
                            }

                            System.Diagnostics.Debug.WriteLine($"Unique GPU entries found: {gpuEntries.Count}");
                            
                            // 重複を除いた項目をComboBoxに追加
                            foreach (var entry in gpuEntries.OrderBy(x => x))
                            {
                                System.Diagnostics.Debug.WriteLine($"Adding to ComboBox: '{entry}'");
                                GpuSelectionComboBox.Items.Add(entry);
                            }
                        }
                    }
                }

                // GPU情報が取得できない場合はデフォルト項目を追加
                if (!gpuDetected || GpuSelectionComboBox.Items.Count == 0)
                {
                    GpuSelectionComboBox.Items.Clear();
                    GpuSelectionComboBox.Items.Add("GPU 0: デフォルト");
                }

                // 現在の設定を反映、または最初のGPUを選択
                int targetIndex = Math.Min(Settings.GpuId, GpuSelectionComboBox.Items.Count - 1);
                targetIndex = Math.Max(0, targetIndex); // 最低でもインデックス0を選択
                
                // UIスレッドで確実に選択を実行
                if (GpuSelectionComboBox.Dispatcher.CheckAccess())
                {
                    GpuSelectionComboBox.SelectedIndex = targetIndex;
                }
                else
                {
                    GpuSelectionComboBox.Dispatcher.Invoke(() => 
                    {
                        GpuSelectionComboBox.SelectedIndex = targetIndex;
                    });
                }
                
                System.Diagnostics.Debug.WriteLine($"GPU ComboBox loaded with {GpuSelectionComboBox.Items.Count} items, selected index: {targetIndex}");
            }
            catch (Exception ex)
            {
                // エラーが発生した場合はデフォルト項目を追加
                GpuSelectionComboBox.Items.Clear();
                GpuSelectionComboBox.Items.Add("GPU 0: デフォルト");
                
                // UIスレッドで確実に選択を実行
                if (GpuSelectionComboBox.Dispatcher.CheckAccess())
                {
                    GpuSelectionComboBox.SelectedIndex = 0;
                }
                else
                {
                    GpuSelectionComboBox.Dispatcher.Invoke(() => 
                    {
                        GpuSelectionComboBox.SelectedIndex = 0;
                    });
                }
                
                System.Diagnostics.Debug.WriteLine($"GPU list loading failed: {ex.Message}");
            }
            finally
            {
                _isLoadingGpu = false;
                System.Diagnostics.Debug.WriteLine("=== LoadGpuList() completed ===");
            }
        }

        private async void RefreshGpuListButton_Click(object sender, RoutedEventArgs e)
        {
            // GPUリストを再初期化
            await GpuManager.InitializeAsync();
            // キャッシュから再読み込み
            LoadGpuListFromCache();
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
        public int GpuId { get; set; } = 0; // デフォルトはGPU 0
        
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
                DetailedDescription = this.DetailedDescription,
                GpuId = this.GpuId
            };
        }
    }
}