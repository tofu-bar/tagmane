using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;

namespace tagmane
{
    public partial class ReplaceTagWindow : Window
    {
        public string SourceTag => SourceTagComboBox.Text;
        public string DestinationTag => DestinationTagTextBox.Text;
        public bool UseRegex => UseRegexCheckBox.IsChecked ?? false;
        public bool UsePartialMatch => UsePartialMatchCheckBox.IsChecked ?? false;
        public bool ApplyToAll => ApplyToAllCheckBox.IsChecked ?? false;

        public ReplaceTagWindow(List<string> allTags)
        {
            InitializeComponent();
            SourceTagComboBox.ItemsSource = allTags;
        }

        private void ExecuteButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(SourceTag) || string.IsNullOrWhiteSpace(DestinationTag))
            {
                MessageBox.Show("置換元タグと置換先タグを入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            DialogResult = true;
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }

        private void InsertRegexPattern(object sender, RoutedEventArgs e)
        {
            var button = sender as Button;
            string pattern = "";

            switch (button.Name)
            {
                case "InsertAnyCharButton":
                    pattern = ".+";
                    break;
                case "InsertDigitButton":
                    pattern = "\\d+";
                    break;
                case "InsertWordCharButton":
                    pattern = "\\w+";
                    break;
            }

            TextBox textBox = SourceTagComboBox.Template.FindName("PART_EditableTextBox", SourceTagComboBox) as TextBox;
            if (textBox != null)
            {
                int selectionStart = textBox.SelectionStart;
                SourceTagComboBox.Text = SourceTagComboBox.Text.Insert(selectionStart, pattern);
                textBox.SelectionStart = selectionStart + pattern.Length;
                textBox.Focus();
            }
        }
    }
}
