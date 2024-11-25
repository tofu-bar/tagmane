using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Input;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace tagmane.Features.TagShuffle
{
    public partial class TagShuffleWindow : Window
    {
        private readonly Func<int, int, string, string, bool, Task> _shuffleAction;
        private readonly List<string> _tagCategories;

        public TagShuffleWindow(Func<int, int, string, string, bool, Task> shuffleAction, List<string> tagCategories)
        {
            InitializeComponent();
            _shuffleAction = shuffleAction;
            _tagCategories = tagCategories;

            // カテゴリコンボボックスの設定
            StartCategoryComboBox.ItemsSource = _tagCategories;
            EndCategoryComboBox.ItemsSource = _tagCategories;
        }

        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            var regex = new Regex("[^0-9]+");
            e.Handled = regex.IsMatch(e.Text);
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private async void ExecuteButton_Click(object sender, RoutedEventArgs e)
        {
            int? startPos = string.IsNullOrWhiteSpace(StartPositionTextBox.Text) ? 
                null : int.Parse(StartPositionTextBox.Text);
            int? endPos = string.IsNullOrWhiteSpace(EndPositionTextBox.Text) ? 
                null : int.Parse(EndPositionTextBox.Text);

            string startCategory = UseStartCategoryCheckBox.IsChecked == true ? 
                StartCategoryComboBox.SelectedItem as string : null;
            string endCategory = UseEndCategoryCheckBox.IsChecked == true ? 
                EndCategoryComboBox.SelectedItem as string : null;

            bool applyToAll = ApplyToAllCheckBox.IsChecked == true;

            await _shuffleAction(
                startPos ?? 0,
                endPos ?? int.MaxValue,
                startCategory,
                endCategory,
                applyToAll
            );

            DialogResult = true;
            Close();
        }
    }
} 