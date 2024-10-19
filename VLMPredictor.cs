using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Security.Cryptography;
using System.Drawing;
using System.Drawing.Imaging;

namespace tagmane
{
    public class VLMPredictor
    {
        private object _currentPredictor;

        public event EventHandler<string> LogUpdated
        {
            add
            {
                if (_currentPredictor is WDPredictor wdPredictor)
                    wdPredictor.LogUpdated += value;
                else if (_currentPredictor is JoyPredictor joyPredictor)
                    joyPredictor.LogUpdated += value;
            }
            remove
            {
                if (_currentPredictor is WDPredictor wdPredictor)
                    wdPredictor.LogUpdated -= value;
                else if (_currentPredictor is JoyPredictor joyPredictor)
                    joyPredictor.LogUpdated -= value;
            }
        }

        public async Task LoadModel(string modelRepo)
        {
            if (modelRepo.Contains("joytag"))
            {
                _currentPredictor = new JoyPredictor();
                await ((JoyPredictor)_currentPredictor).LoadModel(modelRepo);
            }
            else
            {
                _currentPredictor = new WDPredictor();
                await ((WDPredictor)_currentPredictor).LoadModel(modelRepo);
            }
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            BitmapImage image,
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.Predict(image, generalThresh, generalMcutEnabled, characterThresh, characterMcutEnabled);
            else if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.Predict(image, generalThresh);
            else
                throw new InvalidOperationException("No predictor loaded");
        }
    }
}
