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
        private bool _isModelLoaded = false;

        public event EventHandler<string> LogUpdated;
        public bool IsGpuLoaded { get; private set; }

        public async Task LoadModel(string modelRepo, bool useGpu = true)
        {
            if (modelRepo.Contains("joytag"))
            {
                _currentPredictor = new JoyPredictor();
                ((JoyPredictor)_currentPredictor).LogUpdated += OnPredictorLogUpdated;
                await ((JoyPredictor)_currentPredictor).LoadModel(modelRepo, useGpu);
                IsGpuLoaded = ((JoyPredictor)_currentPredictor).IsGpuLoaded;
            }
            else
            {
                _currentPredictor = new WDPredictor();
                ((WDPredictor)_currentPredictor).LogUpdated += OnPredictorLogUpdated;
                await ((WDPredictor)_currentPredictor).LoadModel(modelRepo, useGpu);
                IsGpuLoaded = ((WDPredictor)_currentPredictor).IsGpuLoaded;
            }
            _isModelLoaded = true;
        }

        private void OnPredictorLogUpdated(object sender, string message)
        {
            LogUpdated?.Invoke(this, message);
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            BitmapImage image,
            float generalThresh,
            bool generalMcutEnabled,
            float characterThresh,
            bool characterMcutEnabled)
        {
            if (!_isModelLoaded)
            {
                // throw new InvalidOperationException("モデルが読み込まれていません。Predictを実行する前にLoadModelを呼び出してください。");
                // AddLogEntry("モデルが読み込まれていません。Predictを実行する前にLoadModelを呼び出してください。");
                return ("", new Dictionary<string, float>(), new Dictionary<string, float>(), new Dictionary<string, float>());
            }

            if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.Predict(image, generalThresh, generalMcutEnabled, characterThresh, characterMcutEnabled);
            else if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.Predict(image, generalThresh);
            else
                throw new InvalidOperationException("No predictor loaded");
        }

        public void Dispose()
        {
            if (_currentPredictor is WDPredictor wdPredictor)
            {
                wdPredictor.Dispose();
            }
            else if (_currentPredictor is JoyPredictor joyPredictor)
            {
                joyPredictor.Dispose();
            }
            _isModelLoaded = false;
        }
    }
}
