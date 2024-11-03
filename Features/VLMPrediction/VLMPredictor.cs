using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace tagmane
{
    public class VLMPredictor
    {
        private object? _currentPredictor;
        private bool _isModelLoaded = false;

        public event EventHandler<string>? LogUpdated;
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

        private void OnPredictorLogUpdated(object? sender, string message)
        {
            LogUpdated?.Invoke(this, message);
        }

        public DenseTensor<float>? PrepareTensor(BitmapImage image)
        {
            // safe-guard: モデルが読み込まれていない場合、早めに失敗
            if (!_isModelLoaded) throw new InvalidOperationException("モデルが読み込まれていません。Predictを実行する前にLoadModelを呼び出してください。");

            if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.PrepareTensor(image);
            else if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.PrepareTensor(image);
            else
                throw new InvalidOperationException("No predictor loaded");
        }

        public (string, Dictionary<string, float>, Dictionary<string, float>, Dictionary<string, float>) Predict(
            DenseTensor<float> tensor, 
            float generalThresh=0.35f,
            bool generalMcutEnabled=false,
            float characterThresh=0.85f,
            bool characterMcutEnabled=false)
        {
            if (!_isModelLoaded) throw new InvalidOperationException("モデルが読み込まれていません。");

            if (_currentPredictor is WDPredictor wdPredictor)
                return wdPredictor.Predict(tensor, generalThresh, generalMcutEnabled, characterThresh, characterMcutEnabled);
            else if (_currentPredictor is JoyPredictor joyPredictor)
                return joyPredictor.Predict(tensor, generalThresh);
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
