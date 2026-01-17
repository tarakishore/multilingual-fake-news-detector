# Multilingual Fake News Detection with Explainability

A comprehensive AI-powered system for detecting fake news across multiple languages with transparent, explainable predictions using state-of-the-art multilingual transformers.

## Features

- **Multilingual Support**: Powered by XLM-RoBERTa and MuRIL for 100+ languages
- **Explainable AI**: Token-level attributions using Integrated Gradients and SHAP
- **Multimodal Analysis**: Combined text and image verification
- **Code-Switching Handling**: Supports mixed-language content (Hinglish, Spanglish, etc.)
- **Interactive Web Demo**: Beautiful, modern UI for real-time analysis

## Project Structure

```
multilingual-fake-news-detector/
├── src/
│   ├── data/           # Data preprocessing and dataset utilities
│   ├── models/         # Neural network architectures
│   ├── explainability/ # XAI modules (IG, SHAP)
│   └── utils/          # Metrics and helpers
├── web/                # Interactive web demo
├── examples/           # Sample data
└── tests/              # Unit tests
```

## Installation

```bash
# Clone the repository
cd multilingual-fake-news-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using the Python API

```python
from src.models.classifier import FakeNewsClassifier
from src.explainability.integrated_gradients import IntegratedGradientsExplainer

# Initialize classifier
classifier = FakeNewsClassifier()

# Analyze text
text = "Breaking: Scientists discover miracle cure for all diseases!"
result = classifier.predict(text)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Get explanations
explainer = IntegratedGradientsExplainer(classifier)
attributions = explainer.explain(text)
```

### Using the Web Demo

Open `web/index.html` in your browser for an interactive demo.

## Supported Languages

The system leverages XLM-RoBERTa which supports 100 languages including:
- **High-Resource**: English, Spanish, French, German, Chinese, Arabic
- **Indic Languages**: Hindi, Bengali, Tamil, Telugu, Marathi (via MuRIL)
- **Low-Resource**: Swahili, Vietnamese, Indonesian, and more

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Text/Image                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   XLM-RoBERTa       │       │   ResNet (Image)    │
│   Text Encoder      │       │   Encoder           │
└─────────┬───────────┘       └─────────┬───────────┘
          │                             │
          └───────────┬─────────────────┘
                      ▼
          ┌─────────────────────┐
          │   Multi-Head        │
          │   Attention Fusion  │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │   CNN-BiLSTM        │
          │   Hybrid Head       │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │   Classification    │
          │   + Explainability  │
          └─────────────────────┘
```

## Explainability

The system provides transparent explanations using:

1. **Integrated Gradients**: Token-level attribution scores showing which words contributed to the prediction
2. **SHAP Values**: Feature importance visualization with force plots
3. **Attention Visualization**: Self-attention heatmaps from the transformer

## Datasets

The system is designed to work with:
- **MM-COVID**: Multilingual COVID misinformation dataset
- **MMIFND**: Multimodal Multilingual Indian Fake News Dataset  
- **FakeCovid**: 40+ language fact-checked articles
- **Constraint@AAAI21**: English and Hindi datasets

## Citation

```bibtex
@article{multilingual-fake-news-detector,
  title={Multilingual Fake News Detection with Explainability},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
