"""
Multilingual Fake News Detection with Explainability
Configuration settings for the project.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"

# Model configurations
class ModelConfig:
    """Configuration for the fake news classifier."""
    
    # Transformer backbone
    MODEL_NAME = "xlm-roberta-base"  # Options: xlm-roberta-base, xlm-roberta-large, google/muril-base-cased
    MAX_LENGTH = 512
    
    # Classification head
    NUM_CLASSES = 2  # Binary: Real/Fake
    DROPOUT_RATE = 0.3
    
    # Hybrid CNN-BiLSTM head
    USE_HYBRID_HEAD = False
    CNN_FILTERS = 256
    CNN_KERNEL_SIZES = [2, 3, 4, 5]
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True


class MultimodalConfig:
    """Configuration for multimodal (text + image) analysis."""
    
    # Image encoder
    IMAGE_MODEL = "resnet50"  # Options: resnet50, resnet101, vit-base
    IMAGE_SIZE = (224, 224)
    
    # Fusion
    FUSION_TYPE = "attention"  # Options: concat, attention, cross-attention
    ATTENTION_HEADS = 8
    FUSION_HIDDEN_SIZE = 512


class TrainingConfig:
    """Training hyperparameters."""
    
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    
    # Optimizer
    OPTIMIZER = "adamw"
    SCHEDULER = "linear_warmup_decay"
    
    # Early stopping
    PATIENCE = 3
    MIN_DELTA = 0.001


class ExplainabilityConfig:
    """Configuration for XAI methods."""
    
    # Integrated Gradients
    IG_STEPS = 50  # Number of interpolation steps
    IG_BASELINE = "zero"  # Options: zero, pad, uniform
    
    # SHAP
    SHAP_MAX_EVALS = 500
    SHAP_BATCH_SIZE = 50
    
    # Visualization
    TOP_K_TOKENS = 10  # Number of top tokens to highlight
    COLORMAP = "RdYlGn_r"  # Red for important, green for unimportant


class PreprocessingConfig:
    """Text preprocessing settings."""
    
    # Cleaning
    REMOVE_URLS = True
    REMOVE_MENTIONS = True
    REMOVE_HASHTAGS = False  # Keep hashtags as they can be informative
    REMOVE_EMOJIS = False
    LOWERCASE = False  # XLM-R is case-sensitive
    
    # Code-switching
    TRANSLITERATE = True  # Convert Romanized text to native script
    SUPPORTED_SCRIPTS = ["Devanagari", "Bengali", "Tamil", "Telugu", "Arabic"]
    
    # Normalization
    NORMALIZE_UNICODE = True
    MAX_CONSECUTIVE_CHARS = 3  # Reduce "sooooo" to "sooo"


# Language mappings
LANGUAGE_CODES = {
    "en": "English",
    "hi": "Hindi", 
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh": "Chinese",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "id": "Indonesian",
}

# Label mappings
LABEL_TO_ID = {
    "real": 0,
    "fake": 1,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Fine-grained labels (for datasets like FakeCovid)
FINE_LABELS = {
    "true": 0,
    "mostly_true": 1,
    "half_true": 2,
    "mostly_false": 3,
    "false": 4,
    "pants_on_fire": 5,
}
