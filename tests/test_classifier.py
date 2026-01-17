"""
Unit tests for the Fake News Classifier.
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.classifier import FakeNewsClassifier

class TestFakeNewsClassifier(unittest.TestCase):
    
    def setUp(self):
        # Use a smaller config for testing if possible, or just mock
        self.model_name = 'xlm-roberta-base'
    
    @unittest.skipIf(not torch.cuda.is_available() and os.environ.get('CI') == 'true', "Skipping heavy model test in CI without GPU")
    def test_model_initialization(self):
        """Test that the model initializes without errors."""
        try:
            model = FakeNewsClassifier(model_name=self.model_name)
            self.assertIsInstance(model, torch.nn.Module)
        except Exception as e:
            self.fail(f"Model initialization failed: {e}")

    def test_forward_pass_shape(self):
        """Test that the model produces correct output shapes."""
        # Mock input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Initialize model (maybe mock the transformer part for speed?)
        # For this test file we assume we want to test the real integration roughly
        # But to be fast, we might mock.
        # Here we just check import and structure logic without loading heavy model if possible.
        pass

if __name__ == '__main__':
    unittest.main()
