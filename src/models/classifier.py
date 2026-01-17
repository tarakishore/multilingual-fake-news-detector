"""
Core Classifier Module using XLM-RoBERTa.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

class FakeNewsClassifier(nn.Module):
    """
    Multilingual Fake News Classifier.
    """
    
    def __init__(self, 
                 model_name='xlm-roberta-base', 
                 num_classes=2, 
                 dropout=0.3,
                 use_hybrid_head=False):
        super(FakeNewsClassifier, self).__init__()
        
        self.use_hybrid_head = use_hybrid_head
        
        # Load Transformer Backbone
        logger = logging.getLogger(__name__)
        logger.info(f"Loading transformer model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        embedding_dim = config.hidden_size
        
        if self.use_hybrid_head:
            # Import here to avoid circular dependencies if any
            from .hybrid import RobustBiLSTM
            self.classifier_head = RobustBiLSTM(
                input_dim=embedding_dim,
                hidden_dim=256,
                num_layers=2,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            # Standard Classification Head
            # We use a slightly more complex head than just a linear layer
            self.classifier_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_classes)
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass.
        Args:
            input_ids: (Batch, Seq_Len)
            attention_mask: (Batch, Seq_Len)
        """
        # Transformer Output
        # output[0] is sequence_output (Batch, Seq_Len, Hidden)
        # output[1] is pooled_output (Batch, Hidden) - mainly for NSP, not always best for classification
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        
        if self.use_hybrid_head:
            # Hybrid head handles the sequence
            logits = self.classifier_head(sequence_output)
        else:
            # We pull the <CLS> token (index 0) or use pooled output provided by model
            # For XLM-R, <s> is the first token.
            cls_token_state = sequence_output[:, 0, :]
            logits = self.classifier_head(cls_token_state)
            
        return logits

    def predict(self, text, tokenizer, device='cpu'):
        """
        Inference helper.
        """
        self.eval()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
        return {
            'label': 'fake' if prediction == 1 else 'real',
            'confidence': confidence,
            'logits': logits,
            'probs': probs
        }
