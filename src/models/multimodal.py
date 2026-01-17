"""
Multimodal Model Architecture (Text + Image).
"""

import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalFusionNet(nn.Module):
    """
    Fuses Text (XLM-R) and Image (ResNet) features.
    """
    
    def __init__(self, 
                 text_model, 
                 image_model_name='resnet50', 
                 fusion_hidden_dim=512, 
                 num_classes=2):
        super(MultimodalFusionNet, self).__init__()
        
        # Text Encoder (Pre-initialized classifier or just the transformer)
        # We assume text_model returns embeddings (768)
        self.text_encoder = text_model
        self.text_dim = 768 # Default for base models
        
        # Image Encoder
        # Using ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove the classification head (fc)
        modules = list(resnet.children())[:-1] 
        self.image_encoder = nn.Sequential(*modules)
        self.image_dim = 2048 # ResNet50 output
        
        # Fusion Layer
        # Project both to common dimension
        self.text_proj = nn.Linear(self.text_dim, fusion_hidden_dim)
        self.image_proj = nn.Linear(self.image_dim, fusion_hidden_dim)
        
        # Attention mechanism for fusion
        self.attention = nn.MultiheadAttention(embed_dim=fusion_hidden_dim, num_heads=4)
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim), # Concatenated
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, image_tensor):
        """
        Args:
            input_ids: text tokens
            attention_mask: text mask
            image_tensor: (Batch, 3, 224, 224)
        """
        # 1. Text Features
        # Get sequence output from transformer
        # We need access to the internal transformer of the text_encoder wrapper
        if hasattr(self.text_encoder, 'transformer'):
            text_outputs = self.text_encoder.transformer(input_ids, attention_mask)
            text_feat = text_outputs.last_hidden_state[:, 0, :] # CLS token (Batch, 768)
        else:
            # Fallback if text_encoder is raw model
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_feat = text_outputs.last_hidden_state[:, 0, :]
            
        # 2. Image Features
        # (Batch, 2048, 1, 1) -> (Batch, 2048)
        img_feat = self.image_encoder(image_tensor).squeeze(-1).squeeze(-1)
        
        # 3. Projection
        text_emb = self.text_proj(text_feat)   # (Batch, Fusion_Dim)
        img_emb = self.image_proj(img_feat)    # (Batch, Fusion_Dim)
        
        # 4. Fusion Strategy
        # Simple Concatenation + Attention-like interaction
        # Reshape for multihead attention: (Seq_Len, Batch, Dim) - Sequence length is 1 here effectively per modality
        
        # Stack them: (2, Batch, Dim)
        combined = torch.stack([text_emb, img_emb], dim=0)
        
        # Self-Attention between modalities
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Flatten: (Batch, Dim * 2)
        # Permute back to (Batch, 2, Dim) then flatten
        fusion_output = attn_output.permute(1, 0, 2).reshape(input_ids.size(0), -1)
        
        logits = self.classifier(fusion_output)
        
        return logits
