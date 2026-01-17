"""
Explainability module using Integrated Gradients (Captum).
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import logging

class IntegratedGradientsExplainer:
    """
    Wrapper for Captum's Integrated Gradients to explain model predictions.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(self._forward_func)
        self.device = next(model.parameters()).device

    def _forward_func(self, inputs_embeds, attention_mask=None):
        """
        Custom forward function for IG. 
        IG needs gradients W.R.T embeddings, so we bypass the embedding layer.
        """
        # XLM-R specific: The embeddings are usually model.embeddings(input_ids)
        # But we are passing embeddings directly here.
        # We need to hack the model to accept embeddings or use a wrapper.
        # The FakeNewsClassifier forward takes input_ids. 
        # We need a way to pass embeddings to the transformer backbone.
        
        # Most HF models support `inputs_embeds` in forward().
        # We need to verify FakeNewsClassifier supports it.
        # Let's assume we modify FakeNewsClassifier or use the internal transformer directly.
        
        # For simplicity in this demo wrapper, we assume `model.transformer` accepts `inputs_embeds`
        outputs = self.model.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Determine classification head interaction
        if self.model.use_hybrid_head:
            logits = self.model.classifier_head(sequence_output)
        else:
            cls_token_state = sequence_output[:, 0, :]
            logits = self.model.classifier_head(cls_token_state)
            
        return logits

    def explain(self, text, target_class=None):
        """
        Generate attributions for the input text.
        """
        self.model.eval()
        
        # 1. Tokenize
        encoded = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 2. Get embeddings
        # We need the embedding layer to compute baselines and input embeddings
        embeddings = self.model.transformer.embeddings(input_ids)
        
        # 3. Compute Attributions
        # Baseline: Zero embedding or Pad embedding
        baseline = torch.zeros_like(embeddings)
        
        # Predict class if not provided
        if target_class is None:
            with torch.no_grad():
                logits = self._forward_func(embeddings, attention_mask)
                target_class = torch.argmax(logits, dim=1).item()
        
        attributions, delta = self.ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            target=target_class,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=50
        )
        
        # 4. Summarize Attributions
        # Attributions are (Batch, Seq, Hidden). sum over Hidden dim to get token importance.
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        
        # Map back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'tokens': tokens,
            'attributions': attributions,
            'delta': delta.item(),
            'target_class': target_class
        }

    def visualize(self, explanation_result):
        """
        Return an HTML visualization object (string).
        """
        return viz.format_word_importances(
            words=explanation_result['tokens'],
            importances=explanation_result['attributions']
        )
