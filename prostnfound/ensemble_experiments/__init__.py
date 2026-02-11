# Ensemble Experiments Module
"""
Framework for combining multiple trained ProstNFound models using ensemble techniques.

Supported methods:
- Model Soup: Weight averaging across models
- Output Ensemble: Output averaging with optional learned weights
- Transformer Aggregator: Cross-model attention for combining outputs
- MoE: Mixture of Experts with learned routing
"""
