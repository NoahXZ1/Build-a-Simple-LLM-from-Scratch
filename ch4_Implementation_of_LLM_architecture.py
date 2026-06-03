"""This is for chapter 4, Implementation of LLM architecture."""
# This is the configuration for a small GPT-2 model with 124 mill parameters. 
GPT_CONFIG_124M = {
    "vocab_size": 50257,  #vocabulary size
    "context_length": 1024,  #context_length
    "emb_dim": 768, #Embedding dimension
    "n_heads": 12,  #number of attention heads
    "n_layers": 12,  #number of transformer blocks
    "drop_rate": 0.1,  #Dropout rate
    "qkv_bias": False,  #Query-Key-Value bias
}