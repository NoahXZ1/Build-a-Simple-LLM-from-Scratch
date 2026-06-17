"""This is for chapter 5, "Pretraining on Unlabeled Data"""
#5.1.1 using GPT to generate text(recall of chp4)
import torch
from ch4_Implementation_of_LLM_architecture import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #we shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,   #its possible and common to set dropout to 0. 
    "qkv_bias": False
}
torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)
model.eval()
#utility functions for text to token ID conversion
import tiktoken
from ch4_Implementation_of_LLM_architecture import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  #.unsqueeze(0) adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx= text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size = GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))