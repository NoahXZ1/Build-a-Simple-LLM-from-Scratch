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

#5.1.2 Calculating the text generation loss
inputs = torch.tensor([[16833, 3626, 6100],   #'every effort moves,
                      [40, 1107, 588]])       #'I really like'

targets = torch.tensor([[3626, 6100, 345],       #'effort moves you
                        [1107, 588, 11311]])     #'Really like chcoolate

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)   #probability of each token in vocabulary
#step 3/4 find the index position with highest prob, and the corresponding token ID
token_ids = torch.argmax(probas, dim=-1, keepdim = True)
print("Token IDs:\n", token_ids)
#step 5: converts token IDs back to text:
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]  #this is not 0, just because the number is too small (10^-5)
print("Text 2:", target_probas_2)
#logarithm the probs to avodi underflow when multiplying small numbers together
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
#compute average
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
#show the shape of logits and target tensors
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
#flatten these tensors by combining them over the batch dimension
logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)