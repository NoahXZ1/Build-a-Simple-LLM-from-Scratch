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

#implement a placeholder GPT model architecture class
import re

import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #use a placeholder for transformer blocks
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        #use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

#a simple placeholder that will be replaced by a real TransformerBlock later
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

#a placeholder for LayerNorm
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
#tokenizer a batch consisting of two text inputs for the GPT model using the tiktoken tokenizer from chp2
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
#then we initialize a new 124M parameter DummyGPTModel instance and feed it the tokenized batch:
torch.manual_seed(123)
model=DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
#the output shape is (2,6,50257) which is (patch_size, seq_len, vocab_size)
print(logits) 
"""-----------------------------4.2 Normalizing activations with Layer Normalization---------------------------------"""
#implement a simpel nerual network layer with 5 inputs and 6 outputs that we apply to 2 inputs examples:
torch.manual_seed(123)
batch_example = torch.randn(2,5) #create 2 examples with 5 dims each
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)
#verify the mean and variance before applying LayerNorm
#dim=-1 or 1 indicates calculate the mean and variance across the column dimension, dim=0 is for calculating acroos the row dimension
mean = out.mean(dim=-1, keepdim=True) #use keepdim to maintain the same shape for mean and variance with input
var = out.var(dim=-1, keepdim=True)  #the output without keepdim=True would be a 2D vector like [0.1324, 0.2170] instead of 2*1 matrix like [[0.1324], [0.2170]]
print("Mean:\n", mean)
print("Variance:\n", var)
#applying layer normalization to the layer outputs, including calculating SD
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var=out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
#the mean should be close to 0 and variance should be close to 1 after normalization
print(torch.allclose(mean, torch.zeros_like(mean), atol=1e-7))
#turn off the scientific natation for better readability
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)
# a layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean=x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #eps is a small constant to prevent division by zero
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
#try the LayerNorm module and apply it to the batch input
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased = False, keepdim=True)
print("Means\n", mean)
print("Variance:\n", var)
"""-----------------------------4.3 Implementing a feed forward network with GELU activation---------------------------------"""
# a simple class 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #GELU activation function formula
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))
                                   ))
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()
# creates 100 sample data points in the range -3 to 3
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))
#draw the GELU and ReLU activation functions using subplots
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1,2,i)
    plt.plot(x, y)
    plt.title(label)
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
# plt.show()
#a simple class of feed forward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #this is a small neural network with 2 linear layers and a GELU activation 
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
# initialize a feed forward  module with a token embedding dimension of 768 and feed it a batch input with two samples and three tokens each:
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2,3,768) # create a batch input with batch dimensions 2
out = ffn(x)
print(out.shape)
"""----------------------------------4.4 Implementing a neural network to illustrate shortcut connections--------------------------------"""
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        #implement 5 linear layers with GELU
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),              
        ])

    def forward(self, x):
        for layer in self.layers:
            #compute the output of the current layer
            layer_output = layer(x)
            #check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x+layer_output
            else:
                x = layer_output
        return x
#initialize a neural network with 5 layers
layer_sizes=[3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])
#specifies random seed for the initial weights for reproducibility
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
#implement a function that computes the gradients in the model's backward pass
def print_gradients(model, x):
    output=model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    #calculate loss based on how close the target and output are 
    loss = loss(output, target)
    #backward pass to calculate gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
#this will print a very small mean gradient
print_gradients(model_without_shortcut, sample_input)
#with skip connection
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
"""------------------------------4.5 Implementing a transformer block--------------------------------"""
# implement a transformer block component of GPT
from ch3_self_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att=MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        #add the original input back
        x = x+shortcut
        
        #shortcut connection for feed forward block
        shortcut = x
        x =self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        #add original input back
        x = x+shortcut
        return x
#initialize a transformer block using configuration dictionary GPT_CONFIG_124M
torch.manual_seed(123)
# a sample input(batch_size, seq length, embedding dims)
x = torch.rand(2,4,768)
block=TransformerBlock(GPT_CONFIG_124M)
output=block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

"""--------------------------4.6 Coding the GPT Model---------------------------------"""
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        #the device setting will allow us train the model on a CPU or GPU
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out=model(batch)
print("Output shape:", out.shape)
print(out)
#collect the total number of parameters in the model's parameter tensors
total_params = sum(p.numel() for p in model.parameters())
#the print is 163million which is different from 124million, this is because "weight tying", which is used in the original GPT-2 model
print(f"Total parameters in the model: {total_params}")

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)    
#remove output layer parameter count from the total GPT2 model count according to the weight tying
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
    f"considering weight typing: {total_params_gpt2}")

total_size_bytes = total_params * 4 #calculate the total size in bytes (assuming float32, 4 bytes per parameter)
#convert to MB
total_size_mb = total_size_bytes / (1024*1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

"""--------------------------Exercise4.1 : FFN vs MHA parameter counts---------------------------------"""
def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# Count parameters in one transformer block
first_block = model.trf_blocks[0]
ff_params_one_block = count_trainable_params(first_block.ff)
mha_params_one_block = count_trainable_params(first_block.att)

print("One block - FeedForward params:", ff_params_one_block)
print("One block - MultiHeadAttention params:", mha_params_one_block)
print("FF/MHA ratio (one block):", ff_params_one_block / mha_params_one_block)

# Count parameters across all transformer blocks
ff_params_all_blocks = sum(count_trainable_params(block.ff) for block in model.trf_blocks)
mha_params_all_blocks = sum(count_trainable_params(block.att) for block in model.trf_blocks)

print("All blocks - FeedForward params:", ff_params_all_blocks)
print("All blocks - MultiHeadAttention params:", mha_params_all_blocks)
print("FF/MHA ratio (all blocks):", ff_params_all_blocks / mha_params_all_blocks)

"""--------------------------Exercise4.2 : Initialize GPT-2 Large and count parameters---------------------------------"""
GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,  # embedding dimension for GPT-2 Large
    "n_heads": 20,     # number of attention heads for GPT-2 Large
    "n_layers": 36,    # number of transformer blocks for GPT-2 Large
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model_large = GPTModel(GPT_CONFIG_774M)
# calculate total parameters in the GPT-2 large model
total_params_large = sum(p.numel() for p in model_large.parameters())
total_params_large_weight_tied = (
    total_params_large - sum(p.numel() for p in model_large.out_head.parameters())
)

print("\nGPT-2 Large config:", GPT_CONFIG_774M)
print(f"Total parameters in GPT-2 Large: {total_params_large}")
print(
    "Total parameters in GPT-2 Large (considering weight tying): "
    f"{total_params_large_weight_tied}"
)
#calculate the total size of the GPT-2 large model in MB
total_size_bytes = total_params_large * 4 #calculate the total size in bytes (assuming float32, 4 bytes per parameter)
#convert to MB
total_size_mb = total_size_bytes / (1024*1024)
print(f"Total size of the GPT-2 large model: {total_size_mb:.2f} MB")

"""--------------------------4.7 Generating text-------------------------------"""
#a function for the GPT model to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        #crops current conext if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        #focus only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits =logits[:, -1, :]
        #probas has shape (batch, vocab_size)
        #actually softmax is not necessary here, as the argmax can be applied directly to the logits. (here we add it for demonstration)
        probas = torch.softmax(logits, dim=-1)
        #idx_next has shape(batch, 1)
        #argmax is used to select the token with the highest probability as the next token in the sequence(it returns the indices rather than the values)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        #appends sampled index to the running sequence, where idx has shape(batch,n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
# try our function with "Hello, I am" context as the input
#we first encode the input context into token IDs
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
#adds batch dimension
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
#disable dropout since we are not training the model
model.eval()
out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
print("Output:", out)
print("Output length:", len(out[0]))
#use the .decode method of the tokenizer to convert the IDs back into text
decoded_output = tokenizer.decode(out.squeeze().tolist())
print(decoded_output)

"""--------------------------Exercise4.3: Separate dropout rates for each layer---------------------------------"""
# The original GPT_CONFIG_124M uses a single "drop_rate" applied uniformly to all three
# dropout locations: the embedding layer, the multi-head attention module, and the
# shortcut (residual) connections. This exercise splits that into three independent values.

GPT_CONFIG_124M_EX43 = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,       # dropout after embedding layer
    "drop_rate_attn": 0.1,      # dropout inside MultiHeadAttention
    "drop_rate_shortcut": 0.1,  # dropout on shortcut (residual) connections
    "qkv_bias": False,
}

class TransformerBlockEx43(nn.Module):
    """TransformerBlock that reads three separate dropout rates from cfg."""
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"],   # <-- attention dropout
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])  # <-- shortcut dropout

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModelEx43(nn.Module):
    """GPTModel that reads three separate dropout rates from cfg."""
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])  # <-- embedding dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlockEx43(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

torch.manual_seed(123)
model_ex43 = GPTModelEx43(GPT_CONFIG_124M_EX43)
out_ex43 = model_ex43(batch)
print("Exercise 4.3 - Output shape:", out_ex43.shape)  # expect (2, 6, 50257)