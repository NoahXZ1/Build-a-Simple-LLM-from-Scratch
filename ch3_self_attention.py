#This is code implementation of the chapter 3 in book "Build a Large Language Model from Scratch" 
#begins on 05/29/2026
#Then comes light

import torch
#a small simplified self-attention implementation
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your      (x^1)
     [0.55, 0.87, 0.66], # journey   (x^2)
     [0.57, 0.85, 0.64], # starts    (x^3)
     [0.22, 0.58, 0.33], # with      (x^4)
     [0.77, 0.25, 0.10], # one       (x^5)
     [0.05, 0.80, 0.55]] # step      (x^6)
) 
#computing the dot product of the query, x^2, with every other input token
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
#normalization 
attn_weights_2_tmp=attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
#softmax function for normalizing the attention scores
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
#PyTorch's built-in implementation of the softmax function, naive softmax implementation is numerically unstable
attn_weights_2= torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
#computing the weighted sum of the value vectors, context vector z^(2)
query = inputs[1] #the second input token is the query
context_vec_2=torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
#implementing all context vectors 
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)
print(attn_scores)
#using matrix multiplication(faster than for loop)
attn_scores = inputs @ inputs.T
print(attn_scores)
# normalize 
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
#verify the rows indeed all sum to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
#compute all context vectors using matrix multiplication
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
"""-----------------------------------------------------------------3.4 Implement self-attention with multiple heads------------------------------------------------------------------"""
x_2=inputs[1] #the second input elemnt
d_in = inputs.shape[1] #the input embedding size, d=3
d_out = 2 # the output embedding size, d_out = 2
#initialize the 3 matrices Wq, Wk, Wv
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#compute the query, key, and value vectors:
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
# compute the key and value vectors for all input tokens by matrix multiplication
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
#compute the attention scores(by dot product of query(q) with each key(k) of all input tokens)
#first we compute the attention score of query token
keys_2 = keys[1]#this is the key vector of the query token itself
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
#then generalize the computation to all attention scores by matrix multiplication
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
#compute the attentions weights by applying the softmax function to the attention scores
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim = -1)
print(attn_weights_2)
#compute the context vector by taking the weighted sum of the value vectors(multiply each value vector(v) with its respective attention weight and then summing them up))
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
#implement a compact self-attention Python class

"SelfAttention is a class derived from nn.Module, which is a fundemental building block of PyTorch models, "
"providing necassary functionality for model layer creation and management. "
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    #initialize three trainable weight matrices, W_query, W_key and W_value. 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
#a simple test of the class
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
#The second row matches the content of context_vec_2 before
print(sa_v1(inputs))

#a self_attention class using PyTorch's Linear layers
# nn.Linear has an optimized weight initialization scheme, more stable and efficient for model training
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim = -1
        )
        context_vec = attn_weights @ values
        return context_vec
#using SelfAttention_v2 class similar to v1
#v2 has different output due to a different initial weights for the weight matrices since nn.Linear uses a more sophisticated weight initialization scheme. 
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
#to make the output of sa_v2 same as sa_v1, as nn.Linear initializes its weights differently, we can directly copy the weights from sa_v1 to sa_v2, but need to transpose them since nn.Linear expects weights in the shape of (out features, in features)
# while in sa_v1 we have the shape of(in features, out features)
sa_v2.W_query.weight.data = sa_v1.W_query.data.T
sa_v2.W_key.weight.data = sa_v1.W_key.data.T
sa_v2.W_value.weight.data = sa_v1.W_value.data.T
print(sa_v2(inputs))
# to verify that the output of two implementations are the same. 
print(torch.allclose(sa_v1(inputs), sa_v2(inputs)))
