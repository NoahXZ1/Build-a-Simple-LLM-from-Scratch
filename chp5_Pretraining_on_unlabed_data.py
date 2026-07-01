"""This is for chapter 5, "Pretraining on Unlabeled Data"""
"""-----------------------------------5.1.1 using GPT to generate text(recall of chp4)------------------------------------"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PREFERRED_PYTHON = PROJECT_DIR / ".venv311" / "Scripts" / "python.exe"
FALLBACK_PYTHON311 = Path(r"D:\Program Files\python311\python.exe")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

def rerun_with_python(python_exe):
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        [str(python_exe), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(PROJECT_DIR),
        env=env,
    )
    raise SystemExit(result.returncode)


if PREFERRED_PYTHON.exists() and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve():
    rerun_with_python(PREFERRED_PYTHON)
elif sys.version_info[:2] != (3, 11) and FALLBACK_PYTHON311.exists():
    rerun_with_python(FALLBACK_PYTHON311)

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Please run this script with Python 3.11.")

os.chdir(PROJECT_DIR)
MPL_CONFIG_DIR = PROJECT_DIR / "matplotlib_cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

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
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
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

"""---------------------------------------------5.1.2 Calculating the text generation loss ---------------------------------------------"""
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

"""---------------------------------------------5.1.3 calculating loss for training and validation sets---------------------------------------------"""
#load the "The Verdict" story dataset
file_path = "the-verdict.txt"
with open(file_path, "r", encoding = "utf-8") as file:
    text_data = file.read()
#check the dataset
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)
#define the training and validation dataset
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from ch2_data_processing import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True,
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = False,
    shuffle = False,
    num_workers = 0
)

#iterate through the training dataloader ensuring the data was created correctly
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
#this is a utility function to calculate the loss for a batch of input and target data, given a model and device (GPU)
def calc_loss_batch(input_batch, target_batch, model, device):
    # transfer the input and target batches to the specified device (GPU)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss
#the function to calculate the loss over all the batches
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        #iterate through all batches if no fixed num_batches is give
        num_batches = len(data_loader)
    else:
        # a checker to ensure num_batches does not exceed the total number of batches in the data_loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            #sums loss for each batch
            total_loss += loss.item()
        else:
            break
    # return the average loss over all batches
    return total_loss / num_batches
#the following is a sample of use smaller number of batches to speed up the evaluation process during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():  # disable gradient calculation as we haven't trained the model yet
    #the "device" setting here can ensure we load the data and LLM onto the same device, which can speed up the evalution process
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
"""-----------------------------------------5.2 Training an LLM----------------------------------------------"""
#the main function for pretraining the llm
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    #Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0,-1
    #the main training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            #this is to reset the loss gradients from previous batch iteration.
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            #calculate loss gradients 
            loss.backward()
            #update model weights using loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            #the global step is used to track the number of batches processed, whichn is useful for scheduling evaluations and logging
            global_step +=1
            # these are optinal evaluation steps to monitor training progress and adjust hyperparameters when necessary
            # eval_freq determines how often to evaluate the model while eval_iter determines how many batches to use for evaluation 
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep{epoch+1} (Step{global_step:06d}): "
                      f"Train loss {train_loss:.3f},"
                      f"Val_loss {val_loss:.3f}"
                )
        #print a sample text after each epoch to monitor the model's text generation capability
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
#the evaluate_model
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # dropout is disabled during evaluation for stable, reproducible results
    model.eval()
    #disable gradient tracking, not required in evalution
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches = eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches = eval_iter
        )
        model.train()
        return train_loss, val_loss
#this function takes a text snippet(start_context) as input, converts it into token IDs, and feeds it to the LLM to generate a text sample using the "generate_text_simple" function in chp4.
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded=text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, 
            max_new_tokens=50, context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids.cpu(), tokenizer)
    #compact print format
    print(decoded_text.replace("\n", " "))
    model.train()
# training a GPTModel instance for 10 epochs using an AdamW optimizer and the train_model_simple function we deifined eailer
torch.manual_seed(1234)
model= GPTModel(GPT_CONFIG_124M)
model.to(device)
#the following is used to control whether to run the small model training section, which can cost much time to do.
RUN_SMALL_MODEL_TRAINING = True

# Toggle this on only when you need to run the slow training section.
if RUN_SMALL_MODEL_TRAINING:
    #lr is the learning rate paramter, controls step size at each iteration.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #the parameters() returns all trainable parameters of model
    num_epochs = 10
    train_losses, val_losses, track_tokens_seen=train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs = num_epochs, eval_freq=5, eval_iter=5,
        start_context = "Every effort moves you", tokenizer = tokenizer)
else:
    print("Skip small-model training section (set RUN_SMALL_MODEL_TRAINING=True to enable).")
#a plot of training and validation losses over time, helping us understand the model's learning progress and overfitting issue
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label = "Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny() # create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible plot to set the scale of the second x-axis
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
#the printed plot shows that overfitting has occurred, as training loss continues decrease while val loss stays around 6.5 after 2 epochs.
# however overfitting is not a concern in this case as the val loss didn't increase significantly at the last few epochs (because training may still be in an early stage and the current validation estimate can be noisy with limited epochs/eval steps).
#only draw the plot when small model is trained.
if RUN_SMALL_MODEL_TRAINING:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)
"""-----------------------------------------5.3 Decoding Stratergies to Control Randomness--------------------------------------"""
#we put the model into evaluation mode to turn off random components such as dropout
model.to("cpu")
model.eval()
# we plug the GPTModel instance (model) into the generate_text_simple function, which uses the LLM to generate one token at a time. 
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
#implement temperature scaling
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab= {v: k for k, v in vocab.items()}
#assume the LLM generates the following next-token logits:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
#softmax() to convert logits into probabilities 
#argmax() is used to obtain the index of the token with the highest prob, which is then mapped back to the corresponding token using the inverse_vocab dictionary
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])
#we replace argmax() with multinomial(), which samples from the probability distribution defined by probas
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples = 1).item()
print(inverse_vocab[next_token_id])
# the output will still be "forward" as it is still the most probable token.
#If we repeat the sampling by 1000 times:
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample =  [torch.multinomial(probas,num_samples=1).item()
             for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
# implement temperature scaling by dividing the logits by a temperature parameter before softmax
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits/temperature
    return torch.softmax(scaled_logits, dim=0)
#the following example shows the effect of different temperature
temperatures = [1,0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]
#Exe5.1: shows the sampled tokens for each temperature setting
for j in range(len(temperatures)):
    print_sampled_tokens(scaled_probas[j])
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize = (5,3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x+i*bar_width, scaled_probas[i], bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
"""------------------------------------------5.3.2 Top-k sampling--------------------------------------"""
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top-k logits:", top_logits)
print("Top positions:", top_pos)
#apply PyTorch's where function to set the logits values of tokens that are below the lowest logit value
new_logits = torch.where(
    condition = next_token_logits < top_logits[-1],
    #assign -inf to these lower logits
    input = torch.tensor(float('-inf')),
    other = next_token_logits
)
print(new_logits)
#apply the softmax function to turn these into next-token probs
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
"""-------------------------------------------5.3.3 final modified text generation function with temp scaling and top-k---------------------------------------------"""
#the new generate text function
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id = None):
    #the for loop is same as the previous one, just focus on the last token
    for _ in range(max_new_tokens):
        #this is to ensure the input tensor is not longer than the context size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # focus on the last token
        #filters logits with top_k sampling
        if top_k is not None:
            top_logits, _= torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )
        #apply temperature scaling
        if temperature > 0.0:
            logits = logits/temperature 
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        #use the previous greedy sampling method if temp scaling is not applied
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        #stop generating if the end-of-text token is generated
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
#calling the function in a sample text generation task
torch.manual_seed(123)
#turn the top-k smaller will make the output more focused ahd deterministic
#when setting top_k to 1, it is equivalent to greedy sampling actually. 
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size = GPT_CONFIG_124M["context_length"],
    top_k = 5,
    temperature=1.1
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
"""------------------------------5.4 Loading and saving model weights in PyTorch---------------------------------------"""
# use state_dict(), a dictionary object thta maps each layer to its parameter tensor, by torch.save()
#'model.pth' is the file name to save the model weights. The .pth is a convention for PyTorch model files
torch.save(model.state_dict(), "model.pth")
#load the weights into a new GPTModel instace
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
#save both the model and optimizer state_dict contents:
if RUN_SMALL_MODEL_TRAINING:
    torch.save({
        "model state dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    )
#restore the model and optimizer states by first loading the saved data by torch.load and load_state_dict
checkpoint=torch.load("model_and_optimizer.pth", map_location=device)
model=GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model state dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
"""---------------------------------5.5 Loading pretrained weights from OpenAI------------------------------------"""
import sys
print(sys.executable)
#download gpt_download.py from the LLM book's repository
import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = PROJECT_DIR / url.split('/')[-1]
if not filename.exists():
    urllib.request.urlretrieve(url, filename)
#import the download_and_load_gpt2 function from gpt_download.py which will load the GPT-2 architecture settings and weight parameters into our python session
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir= "gpt2"
)
#inspect the contents of settings and params
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
#inspect weight tensors by printing the whole dictionary via print(params)
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
#create a dictonary that lists the differences between the different GPT model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large(774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl(1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name="gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
#convert the token length from 256 to 1024
NEW_CONFIG.update({"context_length": 1024})
#enable bias vectors:
NEW_CONFIG.update({"qkv_bias": True})
#initialize a new GPTModel instance:
gpt=GPTModel(NEW_CONFIG)
gpt.eval()
#define a small assign utility function that checks whether two tensors or arrays (left and right)
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))
#define a load_weights_into_gpt function
import numpy as np
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb_weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb_weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign( 
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
#using load_weights_into_gpt to load the OpenAI model weights into our GPTModel instance model
load_weights_into_gpt(gpt, params)
gpt.to(device)
#generate new text using the previous generate function if the model is loaded successfully
torch.manual_seed(123)
token_ids=generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens = 25,
    context_size=NEW_CONFIG["context_length"],
    top_k = 50,
    temperature = 1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))