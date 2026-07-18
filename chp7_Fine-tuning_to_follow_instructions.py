"""This is for chpater 7: fine-tuning a model for following instructions"""
#7.2 Preparing a dataset for supervised fine-tuning instruction fine-tuning
import json
import os
import urllib.request

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data =response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction_data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
#the data list loaded froM json file contains 1100 entries of instruction dataset
print("Example entry:\n", data[50])

#another entry, showing the 'input' field maybe empty
print("Another example entry:\n", data[999])
#define the function to convert the entries in the data list into the Alpaca-style input

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instrcution:\n{entry['instruction']}"
    )

    input_text = ( f"\n\n### Input:\n{entry['input']}" if entry["input"] else "")
    
    return instruction_text + input_text
#make a test using data[50]
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
#test using data[999] (###input is empty)
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

#Partitioning the dataset, (split it to training, validation and test sets)
train_portion = int(len(data)*0.85)  #85% data for training
test_portion = int(len(data)*0.10)  #10% data for testing
val_portion = len(data) - train_portion - test_portion  #5% data for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length: ", len(train_data))
print("Validation set length: ", len(val_data))
print("Test set length: ", len(test_data))

"""------------------------------7.3 organizing data into training dataset----------------------------"""
#implement a instruction dataset class
#the class is used to implement format_input(formatting the input data) and pretokenize all inputs in the dataset
import torch
from torch.utils.data import  Dataset
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        #pretokenize all the inputs in the dataset
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
#2.3: then we should padding the encoded texts to the same length
#using the token ID of <|endoftext> directly, which is 50256 for GPT2 tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
#implement the custom collate function to pad the encoded texts to the same length
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    #finds the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []

    for item in batch:  #pads and prepares inputs for each item in the batch
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        #removes extra padded token added earlier
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    #convert the list of inputs to a tensor and transfer it to the target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
#test the custom collate function
inputs_1 = [0,1,2,3,4]
inputs_2 = [5,6]
inputs_3 = [7,8,9]
batch = (inputs_1, inputs_2, inputs_3)
print(custom_collate_draft_1(batch))

#step2.4: create a list of target token IDs for the model to learn
#get the target token IDs by shifting the input token IDs to the right by 1 position, and the last token ID is set to the pad_token_id
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        #truncates the last token ID for inputs 
        inputs = torch.tensor(padded[:-1])
        #shifts the input token IDs to the right by 1 position for targets
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)
#a new custom collate function allows replace the pad_token_id with -100, and max_length to limit the length of the inputs and targets
def custom_collate_fn(batch, pad_token_id=50256,ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        #pads sequences to max_length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        #trucates the last tokens for inputs
        inputs = torch.tensor(padded[:-1])
        #shifts +1 to the right for taregst
        targets = torch.tensor(padded[1:])
        #replaces all but the first padding tokens in targets by ignore_index
        #mask is a boolean tensor which has the same shape as targets, where each element is True if the corresponding element in targets is equal to pad_token_id, and False otherwise.
        mask=targets == pad_token_id  
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        #trucates the inputs and targets to allowed_max_length if needed (optional)
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
#try the sample using the earlier collate function
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
#an example of calculating the loss
logits_1 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]]) #prediction for 1st and 2nd token in the sequence
targets_1 = torch.tensor([0,1])  #Correct target token indices to generate
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)
#demostration: adding additional token ID will affects the loss calculation
logits_2 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_2 = torch.tensor([0,1,1])  #the last token ID is set to -100, which will be ignored in the loss calculation
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)
#replace the third token ID with -100:
targets_3 = torch.tensor([0,1,-100])  #the last token ID is set to -100, which will be ignored in the loss calculation
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

"""------------------------------7.4 Creating data loaders for training and evaluation----------------------------"""
#initialize the device variable to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available(): (for macOS)
#     device = torch.device("mps")
print("Device:", device)
#create a new version of the function with the device argument prefilled by the partial function from the functools module
from functools import partial

customized_collate_fn = partial( custom_collate_fn, device = device, allowed_max_length = 1024)
#set up the dataloaders using custom collate function for batching process
from torch.utils.data import DataLoader

#Windows needs `if __name__ == "__main__":` guard for multiprocessing DataLoader; this script lacks it, so keep 0
num_workers = 0
batch_size = 8

torch.manual_seed(123)  #for reproducibility

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last = True, num_workers=num_workers)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last = False, num_workers=num_workers)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last = False, num_workers=num_workers)

#examine the dimensions of the first batch of inputs and targets from the training dataloader
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
"""------------------------------7.5 Loading the pretrained model and preparing it for fine-tuning----------------------------"""
#we will use the mid-sized GPT-2 model, as small size is not sufficient for instruction fine-tuning
import numpy as np
from gpt_download import download_and_load_gpt2
from ch4_Implementation_of_LLM_architecture import GPTModel

# NOTE: load_weights_into_gpt is copied here (instead of imported from
# chp5_Pretraining_on_unlabed_data) because that module has no
# `if __name__ == "__main__":` guard -- importing it re-runs its entire
# demo pipeline (builds/trains its own models, re-downloads a GPT-2
# checkpoint, and moves several models onto the GPU), which was stacking
# on top of this script's own model and causing CUDA OOM errors.
def assign(left, right):
    right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          f"Right: {right.shape}"
        )
    return torch.nn.Parameter(right.clone().detach())


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

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

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,   #dropout rate
    "qkv_bias": True    #query-key-value bias
}

model_configs = {
    "gpt2-small (124M)":{"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)":{"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":{"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":{"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Extract "355M" from "gpt2-medium (355M)" for use as the download param
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(model_size = model_size, models_dir = "gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();
