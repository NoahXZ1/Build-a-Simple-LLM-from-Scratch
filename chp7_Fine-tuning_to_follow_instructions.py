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