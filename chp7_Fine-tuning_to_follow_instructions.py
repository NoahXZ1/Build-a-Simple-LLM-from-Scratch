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