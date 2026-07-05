"""This is for chapter 6, "Fine-tuning for Classification"""
"""This chapter covers the classification fine-tuning about spam email detection"""
"""instruction fine-tuning is in chpater 7"""

"""-------------------------------6.2 Preparing the dataset----------------------------"""
#download and unzip the dataset
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    #download the zip file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    #unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    #add a .tsv file extansion
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
#previous code will download the dataset and save it as SMSSpamCollection.tsv in the sms_spam_collection folder.
#load it into a pandas dataframe
import pandas as pd
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
#renders the data frame in a Jupyter notebook. Alternatively, using print(df)
df
#examine the class label distribution
print(df["Label"].value_counts())
#creating a balanced dataset
def create_balanced_dataset(df):
    #count the instances of "spam"
    num_spam = df[df["Label"]=="spam"].shape[0]
    #randomly sample the same number of "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"]=="ham"].sample(n=num_spam, random_state=123)
    #combines ham subset with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
#convert the labels to binary values, where ham is 0 and spam is 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

#split the dataset (70% training, 10% validation, 20% testing)
def random_split(df, train_frac, validation_frac):
    #shuffles the entire DataFrame
    df = df.sample(frac = 1, random_state=123).reset_index(drop=True)
    #calculating split indices based on the specified fracs
    train_end = int(len(df)*train_frac)
    validation_end = train_end + int(len(df)*validation_frac)
    #splits the DataFrame into training, validation, and testing sets
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
#test size is implied to be 20% as remainder
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

#save the datasets as CSV
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

"""-------------------------------6.3 Creating Data Loaders----------------------------"""
#to save the important information, we pad the sequences to fixed length(length of the longest sequence in the batch), rather than truncate the length to the shortest sequence
# we use "<|endoftext|>" as the padding token

#we can add the token ID of "<|endoftext|>" to encoded text messages, using GPT2 tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

#implement the PyTorch Dataset class
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)
        #Pretokenizes texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        #truncates sequences if thery are longer than max_length
            self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
            ]
        #pads sequences to max_length
        #this is the list comprehension that pads each encoded text with the pad_token_id to ensure they all have the same length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length   
#create the batches in the training data loader
train_dataset = SpamDataset("train.csv", max_length = None, tokenizer = tokenizer)

print(train_dataset.max_length)
#then pad the validation and test datasets to the same length as the training dataset
val_dataset = SpamDataset(csv_file = "validation.csv", max_length = train_dataset.max_length, tokenizer = tokenizer)
test_dataset = SpamDataset(csv_file = "test.csv", max_length = train_dataset.max_length, tokenizer = tokenizer)
#creates the training, validation, and test data loaders that load the text messages and labels in batches of size 8
from torch.utils.data import DataLoader

#ensure compaptility with most computers
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, drop_last = True
)
val_loader = DataLoader(
    dataset = val_dataset, batch_size = batch_size, num_workers = num_workers, drop_last = False
)
test_loader = DataLoader(
    dataset = test_dataset, batch_size = batch_size, num_workers = num_workers, drop_last = False
)
# Iterate through every batch to confirm the whole train_loader can be consumed
# without errors (catches shape/collation issues in any batch, not just the first).
# The loop body is empty on purpose: we only care about exhausting the iterator.
# input_batch/target_batch are created by tuple-unpacking each yielded batch, and
# Python has no block scope, so after the loop they still hold the LAST batch's values.
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)
#the total number of batches showing the dataset size
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

"""-------------------------------6.4 Initializing the model with pretrained weights----------------------------"""
#employ the same configurations we used to pretrain unlabeled data
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl(1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

#use the GPT2 model class and weights previously downloaded and chpater 5
from gpt_download import download_and_load_gpt2
from chp5_Pretraining_on_unlabed_data import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size =model_size, models_dir ="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

#reuse the text generation utility function from chapter 4, 5 to ensure that the model generates coherent text
from ch4_Implementation_of_LLM_architecture import generate_text_simple
from chp5_Pretraining_on_unlabed_data import text_to_token_ids, token_ids_to_text

text_1 = "Every effort moves you"
token_ids = generate_text_simple(model = model, idx = text_to_token_ids(text_1, tokenizer),max_new_tokens=15, context_size = BASE_CONFIG["context_length"])
print(token_ids_to_text(token_ids, tokenizer))

#test whether the model can classify spam messages by prompting with instructions now
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text_2, tokenizer),max_new_tokens=23, context_size =BASE_CONFIG["context_length"])
print(token_ids_to_text(token_ids, tokenizer))

"""-------------------------------6.5 Adding a classification head---------------------------------"""
# print the model architecture before adding the classification head
print(model)
#freeze the model to get the model ready fro classification fine-tuning
for param in model.parameters():
    param.requires_grad = False
#add a classification layer
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    #we use emb_dim of GPT2 as the input dimension to make it more generalizable, so we can
    #use the same classify head with differnt GPT2 model sizes
    in_features = BASE_CONFIG["emb_dim"],
    out_features = num_classes
)
#set the requires_grad of LayerNorm and last transformer block to True to make them trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
#feed it an example text identical to our previously used example text:
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)  # Add batch dimension
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape) #shape:[batch_size, num_tokens]

#Then we pass the encoded token IDs to the model as usual
with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)
#use the following code to extract the last output token from then output tensor
print("Last output token:", outputs[:, -1, :])
"""--------------------------------6.6 calculating the loss for classification---------------------------------"""
#obtain the class label of last token:
logits = outputs[:,-1,:] 
label = torch.argmax(logits)
print("Class label:", label.item())

#calculate the classification accruracy
def calc_accuracy_loader(data_loader, model, device, num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))