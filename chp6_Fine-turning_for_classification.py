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