#This is code implementation of the book "Build a Large Language Model from Scratch" 
#begins on 05/18/2026, in ch2
#Hail Mary!
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

#Then split the text into a list of tokens
import re

text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text) #split the text by comma, period, or whitespace without removing the delimiters

result = [item for item in result if item.strip()]  #remove empty strings and whitespace-only strings from the result list

# the basic tokenizer for the story text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(len(preprocessed))
#print the first 30 tokens
print(preprocessed[:30])
#build a vocabulary of unique tokens, the index of each token is used as its token ID
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
# print the first 51 items in the vocabulary
vocab = {token: integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV2: # This is a simple but complete tokenizer class
    def __init__(self, vocab):
        self.str_to_int = vocab  # stores the vocabulary as a class attribute for access in the encode and decode methods
        # creates a inverse vocabublary that maps token IDs back to the original text tokens.
        # Iterates through each key-value pair in vocab, assigning the key to s and the value to i.
        self.int_to_str = {i:s for s,i in vocab.items()} 

    def encode(self, text):  # process input text into token IDs
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed = [ item.strip() for item in preprocessed if item.strip()]

        #replace unknown word with<|unk|> token
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>"for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): # converts token IDs back to the text
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # remove space before punctuation
        return text
    
tokenizer = SimpleTokenizerV2(vocab) # an simple tokenizer instance
text = """"It's the last he painted, you know," 
          Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
#print(ids)  

print(tokenizer.decode(ids)) #Token ID back to text

#appling the tokenizer to a simple text sample not in the training set
text = "Hello, do you like the tea?"
#This will return a keyerror because the training set does not contain the word "Hello", so we need larger training set to cover more words. 
# print(tokenizer.encode(text))
#handling unknown tokens
all_tokens =sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))
# print the last 5 tokens
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# a practical sample of two independent texts
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1,text2))
print(text)

tokenizer =SimpleTokenizerV2(vocab)
#1130 ID is for the <|endoftext|>, which 2 1130 are for <|unk|>
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

"""Byte pair encoding(BPE)"""
#BPE is based on OpenAI's open source library: tiktoken
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
#initial BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

"""2.6-data-sample-with-sliding-window"""
#tokenize the Verdict by BPE
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

#removing the first 50 tokens from the dataset for demostration purposes(author mentioned it will"results a slightly more interesting text passage later")
enc_sample = enc_text[50:]
#context size determines how many tokens are included in the input
context_size  = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

#the next-word prediction tasks:
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
#convert previous token IDs back to text
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#dataset class  based on PyTorch built-in Dataset and DataLoader classes
import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def  __init__(self, txt, tokenizer, max_length, stride): # stride controls how many tokens the sliding window moves forward each time
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #tokenize the entire input files
        #use a sliding window to chunk the book into overlapping sequencces of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i+max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        #returns the total number of row from the dataset
    def __len__(self):
        return len(self.input_ids)
    #returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

"""use GPTDatasetV1 to load the inputs in batches via PyTorch DataLoader."""
# drop_last = True means that the last batch will be dropped if it is smaller than the specified batch size, which prevents 
# the model from training on incomplete batches that may not provide enough information for learning, especially when the dataset size is not perfectly divisible by the batch size.  
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

#in actual LLM training, we usually use much larger context size(max_length) like 256 or even larger. 
#batch_size controls how many sequences are included in each batch, while max_length controls how many tokens are included in each sequence.  
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
#Converts dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch) 
#show the result of stride = 1
second_batch = next(data_iter)
print(second_batch)
# use data loader to sample with a batch size greater than 1:
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter=iter(dataloader)
inputs, targets=next(data_iter)
print("Inputs\n", inputs)
print("\nTargets:\n", targets)