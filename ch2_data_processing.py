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

class SimpleTokenizerV1: # This is a simple but complete tokenizer class
    def __init__(self, vocab):
        self.str_to_int = vocab  # stores the vocabulary as a class attribute for access in the encode and decode methods
        # creates a inverse vocabublary that maps token IDs back to the original text tokens.
        # Iterates through each key-value pair in vocab, assigning the key to s and the value to i.
        self.int_to_str = {i:s for s,i in vocab.items()} 

    def encode(self, text):  # process input text into token IDs
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed = [ item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): # converts token IDs back to the text
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # remove space before punctuation
        return text
    
tokenizer = SimpleTokenizerV1(vocab) # an simple tokenizer instance
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)  

print(tokenizer.decode(ids)) #Token ID back to text

#appling the tokenizer to a simple text sample not in the training set
text = "Hello, do you like the tea?"
#This will return a keyerror because the training set does not contain the word "Hello", so we need larger training set to cover more words. 
print(tokenizer.encode(text))