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
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] #logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples +=predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        
        else:
            break
    return correct_predictions / num_examples
#use the calc_accuracy_loader function to calculate the accuracy of the model on the validation dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Traning accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

#define the loss function
#because we are dealing with a binary classification problem, we cannot use accuracy as the loss function, as it is not differentiable
#we use the cross-entropy instead, which measures the confidence of the model's predictions, penalizing incorrect predictions, so its differentiable

#and we only calculate the loss for last output token, which contains all tokens' information
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:] # logits of last output token

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
#calc_loss_batch is only for a single batch
#we use calc_loss_loader to calculate the loss for all batches in the dataloader
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    
    else: # ensure the number of batches doesn't exceed batches in data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss =  calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() # detach from computation graph to avoid memory buildup
        else:
            break
    return total_loss / num_batches
# compute initial loss for each dataset
with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches = 5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches = 5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches = 5)
print(f"Initial training loss: {train_loss:.3f}")
print(f"Initial validation loss: {val_loss:.3f}")
print(f"Initial test loss: {test_loss:.3f}")
"""-------------------------------6.7 Fine-tuning the model for classfication----------------------------"""
#function used for training the model for classification fine-tuning
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    #initialize lists to trach losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0,-1

    #main training loop
    for epoch in range(num_epochs):
        #sets model to training mode
        model.train()

        for input_batch, target_batch in train_loader:
            #reset loss gradients from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            #calculate loss gradients for the model parameters
            loss.backward()
            #updates model weights using loss gradients
            optimizer.step()
            #New part: tracks examples instead of tokens
            examples_seen += input_batch.shape[0]
            global_step += 1
            #Optional: evaluate the model on the training and validation datasets at specified intervals
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
        #calculate the accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches = eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches =eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
#the evaluate_model is same as what we used before for pretraining
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
        model.train()#return the model to training mode after evaluation
        return train_loss, val_loss

#initialize the optimizer, set the number of epochs and initialize the training using train_classifier_simple_function
import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 0.1)
num_epochs =5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq = 50, eval_iter = 5
    )

end_time =time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

#Then we use Matplotlib to plot the loss function for the training and validation set
import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5,3))
    #Plots training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    #creates a second x-axis for examples seen
    ax2 = ax1.twiny()
    #invisible plot for aligning ticks
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    #Adjusts layout to make room
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
#Using the same plot_values function, let's now plot the classification accuracies:
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
#plot_values function is reused to plot the classificationaccuracies
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
#calculate the training performance of training, validation, and test datasets after fine-tuning
training_accuracy = calc_accuracy_loader(train_loader, model, device)
validation_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training Accuracy: {training_accuracy*100:.2f}%")
print(f"Validation Accuracy: {validation_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

"""-------------------------------6.8 Using the Model for Spam Detection--------------------------------"""
#the following function takes a text message as input and returns whether its spam or not using the fine-tuned model. 
#It encodes input text, truncates or pads it, and passes it to the model to do the classification
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    #prepares inputs to the model
    input_ids= tokenizer.encode(text)
    #if max_length is not None:
    supported_context_length = model.pos_emb.weight.shape[0]
    #truncates sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    #pad sequences to the longest sequence's length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    #adds batch dimension
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:] #logits of last output token
    predicted_model =torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_model == 1 else "not spam"
#try on 2 samples
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

#Save the model using torch.save
torch.save(model.state_dict(), "review_classifier.pth")

"""the model can be reloaded by:"""
#model_state_dict = torch.load("review_classifier.pth, map_location = device")
#model.load_state_dict(model_state_dict)
