"""This is for chapter 5, "Pretraining on Unlabeled Data"""
"""-----------------------------------5.1.1 using GPT to generate text(recall of chp4)------------------------------------"""
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
    encoded = tokenizer.encode(text, allowed_special={'|endoftext|>'})
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
device = torch.device("cude" if torch.cuda.is_available() else "cpu")
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
Model= GPTModel(GPT_CONFIG_124M)
model.to(device)
#lr is the learning rate paramter, controls step size at each iteration.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #the parameters() returns all trainable parameters of model
num_epochs = 10
train_losses, val_losses, track_tokens_seen=train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq=5, eval_iter=5,
    start_context = "Every effort moves you", tokenizer = tokenizer)
