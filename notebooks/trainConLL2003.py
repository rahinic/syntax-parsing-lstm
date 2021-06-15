import time
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from datasetConLL2003 import SlidingWindowDataset
from modelConLL2003 import RNNBIOESTagger

print("Step 1: loading Train/Test/Validation datasets...")
print("*"*100)     
validation_dataset = DataLoader(dataset=SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-valid.txt"),
                                batch_size=64,
                                shuffle=True)
print("*"*100)                                
test_dataset = DataLoader(dataset=SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-test.txt"),
                                batch_size=64,
                                shuffle=True)                                
print("*"*100)                                 
train_dataset = DataLoader(dataset=SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-train.txt"),
                                batch_size=64,
                                shuffle=True)
print("*"*100) 
print("All datasets successfully loaded!")


##########################################################################################
ds = SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-valid.txt")
#ds = SlidingWindowDataset()
x1, x2, y = ds.load_dictionaries()
print(f"Length of vocabulary is: {len(x1)} and Length of POS table is: {len(x2)}")
print(f"Length of Target look-up table is: {len(y)}")
################################# 01.Model Parameters ####################################
# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
VOCAB_SIZE = len(x1)+len(x2)+2
EMBED_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_OF_CLASSES = len(y)+1
EPOCHS = 5
LEARNING_RATE = 0.5
BATCH_SIZE = 64

print(f"Our vocab size to the model is therefore: {VOCAB_SIZE}")
################################### 02. NN Model  ########################################
print("Step 02. builing the model...")
model = RNNBIOESTagger(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0.2,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")
############################# 03. Optimizer and Loss  #################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

#define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    #correct = (rounded_preds == y)

    #correct = (rounded_preds == y).float() 

    _,pred_label = torch.max(rounded_preds, dim=0)
    correct = (pred_label == y).float()
    acc = correct.sum() / len(correct)
    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

##########################################################################################
############################## 04. NN Model Train Definition #############################

def train(model, dataset, optimizer, criterion):
    #log_interval = 500
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for idx, (sample, label) in enumerate(dataset):
       
       current_samples = sample
       current_labels = label
       #print(current_samples)
       optimizer.zero_grad()

       predicted_labels = model(current_samples)
       predicted_labels = predicted_labels.to(torch.float)
       current_labels = current_labels.to(torch.float)
    #    print(predicted_labels)
    #    print(current_labels)
    #    print(predicted_labels.size())
    #    print(current_labels.size())
       loss = criterion(predicted_labels, current_labels)
       accuracy = binary_accuracy(predicted_labels, current_labels)

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

##########################################################################################
################################ 05. NN Model Eval Definition ############################
def evaluate(model, dataset, criterion):
    
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for idx, (sample, label) in enumerate(dataset):
            current_samples = sample
            current_labels = label

            predicted_labels = model(current_samples)
            predicted_labels = predicted_labels.to(torch.float)
            current_labels = current_labels.to(torch.float)

            loss = criterion(predicted_labels, current_labels)
            accuracy = binary_accuracy(predicted_labels, current_labels)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

############################################################################################
################################## 06. NN Model training #####################################
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    print(epoch)
     
    #train the model
    train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, test_dataset, criterion)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print("-------------------------------------------------------------------")
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print("-------------------------------------------------------------------")

# for idx, (sample, label) in enumerate(validation_dataset):

#     if idx > 0:
#         break

#     print(sample[0].size())
#     print(label[0].size())

############################################################################################
################################## 07. Model Predictions #####################################

idx_to_BIOES = {}
print("Lets make predictions")

for key, value in y.items():
    idx_to_BIOES[value] = key


for idx, (sample, label) in enumerate(validation_dataset):

    if idx > 0:
        break
print("Our sample to predict:")
print(sample[0])
print("Their actual label:")
print(label[0])

def predict(sentence, model):

    
    # print(f"{tokens_in_line} \n {tokens_to_idx}")

    # token idx to tensor conversion
    #print(sentence)
    idx_to_torch01 = torch.tensor(sentence, dtype=torch.int64)
    idx_to_torch = idx_to_torch01.unsqueeze(1).T


    with torch.no_grad():
        output = model(idx_to_torch)
        print(output)
        #print(output)
        predicted_labels = []
        for item in output:
            predicted_labels.append(idx_to_BIOES[item])

        return predicted_labels

model = model.to("cpu")

example = sample[0]

print("This is:  %s" %idx_to_BIOES[predict(example, model)])             