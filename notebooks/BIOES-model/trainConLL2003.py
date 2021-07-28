import time
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from datasetConLL2003 import SlidingWindowDataset
from modelConLL2003 import RNNBIOESTagger

print("Step 1: loading Train/Test/Validation datasets...")
print("*"*100)     
validation_dataset = DataLoader(dataset=SlidingWindowDataset("data/raw/ConLL2003-bioes-valid.txt"),
                                batch_size=64,
                                shuffle=True)
print("*"*100)                                
test_dataset = DataLoader(dataset=SlidingWindowDataset("data/raw/ConLL2003-bioes-test.txt"),
                                batch_size=64,
                                shuffle=True)                                
print("*"*100)                                 
train_dataset = DataLoader(dataset=SlidingWindowDataset("data/raw/ConLL2003-bioes-train.txt"),
                                batch_size=64,
                                shuffle=True)
print("*"*100) 
print("All datasets successfully loaded!")


##########################################################################################
idx_to_BIOES = {}
ds = SlidingWindowDataset("data/raw/ConLL2003-bioes-valid.txt")
x1, x2, y = ds.load_dictionaries()
for key, value in y.items():
    idx_to_BIOES[value] = key
print(f"Length of vocabulary is: {len(x1)} and Length of POS table is: {len(x2)}")
print(f"Length of Target look-up table is: {len(y)}")

################################# 01.Model Parameters ####################################

# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
VOCAB_SIZE = len(x1)+len(x2)+2
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_OF_CLASSES = len(y)+1
N_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32

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

############################# 03. Optimizer and Loss  ####################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train_accuracy(preds, y):
    predicted_labels_dirty = preds.permute(0,2,1)
    predicted_labels_final = torch.argmax(predicted_labels_dirty, dim=2).tolist()
    actual_labels_final = y.tolist()
    accuracy_of_all_lines = []
    for predicted, actual in zip(predicted_labels_final, actual_labels_final):
        counter = 0
        for pred,act in zip(predicted,actual):
            if pred == act:
                counter = counter+1
        accuracy_of_this_line = counter/50
        accuracy_of_all_lines.append(accuracy_of_this_line)
    accuracy = sum(accuracy_of_all_lines)/len(predicted_labels_final)

    return accuracy

#define metric
def binary_accuracy(preds, y):  #decommisioned 28.07.2021
    
    predsx = preds.permute(0,2,1) #reshape
    predsx2 = torch.argmax(predsx, dim=2) #find BIOES index with max value for each token
    correct = (predsx2 == y)
    acc = correct.sum() / len(preds)
    # print(type(acc))

    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

##########################################################################################
############################## 04. NN Model Train Definition #############################

def train(model, dataset, optimizer, criterion):

    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    epoch_dataset_length_total = 0
    epoch_dataset_length.append(len(dataset))

    model.train()

    for idx, (sample, label) in enumerate(dataset):
       
       current_samples = sample
       current_labels = label

       optimizer.zero_grad()

       predicted_labels = model(current_samples).permute(0,2,1)
       
    #    print(f"Predicted label dimension: {predicted_labels.size()} and actual label dimension: {current_labels.size()}")
    #    print(predicted_labels)
    #    print(current_labels)
    #    predicted_labels = predicted_labels.to(torch.float)
    #    current_labels = current_labels.to(torch.float)
       loss = criterion(predicted_labels, current_labels)
       accuracy = train_accuracy(predicted_labels, current_labels)

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_accuracy += accuracy
    #    epoch_dataset_length_total += epoch_dataset_length.item()

    #    epoch_accuracy =0

    return epoch_loss/len(dataset), epoch_accuracy/sum(epoch_dataset_length)

##########################################################################################
################################ 05. NN Model Eval Definition ############################
def evaluate(model, dataset, criterion):
    
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for idx, (sample, label) in enumerate(dataset):
            current_samples = sample
            current_labels = label

            predicted_labels = model(current_samples).permute(0,2,1)
            # predicted_labels = predicted_labels.to(torch.float)
            # current_labels = current_labels.to(torch.float)

            loss = criterion(predicted_labels, current_labels)
            accuracy = train_accuracy(predicted_labels, current_labels)
            #epoch_accuracy = 0

            epoch_loss += loss.item()
            epoch_accuracy += accuracy

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

############################################################################################
################################## 06. NN Model training #####################################
#N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    print(f"Epoch #: {epoch}")
    epoch_dataset_length = []
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

modelpath = "notebooks"
torch.save(model.state_dict(), os.path.join(modelpath, "conLLmodel.pth"))
# print(y.keys())