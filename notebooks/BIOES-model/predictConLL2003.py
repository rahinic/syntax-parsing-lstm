############################################################################################
################################## 07. Model Predictions #####################################
from modelConLL2003 import RNNBIOESTagger
from datasetConLL2003 import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import numpy as np

ds = SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-valid.txt")

x1, x2, y = ds.load_dictionaries()

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

# load trained model
model.load_state_dict(torch.load("/notebooks/conLLmodel.pth"))
model.eval()

idx_to_BIOES = {}
print("Lets make predictions")

validation_dataset = DataLoader(dataset=SlidingWindowDataset("/data/raw/ConLL2003-bioes-valid.txt"),
                                batch_size=64,
                                shuffle=True)

for key, value in y.items():
    idx_to_BIOES[value] = key

# print(idx_to_BIOES)

def predict(sentence, model):

    # token idx to tensor conversion

    idx_to_torch01 = torch.tensor(sentence, dtype=torch.int64)
    idx_to_torch = idx_to_torch01.unsqueeze(1).T


    with torch.no_grad():
        output = model(idx_to_torch)
        predicted_ouput=torch.argmax(output,dim=2)
        
        predicted_labels = []

        for pred in predicted_ouput:
            for i in pred:
                predicted_labels.append(idx_to_BIOES[int(i)])

        return output, predicted_labels

model = model.to("cpu")
# ==============================================================================
# ACCURACY & PRECISION CALCULATIONS

total_accuracy = []
length_of_sentence = []

def model_accuracy_precision():

    """ returns accuracy of the model using the validation dataset."""

    for idx, (sample,actualy) in enumerate(validation_dataset):


        for x,y in zip(sample,actualy):

            labelsy = torch.squeeze(y,dim=-1)  #actual labels
            probsy, predictedy = predict(x, model) #predicted labels
            

            actual_labels = []
            for idx in labelsy:
                actual_labels.append(idx_to_BIOES[int(idx)])
                

            correct = np.array(actual_labels) == np.array(predictedy) # boolean comparison

            total_accuracy.append(correct.sum())
            length_of_sentence.append(len(x))

            acc=sum(total_accuracy)/sum(length_of_sentence)*100

    return round(acc,2)

print(f"Total Accuracy of our model is: {model_accuracy_precision()}%")
# ======================================================================================

for idx, (sample, label) in enumerate(validation_dataset):

    if idx > 0:
    
        break

print("One example:")
actual_labels, actual_sentence = [], []
for idx in label[0]:
    actual_labels.append(idx_to_BIOES[int(idx)])

print(actual_labels)
example = sample[0]
probsy, predictions = predict(example, model)
probsy_np = probsy.cpu().detach().numpy()
probsy_np =  np.squeeze(probsy_np, axis=0)

print(predictions)
# ====================================================================================
# Export the results of our predictions and their corresponding probabilities.
# This will be used as input to the viterbi algorithm

# Step 1: Export our BIOES predictions
FILEPATH = "/data/processed"

textfile = open(FILEPATH+"/sentence.txt", "w")
for element in predictions:
    textfile.write(element + "\n")
textfile.close()

# Step 2: Export the individual probability of each BIOES tag, given each words+POS tags predictions

np.save(FILEPATH+"/tags_probabilities01.npy", probsy_np)


