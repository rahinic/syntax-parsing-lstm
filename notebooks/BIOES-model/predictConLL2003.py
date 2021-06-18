############################################################################################
################################## 07. Model Predictions #####################################
from modelConLL2003 import RNNBIOESTagger
from datasetConLL2003 import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader

ds = SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-valid.txt")
#ds = SlidingWindowDataset()
x1, x2, y = ds.load_dictionaries()

# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
VOCAB_SIZE = len(x1)+len(x2)+2
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_OF_CLASSES = len(y)+1
EPOCHS = 5
LEARNING_RATE = 0.2
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


model.load_state_dict(torch.load("C:/Users/rahin/projects/paper-draft-03/notebooks/conLLmodel.pth"))
model.eval()

idx_to_BIOES = {}
print("Lets make predictions")

validation_dataset = DataLoader(dataset=SlidingWindowDataset("C:/Users/rahin/projects/paper-draft-03/data/raw/ConLL2003-bioes-valid.txt"),
                                batch_size=64,
                                shuffle=True)

for key, value in y.items():
    idx_to_BIOES[value] = key

print(idx_to_BIOES)
for idx, (sample, label) in enumerate(validation_dataset):

    if idx > 0:
        break
# print("Our sample to predict:")
# print(sample[0])
# print("Their actual label:")
# print(label[0])
print("current BIOES labels for Viterbi algo tokens")
actual_labels = []
for idx in label[0]:
    actual_labels.append(idx_to_BIOES[int(idx)])
print(actual_labels)

def predict(sentence, model):

    
    # print(f"{tokens_in_line} \n {tokens_to_idx}")

    # token idx to tensor conversion
    #print(sentence)
    idx_to_torch01 = torch.tensor(sentence, dtype=torch.int64)
    idx_to_torch = idx_to_torch01.unsqueeze(1).T


    with torch.no_grad():
        output = model(idx_to_torch)
        predicted_ouput=torch.argmax(output,dim=2)
        print(predicted_ouput)
        predicted_labels = []

        for pred in predicted_ouput:
            for i in pred:
                predicted_labels.append(idx_to_BIOES[int(i)])

        return predicted_labels

model = model.to("cpu")

example = sample[0]
predictions = predict(example, model)
print(predictions)             