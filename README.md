# Syntax Parsing with Recurrent Neural Network

## Dataset: 
CoNLL-2003 dataset for NER tagging. The IOB tagging scheme were converted to BIOES using the repo: https://github.com/rahinic/BIO-to-BIOES-tagger

## Neural Network Model: 
Using a supervised many-to-many LSTM model, the neural network tags the tokens and corresponsing POS tags to BIOES tagging scheme with 91.25% accuracy.

## Necessary Packages: 
pickle, pytorch, numpy

## Reference:
"Joint RNN-Based Greedy Parsing and Word Composition" by Legrand & Collobert, ICLR 2015.
<TBA>
