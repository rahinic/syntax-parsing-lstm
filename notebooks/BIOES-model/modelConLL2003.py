from typing import final
from torch import nn
import torch

"""RNN Many-to-many multi-class classification neural network model structure definition"""

class RNNBIOESTagger(nn.Module):

    def __init__(self, 
                embedding_dimension, 
                vocabulary_size,
                hidden_dimension,
                num_of_layers,
                dropout,
                output_dimension
                ):
        super(RNNBIOESTagger, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=embedding_dimension)

        self.lstm = nn.LSTM(embedding_dimension,
                            hidden_dimension,
                            num_of_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dimension*2, output_dimension)#230)#
        # self.fc = nn.Linear(hidden_dimension, output_dimension)

        self.activation_fn = nn.Tanh()


    def forward(self, sample):

        # (1)- Embedding layer
        # Note: batch size x sample size x embedding dim 64x230x50
        # sample size x embedding dimension 230x50
        embedded = self.embedding(sample)
        # print(f"Dimension of Embedding layer of batch and single sample:{embedded.size(), embedded[0].size()} respectively")

        #-------------------------------------------------------------------------

        #(2)- LSTM layer 1
        # Note: Input from Embedding layer. Dim: 64x230x50
        # lstm2out dim: 64x230x32 and lstm2hidden dim: 1x64x32 
        output, (hidden, cell) = self.lstm(embedded)       
        # print(f"Dimenstion of Hidden and Output from LSTM layer: {hidden.size(), output.size()}")

        #-------------------------------------------------------------------------

        #concat the final forward and backward hidden state
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.cat((hidden[-1,:,:], hidden[0,:,:]), dim = 1)


        #(3)- LSTM to linear layer: Final set of tags
        dense_output = self.fc(output)
        # print(f"LSTM to Linear layer output dimension {dense_output.size()}")
        

        #activation function
        outputs=self.activation_fn(dense_output)
 
        return outputs
