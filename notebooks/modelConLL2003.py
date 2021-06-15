from torch import nn
import torch

"""RNN Many-to-many multi-class classification neural network model framework design"""

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
                            batch_first=True)

        self.fc = nn.Linear(hidden_dimension*2, 230)#output_dimension)

        self.activation_fn = nn.Tanh()


    def forward(self, sample):
        #print(sample)
        #print(sample.size())
        embedded = self.embedding(sample)
        #print(embedded.size())
        #print(self.lstm)
        output, (hidden, cell) = self.lstm(embedded)

        print(hidden[0][0])
        print(output[0][0])
        
        print(f"{hidden.size(), output.size(), output.T.size()}")

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        print(hidden[0])
        print(hidden.size())

        dense_output = self.fc(hidden)
        

        #activation function
        outputs=self.activation_fn(dense_output)
        print(outputs[0])
        print(outputs.size())
        return outputs
        #return dense_output