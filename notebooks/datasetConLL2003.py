import pickle
from typing import List, Tuple
from torch._C import dtype
from torch.utils.data import Dataset
import torch

class SlidingWindowDataset(Dataset):

    def file_open(self, filepath: str, ftype: str):
        
        if ftype == "pickle":

            file = open(filepath, "rb")
            content = pickle.load(file)
            file.close()
        else:
            filename = filepath.split('/')[-1]
            print(f"loading dataset: {filename}")
            content = open(filepath, mode='r').read()

        return content 

    def load_dictionaries(self):

        print("loading dictionaries...")

        list_of_dicts = ["C:/Users/rahin/projects/paper-draft-03/data/interim/ConLL2003_vocabulary.pkl",
                        "C:/Users/rahin/projects/paper-draft-03/data/interim/ConLL2003_pos_tags.pkl",
                        "C:/Users/rahin/projects/paper-draft-03/data/interim/ConLL2003_BIOES_tags.pkl"]
        
        vocabulary = self.file_open(filepath= list_of_dicts[0], ftype= "pickle")
        pos_tags = self.file_open(filepath= list_of_dicts[1], ftype= "pickle")
        bioes_tags = self.file_open(filepath= list_of_dicts[2], ftype= "pickle")

        vocabulary['PADDING'] = len(vocabulary)+1
        pos_tags['PADDING'] = len(pos_tags)+1
        bioes_tags['PADDING'] = len(bioes_tags)+1

        print("done")

        return vocabulary, pos_tags, bioes_tags

    def file_processing(self):

        print("Performing dataset pre-processing activities...")

        all_samples, all_pos_tags, all_bioes_tags, all_samples_tuples = [], [], [], []

        #max_sentence_length = []
        dataset = self.file_open(filepath = self.mydataset, ftype="dataset")
        #dataset = self.file_open(filepath= r"C:\Users\rahin\projects\paper-draft-03\data\raw\ConLL2003-bioes-valid.txt", ftype="dataset")
        dataset = dataset.split(". . O O") #break by sentences

        for sentence in dataset: # each sentence

            

            sentence = sentence.split('\n') # each line
            all_words, all_pos, all_bioes = [], [], [] # refresh for each sentence
            for words in sentence:

                if len(words.split(' ')) > 1:
                    all_words.append(words.split(' ')[0])
                    all_pos.append(words.split(' ')[1])
                    all_bioes.append(words.split(' ')[2])

            #max_sentence_length.append(len(all_words))

            all_samples.append(all_words)
            all_pos_tags.append(all_pos)
            all_bioes_tags.append(all_bioes)

        #padding logic max: 1178


            all_samples_tuples.append(list(zip(all_samples, all_pos_tags, all_bioes_tags))) # collate the information
        
        print("done")
            
        return all_samples_tuples

    def file_parser(self, processed_samples: str) -> Tuple[List, List]:

        print("Parsing the dataset now...")

        def sample_word_pipeline(x):
 
            return [self.vocabulary[tok] for tok in x]
        
        def sample_pos_pipeline(x):
            return [self.pos_tags[pos] for pos in x]

        def label_pipeline(x):
            return [self.bioes_tags[bioes] for bioes in x]

        def sliding_window(x):
            window1, window2 = [], []
            start, stop = 0, 5
            for i in range(0,46):
                window1.append(x[start:stop])
                start +=1
                stop +=1

            for window in window1:
                for win in window:
                    window2.append(win)
            return window2
        # sliding window (thanks to https://diegslva.github.io/2017-05-02-first-post/)
        def pytorch_rolling_window(x, window_size, step_size=1):

            # window = []
            # start, stop = 0, window_size-1
            # for i in range(0,45):
            #     window.append(x[start:stop])
            #     start = start+step_size
            #     stop = stop+step_size
            # return window
            # unfold dimension to make our rolling window
            #return torch.tensor(x, dtype=torch.int64).unfold(0,window_size,step_size)
            return x.unfold(0,window_size,step_size)

        samples, labels = [], []

        print("converting tokens to indices to tensors")

        for idx, sample in enumerate(processed_samples):

            current_sample = list(sample)[idx][0]

            if len(current_sample) > 50:
                continue
            else:
                # padding logic
                for padding in range(50-len(current_sample)):
                    current_sample.append('PADDING')

            current_sample_to_idx = sample_word_pipeline(current_sample)
            current_sample_idx_to_windows = sliding_window(current_sample_to_idx)
            current_sample = torch.tensor(current_sample_idx_to_windows, dtype=torch.int64)
            #current_sample = pytorch_rolling_window(current_sample_to_tensor,5)

            # current_sample_window = pytorch_rolling_window(sample_word_pipeline(current_sample),5)
            # current_sample02 = [] 
            # for window in current_sample_window:
                
            #     current_sample02.append(torch.tensor(window, dtype=torch.int64))

            #current_sample = torch.tensor(current_sample_to_idx, dtype=torch.int64)
            #current_sample = pytorch_rolling_window(current_sample_to_tensor,5)
            

            #current_sample = torch.tensor(sample_word_pipeline(current_sample), dtype=torch.int64)
            current_pos = sample[idx][1]

            if len(current_pos) > 50:
                continue
            else:
                # padding logic
                for padding in range(50-len(current_pos)):
                    current_pos.append('PADDING')

            current_pos_to_idx = sample_pos_pipeline(current_pos)
            current_pos_idx_to_windows = sliding_window(current_pos_to_idx)
            current_pos = torch.tensor(current_pos_idx_to_windows, dtype=torch.int64)
            #current_pos = pytorch_rolling_window(current_pos_to_tensor,5)

            # current_pos_window = pytorch_rolling_window(sample_pos_pipeline(current_pos),5)
            # current_pos02 = []
            # for pos in current_pos_window:
            #     current_pos02.append(torch.tensor(pos, dtype=torch.int64))
            #current_pos = torch.tensor(current_pos_to_idx, dtype=torch.int64)
            #current_pos = torch.tensor(sample_pos_pipeline(current_pos), dtype=torch.int64)

            current_bioes = sample[idx][2]

            if len(current_bioes) > 50:
                continue
            else:
                # padding logic
                for padding in range(50-len(current_bioes)):
                    current_bioes.append('PADDING')

            current_bioes_to_idx = label_pipeline(current_bioes)
            current_bioes_idx_to_windows = sliding_window(current_bioes_to_idx)
            current_bioes = torch.tensor(current_bioes_idx_to_windows, dtype=torch.int64)  
           #current_bioes = pytorch_rolling_window(current_bioes_to_tensor,5)  
            # current_bioes_window = pytorch_rolling_window(label_pipeline(current_bioes),5)
            # current_bioes02 = []
            # for bioes in current_bioes_window:
            #     current_bioes02.append(torch.tensor(bioes, dtype=torch.int64))
            #current_bioes = torch.tensor(current_bioes_to_idx, dtype=torch.int64)        

            #current_bioes = torch.tensor(label_pipeline(current_bioes), dtype=torch.int64)

            # current_sample03 = []
            # for token, pos in zip(current_sample02, current_pos02):
            #     current_sample03.append(token+pos)

            # if len(current_sample03) != len(current_bioes02):
            #     print("Attentions! Lengths don't match here:")
            #     print(current_sample03)
            #     print(current_bioes02)

            if len(current_sample+current_pos) != len(current_bioes):
                print("Attention! Lengths don't match here:")
                print(current_sample)
                print(current_pos)
                print(current_bioes)
                exit

            samples.append(current_sample+current_pos)
            labels.append(current_bioes)

        print("done")

        if(len(samples) == len(labels)):
            print("Length matches! Hurray!")
        else:
            print("Oops, something went wrong, check the dimension of samples and labels")

        return samples, labels



    def __init__(self, myDataset=None):

        self.mydataset = myDataset
        
        self.vocabulary, self.pos_tags, self.bioes_tags = self.load_dictionaries()

        self.all_samples = self.file_processing()

        self.samples, self.labels = self.file_parser(self.all_samples)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx) :


        return self.samples[idx], self.labels[idx]
