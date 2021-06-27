# Dictionary Construction:
# (1) Word look-up table W
# (2) POS look-up table P
# (3) BIOES look-up table T

import pickle

def read_file(input_file: str):

    current_file = input_file.split('\\')[-1]
    print(f"Currently processing the file: {current_file}")    
    file_content = open(input_file, mode='r')

    return file_content.readlines()

def find_tokens(input_dataset: str):

    all_tokens = []
    all_pos_tags = []
    all_ner_tags = []

    for line in input_dataset:
        if len(line) != 1:
            list_of_entities = line.split(' ')

        #print(list_of_entities)
        if len(list_of_entities) != 0:
            all_tokens.append(list_of_entities[0])
            all_pos_tags.append(list_of_entities[1])
            all_ner_tags.append(list_of_entities[2])
    
    return all_tokens, all_pos_tags, all_ner_tags

# Step 1: get file contents
train_dataset = read_file(r"data\raw\ConLL2003-bioes-train.txt")
test_dataset = read_file(r"data\raw\ConLL2003-bioes-test.txt")
valid_dataset = read_file(r"data\raw\ConLL2003-bioes-valid.txt")

# Step 2: get tokens, pos and bioes tags in all 3 files
v1, p1, t1 =  find_tokens(train_dataset)
v2, p2, t2 =  find_tokens(test_dataset)
v3, p3, t3 =  find_tokens(valid_dataset)
tokens_in_all_files = v1 + v2 + v3
pos_tags_in_all_files = p1 + p2 + p3
bioes_tags_in_all_files =  t1 + t2 + t3

# Step 3: Remove duplicates
all_unique_tokens = list(set(tokens_in_all_files))
all_unique_pos_tags = list(set(pos_tags_in_all_files))
all_unique_bioes_tags = list(set(bioes_tags_in_all_files))

# Step 4: Vocabulary, POS tags and BIOES tags dictionaries
vocabulary = {}
pos_tags = {}
BIOES_tags = {}
for idx, token in enumerate(all_unique_tokens):
    vocabulary[token] = idx
for idx, pos in enumerate(all_unique_pos_tags):
    pos_tags[pos] = idx
for idx, bioes in enumerate(all_unique_bioes_tags):
    BIOES_tags[bioes] = idx  

# Step 5: Export dictionaries
vocab_to_file = open("data/interim/ConLL2003_vocabulary.pkl","wb")
pickle.dump(vocabulary, vocab_to_file)
vocab_to_file.close()

tags_to_file = open("data/interim/ConLL2003_pos_tags.pkl","wb")
pickle.dump(pos_tags, tags_to_file)
tags_to_file.close()

bioes_to_file = open("data/interim/ConLL2003_BIOES_tags.pkl","wb")
pickle.dump(BIOES_tags, bioes_to_file)
bioes_to_file.close()

print("Done! All 3 dictionaries are now available for you...")