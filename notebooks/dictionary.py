import nltk
from nltk.corpus import gutenberg
from nltk.data import load
import pickle

#download corpus: Jane Austen's Sense and Sensibility hehe
sense = nltk.corpus.gutenberg.words('austen-sense.txt')

#remove duplicates
unique_words_in_corpus = list(set(sense))

# (1) word dictionary construction
print("Building word look-up table W:")
vocabulary = {}

for idx,word in enumerate(unique_words_in_corpus):
    word = str(word).lower()
    vocabulary[word] = idx

vocabulary['PAD'] = len(vocabulary)+1

print("done!")
# print("sample from vocab look-up dict:")
# print(dict(list(vocabulary.items())[:20]))


# (2) POS tags dictionary
print("Building POS look-up table P:")
tagdict = load('help/tagsets/upenn_tagset.pickle')


pos_tags = {}
for idx, tag in enumerate(list(tagdict.keys())):
    pos_tags[tag] = idx

pos_tags['PAD'] = len(pos_tags)+1
print("done!")
# print("\nsample from tags look-up dict:")
# print(dict(list(pos_tags.items())[:20]))

vocab_to_file = open("C:/Users/rahin/projects/paper-draft-03/data/processed/vocabulary.pkl","wb")
pickle.dump(vocabulary, vocab_to_file)
vocab_to_file.close()

tags_to_file = open("C:/Users/rahin/projects/paper-draft-03/data/processed/pos_tags.pkl","wb")
pickle.dump(pos_tags, tags_to_file)
tags_to_file.close()

print("Success! Word and POS tags dictionary successfully constructed and exported!")