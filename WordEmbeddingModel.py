from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

GLOVE_PATH = '''glove.twitter.27B/glove.twitter.27B.25d.txt'''

class Model():
    def __init__(self) -> None:...
    def __getitem__(self, arg):...

class GloveModel(Model):
    def __init__(self, filepath = None) -> None:
        model = {}
        path = filepath if filepath != None else GLOVE_PATH
        curpath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curpath, path), 'r') as f:
            # for line, i in zip(f, tqdm(range(len(f.readlines())))):
            for line in f:
                values = line.split()
                word = values[0]
                model[word] = np.asarray(values[1:], "float32")
        self.model = model

    def __getitem__(self, arg):
        return self.model[arg]
    def __contains__(self, arg):
        return arg in self.model
    def getCosinSimilarity(self, listOfWord1, listOfWord2):
        sent1_embed = [self.model[x] for x in listOfWord1 if x in self.model]
        sent2_embed = [self.model[x] for x in listOfWord2 if x in self.model]
        if len(sent2_embed) == 0 or len(sent1_embed) == 0:
            return 0
        ave_embedding1 = np.average(sent1_embed, axis=0)
        ave_embedding2 = np.average(sent2_embed, axis=0)
        return cosine_similarity(ave_embedding1.reshape(1, -1), ave_embedding2.reshape(1, -1))[0][0]

# initialise gensim word2vec first before using this model
# class Word2VecModel(Model):
#     def __init__(self, name) -> None:
#         self.model = Word2Vec.load(f'word2VecModel/{name}.model')
#     def __getitem__(self, arg):
#         return self.model.wv[arg]
#     def __contains__(self, arg):
#         return arg in self.model.wv
#     def getCosinSimilarity(self, listOfWord1, listOfWord2):
#         sent1_embed = self.model.wv[[x for x in listOfWord1 if x in self.model.wv]]
#         sent2_embed = self.model.wv[[x for x in listOfWord2 if x in self.model.wv]]
#         if len(sent2_embed) == 0 or len(sent1_embed) == 0:
#             return 0
#         ave_embedding1 = np.average(sent1_embed, axis=0)
#         ave_embedding2 = np.average(sent2_embed, axis=0)
#         return cosine_similarity(ave_embedding1.reshape(1, -1), ave_embedding2.reshape(1, -1))[0][0]
