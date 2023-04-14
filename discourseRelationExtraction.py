from gensim.models.word2vec import Word2Vec
import numpy as np

class Clause():
    def __init__(self, pronoun, verb, noun) -> None:
        self.pronoun = pronoun
        self.verb = verb
        self.noun = noun
    def getPronoun(self):return self.pronoun
    def getVerb(self):return self.verb
    def getNone(self):return self.noun

class Relation():
    def __init__(self, keywords):
        self.unaryKey = [keyword  for keyword in keywords if not np.shape(keyword)]
        self.multiKey = [keyword for keyword in keywords if np.shape(keyword)]
        
    def possibleActualClauses(self):...

class ConjunctionRelation():
    def __init__(self, keyword):self.keyword = keyword

class DiscourseRelations():
    """class define all relations and related method
    """
    allRelations = {

    'Y':['but', 'however'],
    'XandY':[ 'also', 'similarly'],
    'X':['rather than','instead of', 'not', 'particularly'],
    '-X-Y':['nor', 'neither']
    }

    def getRelation(self, data):
        return data[0]
    
    def getAllRelationsName(self):
        return self.allRelations.keys()
    
    def getAllRelationsWithNumKeyword(self, n):
        """return relations that have n
                    
        Parameters
        ----------
        n : int
            keyword that have size n

        Returns
        -------
        dict
            relation as key and keyword of size n in value
        """
        # result = {key: [x for x in lst if len(x) == n and type(x) != str] for key, lst in self.allRelations.items()}
        result = {}
        for key,value in self.allRelations.items():
            lst = []
            for x in value:
                if np.shape(x):
                    if np.shape(x)[0] == n:
                        lst.append(x)
                else:
                    if n==1:
                        lst.append(x)
            result[key] = lst         
        return result
        # return {key: item for key, item in result.items() if len(item)}
    
    def __getitem__(self, arg):
        return self.allRelations[arg]

    def selectSuitableModel(self, listOfWord):
        modelCount = []
        for model in self.models:
            modelCount.append(np.count_nonzero([word in model for word in listOfWord]))
        return np.argmax(modelCount)

    def masking(self, tag, wordTagPairs):
        masks = []
        for index, _word, _tag in enumerate(wordTagPairs):
            if _tag in tag:
                masks.append((index, _tag))
                wordTagPairs[index][0] = '[MASK]'
        return wordTagPairs, masks, tag, 
        
    

    def identifiedRelation(self, wordTagPairs):
        # doing CC
        wordTagPairs, masks, tag = self.masking(self, ["CC"], wordTagPairs)
    
    def getValidClauses(self, relations, clauses):
        validClauses = []
        for relation, clause in zip(relations, clauses):
            if relation == 'Y':
                validClauses.append(clause)
            elif relation == 'X':
                pass
        return validClauses
    
    def getValidClausesImprove(self, sent):
        """implementing a vote factor for all relation and by voting the clause,


        Parameters
        ----------
        sent : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if len(sent) < 2:
            return sent
        validClauses = []
        recorder = []
        vote = np.zeros(len(sent))
        for i, word in enumerate(sent):
            if type(word) != list:
                if word in self.allRelations.keys():
                    recorder.append(i)
                    if word == 'Y':
                        voteValue = (-1, 1)
                    elif word == 'X':
                        voteValue = (1, -1)
                    elif word == 'XandY':
                        voteValue = (1, 1)
                    elif word == '-X-Y':
                        voteValue = (1, 1)

                    if i-1 < 0:
                        vote[i+1] += voteValue[1]
                    elif i+1 > (len(sent) -1 ):
                        vote[i-1] += voteValue[0]
                    else:
                        vote[i-1] += voteValue[0]
                        vote[i+1] += voteValue[1]
        
        for j, clause in enumerate(sent):
            if j not in recorder and vote[j] >= 0:
                validClauses.append(clause)
        return validClauses
                



    

