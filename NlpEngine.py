from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import conll2000
import nltk
from . import discourseRelationExtraction
import numpy as np
import string
import copy
from .WordEmbeddingModel import GloveModel

PORT_USING = 9000

#RUN python3 setup.py First in seperate terminal Before using This package

from .Chunker import BigramChunker

class Engine:

    def __init__(self) -> None:
        #initialise the corenlp
        self.parser  = CoreNLPParser(f'http://localhost:{PORT_USING}')
        chunkTypes = ['VP', 'NP', 'S', 'PRT', 'INTJ']
        trainSents = conll2000.chunked_sents('train.txt', chunk_types=chunkTypes)
        self.bigramChunker = BigramChunker(trainSents)
        self.relations = discourseRelationExtraction.DiscourseRelations()
        self.wordEmbeddingModel = [GloveModel()]

    def generateCommand(self, commands, sent):
        masked = self.masking(sent)
        originalSentence= nltk.tokenize.word_tokenize(sent)

        if masked.count('[mask]') > 3:
            connectiveRelations = self.varifyConnectionWordViterbi(originalSentence, masked)
        else:
            connectiveRelations= self.varifyConnectionWord(originalSentence, masked)
        
        clauses = Util.splitListSeperate(masked, '[mask]', False)
        clausesCopy = copy.deepcopy(clauses)
        j = 0

        for i, clause in enumerate(clausesCopy):
            if clause == '[mask]':
                clausesCopy[i] = connectiveRelations[j]
                j += 1
        validClauses = self.relations.getValidClausesImprove(clausesCopy)

        #evaluate each command and get the similarity
        #find backup 
        backup = []
        for command in commands:
            replaceableIndex = command.getAllWordIndex('REPLACE')
            for clause in validClauses:
                sent1 = clause
                possibleWord = self.getNonVerbs(' '.join(clause))
                for possibleCombination in Util.generateRelations(len(replaceableIndex), possibleWord):
                    sent2 = command.replaceCommand(replaceableIndex, possibleCombination, 'l')
                    similarity = self.getCosinSimilarity(originalSentence, sent2, 0) * self.getCosinSimilarity(clause, sent2, 0) 
                    backup.append((similarity, ' '.join(sent2)))
        
        backup.sort(key = lambda a:a[0], reverse=True)
        return backup
                
    def getVerbs(self, sent:list[str]):
        return [ word[0] for word in sent if 'VB' in word[1]]
    
    def getNonVerbs(self, sent:str):
        sent = nltk.pos_tag(nltk.tokenize.word_tokenize(sent))
        escape = ['TO']
        result = []

        for word, pos in sent:
            flag = True
            for e in escape:
                if e in pos:
                    flag = False
            if flag==True:
                result.append(word)

        return result
                      
    def getMaskedByChunk(self,sent):
        sent = nltk.pos_tag(nltk.word_tokenize(sent))
        return self.bigramChunker.masking(sent)
        
    def findAllSetOfClauses(self,tree:nltk.Tree):
        subtrees = Util.traverseAllAndStore(tree, 0)
        # subtrees = sorted(subtrees, key=lambda x: x[1])
        d = {}

        for subtree in subtrees:
            d.setdefault(subtree[1], []).append(subtree[0])

        for key,trees in d.items():
            _trees = [x for x in trees if (x.label() == 'VP' or 'S' in x.label() or x.label() == 'VB')]
            if len(_trees) > 1:
                return [x.flatten() for x in _trees]
            
        return None
    
    def getMaskingVP(self,sent):
        trees = self.findAllSetOfClauses(next(self.parser.raw_parse(sent)))
        result = nltk.tokenize.word_tokenize(sent)
        if trees:
            for tree in trees:
                result = Util.splitList(result, list(tree.flatten()), '', firstOccurence=True)
            connectionWords = Util.splitListSeperate(result, '')
            sentToken = nltk.tokenize.word_tokenize(sent)
            finalResult = sentToken
            for connectionWord in connectionWords:
                finalResult = Util.splitList(finalResult, connectionWord, '[mask]', False, firstOccurence=True)
            return finalResult
        return result
    
    def traverseRule(self,tree, parent,valid):
        if type(tree) != nltk.Tree:
            if not valid:
                return tree
            else:
                return '[mask]'
            
        for index, subtree in enumerate(tree):
            if hasattr(tree, "label") and tree.label():
                root = tree.label()
                if 'S'  in parent and tree.label() not in ['NP', 'VP', 'S', 'SBAR']:
                    tree[index] = self.traverseRule(subtree, root, True)
                elif 'PP' in parent and tree.label() not in ['S', 'SBAR']:
                    tree[index] = self.traverseRule(subtree, root, True)
                else:
                    if valid == True:
                        tree[index] = self.traverseRule(subtree, root, True)
                    else:
                        tree[index] = self.traverseRule(subtree, root, False)

        return tree
    
    def getMaskingByRule(self,sent):
        x = self.parser.raw_parse(sent)
        maskingSentence = list(self.traverseRule(next(x),'ROOT', False ).flatten())
        return maskingSentence

    
    def masking(self, sent):
        """mask the sentence, with several method

        Parameters
        ----------
        sent : string
            sentence for masking
        mode : str, optional
            mode to choose to select the masked word, by default MC('MOSTCOUNT'), other option: MV('MOJORITY VOTING')

        Returns
        -------
        list[str]
            masked sentence
        """
        connectionWordSelectionMethod = [self.getMaskingVP, self.getMaskingByRule, self.getMaskedByChunk]
        temp = []

        for CWSmethod in connectionWordSelectionMethod:
            temp.append(CWSmethod(sent))
        sentToken = nltk.tokenize.word_tokenize(sent)
        #apply majority voting
        maxOccur = max([len([y for y in x if y == '[mask]']) for x in zip(*temp)])
        if sum([x.count('[mask]') for x in temp]) != 0:
            tempMaskMaxcount = ['[mask]' if x.count('[mask]') == maxOccur and x[len(x)-1] not in string.punctuation else x[len(x)-1] for x in zip(*temp, sentToken)]
            tempMaskMajorityVoting = ['[mask]' if x.count('[mask]') > len(connectionWordSelectionMethod)/2 and x[len(x)-1] not in string.punctuation else x[len(x)-1] for x in zip(*temp, sentToken)]
            finalMaskedSentence = tempMaskMaxcount if tempMaskMaxcount.count('[mask]') > tempMaskMajorityVoting.count('[mask]') else tempMaskMajorityVoting
            groupedMaskedSentece = []
            last = None
            for elem in finalMaskedSentence:    
                if elem != last and elem == '[mask]':
                    groupedMaskedSentece.append(elem)
                    last = elem
                elif elem != '[mask]':
                    groupedMaskedSentece.append(elem)
                    last = elem
        else:
            return sentToken

        return groupedMaskedSentece
    
    def varifyConnectionWordViterbi(self, originalSentence, maskedSentence:list):
        maskedSentence = copy.deepcopy(maskedSentence)
        relations = self.relations
        relationNames = relations.getAllRelationsName()
        maskedIndex = np.where(np.asanyarray(maskedSentence) == '[mask]')[0]
        viterbiMatrix = {relationName: [(None, 0) for i in range(len(maskedIndex))] for relationName in relationNames}
        relationKeyword = relations.getAllRelationsWithNumKeyword(1)

        bestRelation = None
        for relation,words in relations.allRelations.items():
            similaritys = []
            bestSimilarity = 0
            
            for word in relationKeyword[relation]:
                maskedSentence[0] = word
                similaritys.append(self.getCosinSimilarity(' '.join(maskedSentence).split(), originalSentence, 0))
            similarity = np.average(similaritys)
            if similarity > bestSimilarity:
                bestSimilarity = similarity
                bestRelation = relation
            viterbiMatrix[relation][0] = (bestRelation, bestSimilarity)
        maskedSentence[0] = '[mask]'

        for i in range(1,len(maskedIndex)):
            for key in viterbiMatrix.keys():
                bestSimilarity = 0
                bestKey = None
                #find the best similarity
                for relation,words in relations.allRelations.items():
                    for x in range(i):
                        index = maskedIndex[x]
                        tempRelation = viterbiMatrix[relation][x][0]
                        maskedSentence[index] = relationKeyword[tempRelation][0]
                    similaritys = []
                    # calculate single similarity
                    for word in relationKeyword[relation]:
                        maskedSentence[maskedIndex[i]] = word
                        similaritys.append(self.getCosinSimilarity(' '.join(maskedSentence).split(), originalSentence, 0))
                    similarity = np.average(similaritys) * viterbiMatrix[relation][i-1][1]
                    if similarity > bestSimilarity:
                        bestSimilarity = similarity
                        bestKey = relation
                
                    maskedSentence[maskedIndex[i]] = '[mask]'
                viterbiMatrix[key][i] = (bestKey, bestSimilarity)
            
        result = []
        maximumRelation = max([values[len(maskedIndex)-1] for x, values in viterbiMatrix.items()] , key= lambda a: a[1])
        result.insert(0, maximumRelation)

        #back tracking
        for i in range(len(maskedIndex)-2, -1, -1):
            maximumRelation = viterbiMatrix[maximumRelation[0]][i][0]
            result.insert(0, maximumRelation[0])
        
        return result

    def varifyConnectionWord(self, originalSentence, maskedSentence:list):
        maskedSentence = copy.deepcopy(maskedSentence)
        relations = self.relations
        maskedIndex = np.where(np.asanyarray(maskedSentence) == '[mask]')[0]
        bestCombination = [None for i in maskedIndex]
        bestSimilarity = 0

        for relationCombinations in Util.generateRelations(len(maskedIndex), relations.getAllRelationsWithNumKeyword(1).keys()):
            similaritys = []
            for keywordCombination in Util.generateRelationsInTime(len(maskedIndex), [relations.getAllRelationsWithNumKeyword(1)[relation] for relation in relationCombinations]):
                for keyword, x in zip(keywordCombination, maskedIndex):
                    maskedSentence[x] = keyword
                similaritys.append(self.getCosinSimilarity(' '.join(maskedSentence).split(), originalSentence,0))
            if np.average(similaritys) > bestSimilarity:
                bestSimilarity = np.average(similaritys)
                bestCombination = relationCombinations

        return bestCombination
    
    def getCosinSimilarity(self, listOfWord1, listOfWord2, N):
        try:
            return self.wordEmbeddingModel[N].getCosinSimilarity(listOfWord1, listOfWord2)
        except IndexError:
            raise(ValueError("model index doesn't exist"))
    

class Util:
    @staticmethod
    def generateRelations(n, option):
        if n == 0:
            return [[]]
        else:
            result = []
            for s in Util.generateRelations(n-1, option):
                for x in option:
                    result.append(s+[x])
            return result

    @staticmethod
    def generateRelationsInTime(n, option):
        if n == 0:
            return [[]]
        else:
            result = []
            for s in Util.generateRelationsInTime(n-1, option):
                option[n-1]
                for x in option[n-1]:
                    result.append(s+[x])
            return result
    
    @staticmethod
    def groupWordByConsecutive(sent, wordList):
        # initialize variables
        groups = []
        flag = False
        group = []
        # iterate over the sentence
        for word in sent:
            if word in wordList:
                # add word to current group
                group.append(word)
                flag = True
            else:
                if flag:
                    # add completed group to list of groups
                    groups.append(group)
                    group = []
                    flag = False

        # add last group to list of groups
        if group:
            groups.append(group)

        return groups
    
    @staticmethod
    def traverseAllAndStore(tree:nltk.Tree, layer):
        #tree not reach end
        _layer = int(layer)
        if tree.leaves():
            result = []
            # go through each tree and apply a traverse
            for index, subtree in enumerate(tree):
                if type(subtree) == nltk.Tree:
                    result.append((subtree, layer))
                    result += Util.traverseAllAndStore(subtree, _layer+1)
            return result
    
    @staticmethod
    def splitList(lst, pattern, replaceOption = None, abbreviate = True, firstOccurence = False):
        """splite list with another sublist, for example, splitList([1,2,3,4,5], [3,4])
        [1,2,5]

        Parameters
        ----------
        lst : list[T]
            list to be split
        pattern : list[T]
            pattern to split the list
        replaceOption : T, optional 
            replace pattern if spesified, by default None
        abbreviate : bool, optional
            if true, multiple element will be replaced with one replace option, else it will replace with the  , by default True
        firstOccurence : bool, optional
            if true,only first occurance considered, by default False

        Returns
        -------
        _type_
            _description_
        """
        result = []
        current = []

        #number of element that list are matched to pattern
        count = 0
        occurence = 0
        
        for index, item in enumerate(lst):
            if firstOccurence != True or occurence != 1:
                #currently, pattern are not fully satisfied
                if count != len(pattern):    
                    if item == pattern[count]:
                        count += 1
                        current.append(item)
                    else:
                        count = 0
                        for z in current:
                            result.append(z)
                        current = []
                        result.append(item)
                else:
                    occurence +=1
                    count = 0
                    if abbreviate == False and replaceOption and current:
                        if replaceOption == None:
                            result.append(current)  
                        else:
                            result += len(current) * [replaceOption]
                    else:
                        result.append(current) if replaceOption == None else result.append(replaceOption)
                    current = []
                    result.append(item)
            else:
                result.append(item)
                
        if len(current) == len(pattern):
            if abbreviate == False and replaceOption and current:
                if replaceOption == None:
                    result.append(current)  
                else:
                    result += len(current) * [replaceOption]
            else:
                result.append(current) if replaceOption == None else result.append(replaceOption)
            current = []
        else:
            for z in current:
                result.append(z)
            current = []

        return result
    
    @staticmethod
    def splitListSeperate(lst, element, replace = True):
        """split list with single element

        Parameters
        ----------
        lst : list[T]
            list that wait to split
        element : T
            element used to split the list

        Returns
        -------
        list[list[T]]
            seperated list
        """
        split_list = []
        temp_list = []

        for elem in lst:
            if elem == element:
                split_list.append(temp_list)
                if replace == False:
                    split_list.append(elem)
                temp_list = []
            else:
                temp_list.append(elem)

        split_list.append(temp_list)
        split_list = [x for x in split_list if x]
        return split_list
