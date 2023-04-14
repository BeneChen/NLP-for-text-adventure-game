# {tag:[total_tag_count, dict{tag_next, count}]}
from typing import Tuple
from math import log

tolerance_metrix = []
class HidenMarkovClassifier():
    """_summary_
    parameter:
    tolerance_metrix:   this is the probabilistic tolerance that used when calculating the probability, for example
                        for metrix like:  {NNP:1}, the probability for a unseen word that is NNP are 1/<number of NNP appear>
                        the value should be either 1 or zero, if not specified, default 1
                        this metrix need fair consideration as this could effect the efficiency and accuracy of algorithm

    """
    ESTATE,ISTATE = 'ENDSTATE-endState-this-word-is-unique-12345' , 'INITSTATE-initState-this-word-is-unique-12345'
    EOBSER,IOBSER = 'EOBSERVATION-endObservation-this-word-is-unique-12345','IOBSERVATION-initialObservation-this-word-is-unique-12345'
    
    def __init__(self, tolerance_metrix:dict = {}, transitionEpsilon = 0, emissionEpsilon=0, transitionK = 2, emissionK = 2) -> None:
        """_summary_

        Parameters
        ----------
        tolerance_metrix : dict, optional
            this is the probabilistic tolerance that used when calculating the probability, for example
            for metrix like:  {NNP:1}, the probability for a unseen word that is NNP are 1/<number of NNP appear>
            the value should be either 1 or zero, if not specified, default 1
            this metrix need fair consideration as this could effect the efficiency and accuracy of algorithm
            , by default {}
        transitionEpsilon : int, optional
            transition epsilon used for smoothing,
            , by default 0
        emissionEpsilon : int, optional
            emission epsilon used for smoothing, by default 0
        transitionK : int, optional
            k used for transition smoothing , by default 2
        emissionK : int, optional
            K used for emission probability smoothing, by default 2
        """
        self.tolerance_metrix = tolerance_metrix
        self.emission_model = self.counting_model = self.transition_model = None
        self.transitionEpsilon = transitionEpsilon
        self.emissionEpsilon = emissionEpsilon
        self.transitionK = transitionK
        self.emissionK = emissionK
        self.dataSize = 0
        self.parameterCheck()

    def parameterCheck(self):
        if self.emissionK <= 0:
            raise ValueError('emissionK have to be higher than 0')
        if self.transitionK <= 0:
            raise ValueError('transitionK have to be higher than 0')
        if self.emissionEpsilon < 0:
            raise ValueError('emissionEpsilon can not be smaller than 0')
        if self.transitionEpsilon < 0:
            raise ValueError('transitonEpsilon can not be smaller than 0')
        if type(tolerance_metrix) != dict[str:int] and len(tolerance_metrix) != 0:
            raise TypeError('toleranceMetrx have to be dict with str in key and int in value')

    def fit(self,text, ):
        self.dataSize = len(text)
        self.transition_model, self.counting_model = self.generate_transition_model(text)
        self.emission_model = self.generate_emission_model(text, )

    def generate_emission_model(self,text, epoch = 1):
        datas:dict[str:(int, dict)] ={}
        # for i in epoch:
        for sent in text:
            for word, tag in sent:
                if tag in datas.keys():
                    tag_count, b = datas[tag][0], datas[tag][1]
                    if word in b:
                        b[word] +=1
                    else:
                        b[word] =1
                    datas[tag] = (tag_count+1, b)
                else:
                    datas[tag] = (1, {word:1})

        emissionMetrix = {}
        # for i in range(epoch):
        for tag, (tag_count, b) in datas.items():
            emissionMetrix[tag] = {}
            for word, count in b.items():
                emissionMetrix[tag][word] = (count+self.emissionEpsilon)/(tag_count +self.emissionK*self.emissionEpsilon)

        return emissionMetrix
    
    def generate_transition_model(self,text):
        counting_metrics = {}
        ISTATE, ESTATE = self.ISTATE ,self.ESTATE
        prev = self.ISTATE

        for sent in text:
            #count init state
            for word, tag in sent:
                if prev in counting_metrics.keys():
                    mini_metric = counting_metrics[prev]
                    mini_metric[0] +=1
                    mini_metric[1][tag] = mini_metric[1][tag] +1 if tag in mini_metric[1].keys() else 1
                else:
                    counting_metrics[prev] = [1, {tag:1}]
                prev = tag

            #get end state
            if prev in counting_metrics.keys():
                mini_metric = counting_metrics[prev]
                mini_metric[0] +=1
                mini_metric[1][ESTATE] = mini_metric[1][ESTATE] +1 if ESTATE in mini_metric[1].keys() else 1
            else:
                counting_metrics[prev] = [1, {ESTATE:1}]

            prev = ESTATE
            if prev in counting_metrics.keys():
                    mini_metric = counting_metrics[prev]
                    mini_metric[0] +=1
                    mini_metric[1][ISTATE] = mini_metric[1][ISTATE] +1 if ISTATE in mini_metric[1].keys() else 1
            else:
                counting_metrics[prev] = [1, {ISTATE:1}]
            prev = ISTATE

    
        ## resemble:
        ## {tag_from: {tag_to: possibility}}
        metric = {}
        for tag, mini_metric in counting_metrics.items():
            total_tag_count =mini_metric[0]
            new_mini_metric={}
            for tag_next, count in mini_metric[1].items():
                new_mini_metric[tag_next]= (count+self.transitionEpsilon)/(total_tag_count+self.transitionK*self.transitionEpsilon)
            metric[tag] = new_mini_metric

        return metric, {key:item[0] for key, item in counting_metrics.items()}
    
    def predict(self, text:list[str]):
        """predict tag list based on text input

        Parameters
        ----------
        text : list[str]
            list of text input to generate predicted data

        Returns
        -------
        list[str]
            list of tag associate with text
        """
        textGotPeriod = True
        if text[-1] != '.':
            textGotPeriod = False
            text.append('.')

        #init tag set
        virtebiMetrix = {tag:[] for tag in self.counting_model if tag != self.ISTATE and tag!= self.ESTATE}
        timeStamp = 1

        # first state
        invalidTags = []
        for tag in virtebiMetrix:
            a = self._getTransition(self.ISTATE, tag)
            b = self._getEmission(tag, text[0])
            if a ==0 or b ==0:
                virtebiMetrix[tag].append((0, self.ISTATE))
                invalidTags.append(tag)
            else:
                virtebiMetrix[tag].append((log(a*b), self.ISTATE)) 
        for invalidTag in invalidTags:
            del virtebiMetrix[invalidTag]
        
        # recursive 
        for word in text[1:]:
            invalidTags = []
            for tag in virtebiMetrix.keys():
                delta,psi = self._getArgMax(virtebiMetrix, word, tag, timeStamp)
                if delta == 0 or psi == None:
                    invalidTags.append(tag)
                virtebiMetrix[tag].append((delta, psi))
            for invalidTag in invalidTags:
                del virtebiMetrix[invalidTag]
            timeStamp+=1

        #final state
        delta, _maxTag = self._getArgMax(virtebiMetrix,self.EOBSER , self.ESTATE, timeStamp, True )
        finalResult = [_maxTag]

        #back track
        for i in range(timeStamp-1, 0, -1):
            _maxTag = virtebiMetrix[_maxTag][i][1]
            finalResult.insert(0, _maxTag)
        if textGotPeriod == True:
            return finalResult
        else:
            return finalResult[:-1]
    
    def _getArgMax(self, virtebiMetrix:dict[str:list], observation,tag, timeStamp, finalState = False)-> Tuple[float, str]:
        if timeStamp >= 0:
            b = self._getEmission(tag, observation) if finalState == False else 1
            resultList = []
            for _tag, _line in virtebiMetrix.items():
                a = self._getTransition(_tag, tag)
                if a != 0 and b != 0:
                    #find prev delta
                    prevDelta = _line[timeStamp-1][0]
                    #calcuate delta for this time stamp
                    tempDelta = prevDelta+log(b*a) 
                    if tempDelta != 0:
                        resultList.append((tempDelta, _tag))
            if len(resultList) != 0:
                return max(resultList, key = lambda a: a[0]) 
            else:
                return 0, None
        else:
            raise IndexError('time stamp not correct, time stamp should be higer than 0')
        

    def _getEmission(self, tag, word):
        """this return the emission probability and apply smoothing when probability is too small


        Parameters
        ----------
        tag : string
            tag in current time stamp
        word : string

        Returns
        -------
        float
            probabiliy in emission model

        Raises
        ------
        KeyError
            if tag doesn't appear in emission model
        """
        if self.emission_model != None:
            if tag in self.emission_model.keys():
                if word in self.emission_model[tag]:
                    return self.emission_model[tag][word]
                else:
                    return self.tolerance_metrix[tag]/self.counting_model[tag] if tag in self.tolerance_metrix else (self.counting_model[tag]+self.emissionEpsilon)/(self.dataSize*2000+ self.emissionK*self.emissionEpsilon)
            else:
                raise KeyError(f'{tag} no such tag in emission matrix, this error should not appear, if appear, please contact admission chi.b.chen@kcl.ac.uk, with error code 10000')
            

    def _getTransition(self, tagFrom, tagTo):
        if self.emission_model != None:
            if tagFrom in self.transition_model:
                if tagTo in self.transition_model[tagFrom]:
                    return self.transition_model[tagFrom][tagTo]
                else:
                    return self.transitionEpsilon/(self.dataSize*10 + self.transitionK*self.transitionEpsilon)
            else:
                raise KeyError('no such tag in transition matrix, this error should not appear, if appear, please contact admission chi.b.chen@kcl.ac.uk, with error code 10001')
                









