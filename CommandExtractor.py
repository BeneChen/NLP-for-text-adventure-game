import re
from .HiddenMarkovModelPretrain import HidenMarkovClassifier as PosTagger
from nltk.corpus import treebank
import numpy as np
from . import NlpEngine

class CommandExtractor:
    """
    command extractor provide easy used API for general command extractor
    """
    def __init__(self, commandSet:list[str]) -> None:
        self.OpenSymbolSet = ['<', '[']
        self.CloseSymbolSet = ['>', ']']
        self.commandLines = self.generateCommandLine(commandSet)

        transitionEpsilon =160
        emissionEpsilon = 600
        dataTrained = 3000
        transitionK = 2
        emissionK = 2
        text = treebank.tagged_sents()[:dataTrained]
        toleranceMetrix = {}
        self.posTagger = PosTagger(tolerance_metrix=toleranceMetrix, 
                                       emissionEpsilon=emissionEpsilon,
                                       emissionK=emissionK,
                                       transitionEpsilon=transitionEpsilon,
                                       transitionK= transitionK)
        self.posTagger.fit(text)
        for commandline in self.commandLines:
            partOfSpeechs = self.posTagger.predict(commandline.getText())
            for pos, command in zip(partOfSpeechs, commandline):
                command.setPos(pos)
        self.nlpEngine = NlpEngine.Engine()

    def commandGenerator(self,sent,validationFunction = None):
        """generate a likely command, based on commandlines currently have
        if a validation function spesified, next command generate will be valid, 
        other wise, command generate will be a command that follow the command patterns

        Parameters
        ----------
        sent : string
            sent for input
        validationFunction : function, optional
            a validation function to validate the validation of command,should return boolean, by default None
        """
        backup = self.nlpEngine.generateCommand(self.commandLines, sent)
        for option in backup:
            option = option[1]
            if validationFunction != None:
                if validationFunction(option):
                    yield(option)
            else:
                yield(option)

    def resetCommandSet(self, commandSet:list[str]):
        """update commandset used.

        Parameters
        ----------
        commandSet : list[str]
            new command set
        """
        self.commandLines = self.generateCommandLine(commandSet)
        for commandline in self.commandLines:
            partOfSpeechs = self.posTagger.predict(commandline.getText())
            for pos, command in zip(partOfSpeechs, commandline):
                command.setPos(pos)

    def generateCommandLine(self,commandSet:list[str]):
        commandLines = []
        if self.verifyGrammer(commandSet):
            
            replaceRegex = re.compile(r'<[a-zA-Z]+>')
            backupRegex = re.compile(r'\[[a-zA-Z,]+\]')
            for lineIndex, command in enumerate(commandSet):
                _command = Command()
                for word in command.split():
                    if replaceRegex.match(word):
                        _command.addReplaceWord(word[1:-1].strip())
                    elif backupRegex.match(word):
                        _command.addBackup(word[1:-1].split(','))
                    else:
                        _command.addNormalWord(word)
                commandLines.append(_command)
            return commandLines
        else:
            return SyntaxError('symbol have to appear in pairs')
    
    def verifyGrammer(self, commandSet):
        counter = 0
        for command in commandSet:
            for x in command:
                if x in self.OpenSymbolSet:
                    counter += 1
                elif x in self.CloseSymbolSet:
                    counter -= 1
        return True if counter == 0 else False

    def __str__(self) -> str:
        return str([str(command) for command in self.commandLines])
    
    def parse(self, sentence:str):
        commands = np.asanyarray(sentence.split())
        pos = np.asanyarray(self.posTagger.predict(sentence.split()))

class CommandSet(list):...

class Command:
    def __init__(self) -> None:
        self.words = []

    def addNormalWord(self, word):
        """add normal word 

        Parameters
        ----------
        word : string
            a word of command
        """
        self.words.append(NormalWord(word))

    def addReplaceWord(self, replaceTerm):
        """add a replace word, by create replaceable object

        Parameters
        ----------
        replaceTerm : string
        """
        self.words.append(ReplaceableWord(replaceTerm))

    def addBackup(self, backUp:list):
        """add backup option if previous command is a replaceable word

        Parameters
        ----------
        backUp : list
            list of backup option that can used to replacement
        """
        if hasattr(self.words[-1], 'addBackUp'):
            self.words[-1].addBackUp(backUp)

    def __str__(self) -> str:
        return str([str(c) for c in self.words])
    
    def __getitem__(self, arg):
        return self.words[arg]
    
    def getType(self):
        return [c._type for c in self.words]

    def getText(self):
        return [command.content for command in self.words]
    
    def replaceCommand(self, listOfIndex, listOfWord, mode = 's'):
        """repalce command with list of index and list of word that will be used to repalce
        word in these index
        for example
        list of index = [1,3]
        list of wrod  = ['I', 'am']
        "I" will be in position 1, "am" should be in postion 3

        Parameters
        ----------
        listOfIndex : list[int]
            position need replace
        listOfWord : list[str]
            item replace in position
        mode : str, optional
            form of return, return item will be string if 's' set, 'l' will be a list. by default 's',

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            when two list have different length
        
        """
        if len(listOfIndex) != len(listOfWord):
            raise ValueError('length of index need to be equal to length of words')
    
        _words = [word.getContent() for word in self.words]
        for index, word in zip(listOfIndex, listOfWord):
            _words[index] = word
            
        if mode == 's':
            return ' '.join(_words)
        
        elif mode == 'l':
            return _words
        else:
            raise ValueError('mode incorrect')
            
    def getAllWord(self, wordType):
        """get all word with word type passed in

        Parameters
        ----------
        wordType : String
            word type

        Returns
        -------
        list[(int, string)]
            series of word that have spesific type, and index associated
        """
        return [(index,word) for index, word in enumerate(self.words) if word._type == wordType]
    
    def getAllWordIndex(self, wordType):
        """same as above but only return index

        Parameters
        ----------
        wordType : string
            word type

        Returns
        -------
        list[int]
            list of index
        """
        return [index for index, word in enumerate(self.words) if word._type == wordType]

class Word:
    def __init__(self, content:str) -> None:
        self.content = content

    def __str__(self) -> str:
        return self.content
    
    def setPos(self, pos):
        self._pos = pos
    
    def getPos(self):
        return self._pos
    
    def getContent(self):
        return self.content

class NormalWord(Word):
    _type = 'COMMAND'
    def __init__(self, content) -> None:
        super().__init__(content)
    
class ReplaceableWord(Word):
    _type = "REPLACE"
    def __init__(self, content) -> None:
        super().__init__(content)
        self.backUp = []

    def __str__(self):
        return str(f'replace: {self.content}, backup: {self.backUp}')
    
    def addBackUp(self, backUpOption):
        self.backUp += backUpOption
    
    def hasBackUp(self):
        return True if self.backUp else False
    
    def getBackUp(self):
        return self.backUp