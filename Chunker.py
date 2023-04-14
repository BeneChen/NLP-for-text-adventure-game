import nltk
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data) 
    
    def masking(self, sentence):
        """mask sentence based on chunker

        Parameters
        ----------
        sentence : string
            sentence waiting for masked

        Returns
        -------
        list[str]
            masked sentence
        """
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        maskedSentence = ['[mask]' if chunktag == 'O' else word for ((word,pos),chunktag) in zip(sentence, chunktags)]
        return maskedSentence
