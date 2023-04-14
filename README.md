#   NLP for Text Adventure Game
This application uses natural language processing (NLP) techniques to extract commands from text input for a text adventure game. Several setup steps are required before the application can be used.

####    Setup
* Note that this application currently only supports macOS.

1.  Download the GloVe model from https://nlp.stanford.edu/data/glove.twitter.27B.zip. Only the glove.twitter.27B.25d.txt file is needed, which has a maximum size of 300MB. The path should be glove.twitter.27B/glove.twitter.27B.25d.txt. However, other models can also be used by setting the appropriate parameter.

2.  Note: If you have a different file name after unzipping glove.twitter.27B.zip, set the GLOVE_PATH parameter in WordEmbeddingModel.py.

3.  Download the grammar parser from https://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip, unzip it, and place it under the Parser folder. The path to the package should be Parser/stanford-corenlp-4.5.4. 

4.  In a terminal, run the following command:

    cd Parser/stanford-corenlp-4.5.4 && java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 5000

5.  Run setup.py.

####    Usage
A portal on port 9000 will be used to load the local server. Always check if the parser server is running. If it is not running, repeat step 3 above.

To initialize the command extractor, use the commandExtractor() function. It takes in commands formalized in a text file or in a list that looks like this:

[
    'go <direction>',
    'pick <item>',
    'talk <character>',
    'enter the battle with <monster> [vifi,ligi,act]]'
]

Where a grammar needs to be followed strictly, words in <> are replaceable words, and square brackets after the <> state the backup option of replaceable words. This is optional, but adding backup options could help formalise the command. no space should stay between word inside bracket and bracket

The only method that a developer will use is commandGenerator in commandExtractor. This method returns a command with the highest confidence about the intention of the input, which satisfies the validation function passed in. The validation function should validate the command to tell the commandGenerator what command it is currently confident with and should return. The validation function is optional, and if not specified, all generated commands will be returned.

There are two parameters that can be adjusted: GloVe path and the validation function. A better GloVe model can be used as long as the path is correct.

Updates
Future updates will be available on https://github.com/BeneChen/NLP-for-text-adventure-game.