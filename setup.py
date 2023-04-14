from sys import platform
import os

if platform == "linux" or platform == "linux2":
    # linux
    pass
elif platform == "darwin":
    # OS X
    os.system('cd Parser/stanford-parser-4.2.0 && java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -threads 4 -port 9000 -timeout 5000')
elif platform == "win32":
    # Windows...
    pass

'''
below solution come from 
https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
'''
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("treebank")
nltk.download("conll2000")
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')