import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# download packages
nltk.download('punkt')

with open('./data/author-quote.txt') as f:
    lines = f.readlines()

lines = [
    x.split('\t')[1].strip()
    for x in lines if x.strip()
]

''.translate(None, nltk)
sentences = [ sent_tokenize(x) for x in lines ]
words = [ word_tokenize(x) for x in lines ]