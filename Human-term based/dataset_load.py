import gzip
import pickle
import re
import numpy as np

# open pickle file
# usually for pickled IMDB dataset
def open_pickle(path):
    with open(path, 'rb') as f:
        X = pickle.load(f)
    return X

'''
This bundle is for loading the AMAZON DATA
'''
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
        
def extract_review_amazon(path, key):
    corpus = []
    y = []
    text = parse(path)
    for l in text:
        corpus.append(l[key])
        y.append(l['overall'])
    return corpus, y


'''
IMDB
For agreement data extraction purpose
'''
def load_unigrams(path, X, y):
    word_list = []
    connotation = {}
    
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            word_list.append(line.strip())
            
    for word in word_list:
        pos_count = 0
        neg_count = 0
        for i, doc in enumerate(X):
            if word in doc.lower():
                if (y[i] == 1):
                    pos_count += 1
                else:
                    neg_count += 1
                    
        if pos_count > neg_count:
            connotation[word] = 1
        else:
            connotation[word] = 0
    
    return word_list, connotation

def generate_appearance(X_train_corpus, X_test_corpus, word_list, connotation):
    y_train_agreement = []
    for i in range(len(X_train_corpus)):
        doc_agreement = []
        for word in word_list:
            if word in X_train_corpus[i]:
                if connotation[word] == 1:
                    doc_agreement.append(1)
                else:
                    doc_agreement.append(-1)
            else:
                doc_agreement.append(0)
        y_train_agreement.append(doc_agreement)
        
    y_test_agreement = []
    for i in range(len(X_test_corpus)):
        doc_agreement = []
        for word in word_list:
            if word in X_test_corpus[i]:
                if connotation[word] == 1:
                    doc_agreement.append(1)
                else:
                    doc_agreement.append(-1)
            else:
                doc_agreement.append(0)
        y_test_agreement.append(doc_agreement)
        
    return np.array(y_train_agreement), np.array(y_test_agreement)


'''
Preprocessing
'''

def load_list(filename, split_delimiter):
    vocabulary = []
    with open(filename, 'r') as f:
        for l in f:
            vocabulary.append(l.strip().split(split_delimiter))
    return np.asarray(vocabulary)

def cleanhtml(text):
    cleanr = re.compile('<.*?>')
    cleantag = re.sub(cleanr, '', text)
    cleantext = cleantag.replace('br', '')
    return cleantext

def replace_contraction(corpus, cont_list):
    for i in range(0, cont_list.shape[0]):
        corpus = corpus.lower().replace(cont_list[i,0], cont_list[i,1])
    return corpus

def update_corpus_contraction(X_corpus):
    cont_list = load_list("contraction_list.txt", ',')
    print(cont_list.shape)
    print('corpus update start')
    for i in range(0,len(X_corpus)):
        X_corpus[i] = cleanhtml(X_corpus[i])
        X_corpus[i] = replace_contraction(X_corpus[i], cont_list)
    print('corpus update end')
    print()
    return X_corpus