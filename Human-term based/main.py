import numpy as np
from human_terms_network import Human_Terms_Network
from dataset_load import *

import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit

def amazon_data():
    path = r"..\..\data\reviews_Amazon_Instant_Video_5.json.gz"
    X, y = extract_review_amazon(path, 'reviewText')

    y_label = np.asarray(y)
    neutral_indices = np.where(y_label == 3)[0]
    y_label[y_label<3] = 0
    y_label[y_label>3] = 1   

    X_discarded = np.delete(X,neutral_indices)
    y_discarded = np.delete(y_label, neutral_indices)

    del X
    del y_label

    # split
    print('Split train-test...')
    X_train_split, X_test_split, y_trn, y_test = train_test_split(X_discarded, y_discarded, test_size=0.33, random_state=42)

    # preprocessing
    print('preprocess the data...')
    X_train_corpus_update = update_corpus_contraction(X_train_split)
    X_test_corpus_update = update_corpus_contraction(X_test_split)


    # count vectorizer

    print('perform count vectorizer...')
    token = r"(?u)\b[\w\'/]+\b"
    cv = CountVectorizer(lowercase=True, max_df=1.0, min_df=100, binary=True, token_pattern=token)
    cv.set_params(ngram_range=(1,1))

    cv.fit(X_train_split)

    X_train = cv.transform(X_train_corpus_update)
    X_test = cv.transform(X_test_corpus_update)

    words = cv.get_feature_names()

    print('load word list...')
    word_list, connotation = load_unigrams('./amazon-video-unigrams.txt', X_train_corpus_update, y_train)

    print('Generate appearance agreement...')
    y_train_agreement, y_test_agreement = generate_appearance(X_train_corpus_update, X_test_corpus_update, 
                                                          word_list, connotation)

    return X_train, X_test, y_train_agreement, y_test_agreement, y_train, y_test

if __name__ == '__main__':

    
    X_train, X_test, y_train_agreement, y_test_agreement, y_train, y_test = amazon_data()



    print('build model...')
    htm = Human_Terms_Network(input_shape=X_train.shape[1], human_terms_shape=len(word_list))

    htm.set_data(X_train, X_test, y_train_agreement, y_test_agreement, y_train, y_test)

    htm.train(epochs=50, verbose=1)

    

