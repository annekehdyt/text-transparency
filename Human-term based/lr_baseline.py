from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import numpy as np

class LRBaseline():
    def __init__(self, X_train_corpus, X_test_corpus, y_train, y_test, human_terms=None, min_df=100,
                imdb=True, amazon=False, random_state=42):
        # Data
        self.X_train_corpus = X_train_corpus
        self.X_test_corpus = X_test_corpus
        self.y_train = y_train
        self.y_test = y_test
        self.human_terms = human_terms
        
        #count vectorizer purpose
        #only for baseline
        self.min_df = min_df
        self.random_state = random_state

        #
        self.imdb = imdb
        self.amazon = amazon
        self.token = r"(?u)\b[\w\'/]+\b"

    def baseline(self, penalty='l2'):
        self.baseline_cv = CountVectorizer(min_df=self.min_df, binary=True, lowercase=True, token_pattern=self.token)

        X_train = self.baseline_cv.fit_transform(self.X_train_corpus)
        X_test = self.baseline_cv.transform(self.X_test_corpus)

        self.clf_base = LogisticRegression(random_state=self.random_state, penalty=penalty)
        self.clf_base.fit(X_train, self.y_train)

        p = self.clf_base.predict_proba(X_test)[:,1]

        return self.clf_base.score(X_train, self.y_train), self.clf_base.score(X_test, self.y_test), log_loss(self. y_test, p)


    def human_terms_baseline(self, penalty='l2'):
        self.human_terms_cv = CountVectorizer(vocabulary=self.human_terms)

        X_train = self.human_terms_cv.fit_transform(self.X_train_corpus)
        X_test = self.human_terms_cv.transform(self.X_test_corpus)


        indices = np.where(np.sum(X_test, axis=1) != 0)[0]
        r_rate = (X_test.shape[0] - len(indices))/X_test.shape[0]

        self.clf_human = LogisticRegression(random_state=self.random_state, penalty=penalty)
        self.clf_human.fit(X_train, self.y_train)

        p = self.clf_human.predict_proba(X_test[indices])[:,1]

        return self.clf_human.score(X_train, self.y_train), self.clf_human.score(X_test[indices], self.y_test[indices]), log_loss(self.y_test[indices], p), r_rate
