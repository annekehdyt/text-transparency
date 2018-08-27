from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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

    def baseline(self, penalty='l2'):
        self.baseline_cv = CountVectorizer(min_df=self.min_df)

        X_train = self.baseline_cv.fit_transform(self.X_train_corpus)
        X_test = self.baseline_cv.transform(self.X_test_corpus)

        clf = LogisticRegression(random_state=self.random_state, penalty=penalty)
        clf.fit(X_train, self.y_train)

        return clf.score(X_train, self.y_train), clf.score(X_test, self.y_test)


    def human_terms_baseline(self, penalty='l2'):
        self.human_terms_cv = CountVectorizer(vocabulary=self.human_terms)

        X_train = self.human_terms_cv.fit_transform(self.X_train_corpus)
        X_test = self.human_terms_cv.transform(self.X_test_corpus)

        clf = LogisticRegression(random_state=self.random_state, penalty=penalty)
        clf.fit(X_train, self.y_train)

        return clf.score(X_train, self.y_train), clf.score(X_test, self.y_test)
