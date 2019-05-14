import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

class_names = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
    ]

train = pd.read_csv('./data/train.csv').fillna(' ')
test = pd.read_csv('./data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']

tt_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    max_features=50000)

word_vectorizer.fit(tt_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=500000)

char_vectorizer.fit(tt_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
for c in class_names:
    train_target = train[c]
    classifier = LogisticRegression(C=0.01, solver='sag')

    cv_score = np.mean(cross_val_score(classifier,
        train_features, train_target,
        cv=3, scoring='roc_auc'))

    scores.append(cv_score)
    print('Cross Val score for {} is {}'.format(c, cv_score))

    classifier.fit(train_features, train_target)

print('Total Cross Val score is {}'.format(np.mean(scores)))