
# Removing URLs:
df['text'] = df['text'].apply(lambda x:re.sub(r'http\S+', '', str(x)))
df['text'] = df['text'].apply(lambda x:re.sub(r'nRead|nread\S+', '', str(x)))

# Removing all special (non word) characters:
df['text']= df['text'].apply(lambda x:re.sub(r'\W', ' ', str(x)))

# Removing characters containing digits:
df['text'] = df['text'].apply(lambda x:re.sub(r'\w*\d\w*', '', str(x)))
df['text'] = df['text'].apply(lambda x:re.sub(r'xef', '', str(x)))

# Removing all single characters (like 's' in apostrophe):
df['text'] = df['text'].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x)))

# Removing single characters from the start:
df['text'] = df['text'].apply(lambda x:re.sub(r'\^[a-zA-Z]\s+', ' ', str(x)))

# Substituting multiple spaces with single space:
df['text'] = df['text'].apply(lambda x:re.sub(r'\s+', ' ', str(x), flags=re.I))

# Removing 'b' from the beginning of the tweets:
df['text'] = df['text'].apply(lambda x:re.sub(r'^b\s+', '', str(x)))

# Removing single letter words:
df['text'] = df['text'].apply(lambda x:re.sub(r'(?:^| )\w(?:$| )', '', str(x)).strip())

# Lowering all letters:
df['text'] = df['text'].apply(lambda x:x.lower())

# Tokenizing:
df['tokenized_text'] = df['text'].apply(word_tokenize)

# Removing stopwords:
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [item for item in x if item not in stopwords.words("English")])

# lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)
