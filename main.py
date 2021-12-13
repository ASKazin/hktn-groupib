# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import mailbox
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from scipy.sparse import hstack
import scipy

trainfile = '/content/drive/MyDrive/groupib/train.mbox'
testfile = '/content/drive/MyDrive/groupib/test.mbox'

"""## Парсинг данных"""


def read_mbox(mb_file):
    mb = mailbox.mbox(mb_file)
    mbox_dict = {}

    for i, mail in enumerate(mb):
        mbox_dict[i] = {}
        for header in mail.keys():
            mbox_dict[i][header] = mail[header]
        mbox_dict[i]['Text'] = mail.get_payload()
    df = pd.DataFrame.from_dict(mbox_dict, orient='index')
    return df


def read_msg(mb_msg):
    if type(mb_msg) == list:
        mb = mailbox.mboxMessage(mb_msg[0])
        return mb.get_payload()
    else:
        return mb_msg


train = read_mbox(trainfile)
test = read_mbox(testfile)
test['Text'] = test['Text'].apply(read_msg)

"""## Препроцессинг"""


def subject_cleaner(text):
    text = text.str.strip()
    text = text.str.lower()
    return text


def body_cleaner(text):
    # text = text.str.replace(r'<[^>]+>', '')
    # text = text.str.replace(r'{[^}]+}', '')
    # text = text.str.replace(r'#message', '')
    # text = text.str.replace(r'\n{1,}', '')
    # text = text.str.replace(r'={1,}', ' ')
    # text = text.str.replace(r'-{2,}', ' ')
    # text = text.str.replace(r'\*{1,}', ' ')
    # text = text.str.replace(r'&nbsp{1,}', ' ')
    # text = text.str.replace(r'\t', ' ')
    # text = text.str.replace(r'\s{1,}', ' ')
    text = text.str.strip()
    text = text.str.lower()
    return text


train['Subject'] = subject_cleaner(train['Subject'])
test['Subject'] = subject_cleaner(test['Subject'])
train['Text'] = body_cleaner(train['Text'])
test['Text'] = body_cleaner(test['Text'])

train['Content-Type'] = train['Content-Type'].apply(lambda x: x.lower().split(';')[0])

test['Content-Type'] = test['Content-Type'].str.lower().str.strip()
test['Content-Type'] = test['Content-Type'].apply(lambda x: str(x).split(';')[0])
test['Content-Type'][test['Content-Type'] == 'nan'] = ''
test['Content-Type'][test['Content-Type'] == ''] = 'no_type'
test['Content-Type'][test['Content-Type'] == 'text/html content-transfer-encoding: 8bit\\r\\n'] = 'text/html'

# train_from = {b : a for a,b in enumerate(train['From'].unique())}
# for i in test['From'].unique():
#   cnt = len(train_from) - 1
#   if i in train_from.keys():
#     pass
#   else:
#     cnt += 1
#     train_from[i] = cnt
# train['From_keys'] = train['From'].apply(lambda x: train_from[x])
# test['From_keys'] = test['From'].apply(lambda x: train_from[x])


# нераспарсеная почта
test['Text'] = test['Text'].fillna('email')

test['Subject'] = test['Subject'].fillna('')

# test = test[test['Text'].apply(lambda x: type(x)!=list)]

"""## Токенизация"""

tfidf_type = TfidfVectorizer(
    norm='l2',
    analyzer='word',
    ngram_range=(1, 2),
)
tfidf_type.fit(train['Content-Type'])
train_data_tp = tfidf_type.transform(train['Content-Type'])
test_data_tp = tfidf_type.transform(test['Content-Type'])

tfidf_from = TfidfVectorizer(
    norm='l2',
    analyzer='word',
    ngram_range=(1, 1),
)
tfidf_from.fit(train['From'])
train_data_f = tfidf_from.transform(train['From'])
test_data_f = tfidf_from.transform(test['From'])

tfidf_subject = TfidfVectorizer(
    norm='l2',
    analyzer='word',
    ngram_range=(1, 2),
)
tfidf_subject.fit(train['Subject'])
train_data_s = tfidf_subject.transform(train['Subject'])
test_data_s = tfidf_subject.transform(test['Subject'])

tfidf_body1 = TfidfVectorizer(
    # norm='l2',
    analyzer='word',
    ngram_range=(1, 2),
)
tfidf_body1.fit(train['Text'])
train_data_t1 = tfidf_body1.transform(train['Text'])
test_data_t1 = tfidf_body1.transform(test['Text'])

tfidf_body2 = TfidfVectorizer(
    # norm='l2',
    analyzer='char_wb',
    ngram_range=(2, 5),
)
tfidf_body2.fit(train['Text'])
train_data_t2 = tfidf_body2.transform(train['Text'])
test_data_t2 = tfidf_body2.transform(test['Text'])

# from_keys_tr = scipy.sparse.csr_matrix(np.expand_dims(train['From_keys'].values, axis=1))
# from_keys_ts = scipy.sparse.csr_matrix(np.expand_dims(test['From_keys'].values, axis=1))

# train_data = hstack([train_data_f, train_data_tp,train_data_s, train_data_t])
# test_data = hstack([test_data_f,test_data_tp,test_data_s, test_data_t])

train_data = hstack([train_data_t1, train_data_t2])
test_data = hstack([test_data_t1, test_data_t2])

"""## Модель_1"""

clf = OneClassSVM().fit(train_data)

pred = clf.predict(test_data)

pred[pred == -1] = 0

pred[pred == 0].shape  ##много

test['Label'] = pred

test['Label_2'] = test['Content-Type'].apply(lambda x: 1 if x == 'text/plain' else 0)

from sklearn.metrics import accuracy_score

accuracy_score(test['Label'], test['Label_2'])

"""## Дополнительный check для теста

### тест отправителя From
- enron - 0, 
- внешние - 1
"""

test['check_email'] = test['From'].str.contains('~@enron') * 1

"""### тест на наличие warning words в теме письма
- 1 слово - 0.5 баллов, 
- 2 слова и более - 1 бал, 
- 0 баллов нет
"""

wwords = ['payment', 'urgent', 'bank', 'account', 'access', 'block', 'limit', 'confirm', 'important',
          'password', 'require', 'file', 'download', 'request', 'security', 'validat', 'suspend', 'verificat',
          'update', 'cash', 'fraud', 'error', 'alert', 'lock', 'card', 'bill', 'official', 'online', 'secure',
          'profile', 'modif', 'deposit', 'offer', 'verif', 'inquiry', 'free', 'unusual', 'identif',
          ]


def check_sub(x):
    cnt = 0
    x = re.sub(r'[^A-Za-z]', ' ', x)
    for i in x.split():
        if i in wwords:
            cnt += 1
    if cnt == 0:
        return 0
    if cnt == 1:
        return 0.5
    if cnt > 1:
        return 1


test['check_subject'] = test['Subject'].apply(check_sub)

"""### тест на наличие warning words в теле письма
- 1 слово - 0.5 баллов, 
- 2 слова и более - 1 бал, 
- 0 баллов нет
"""

test['check_body'] = test['Text'].apply(check_sub)

"""### тест на наличие http ссылок, за исключением номинального имени сайта, в теле письма
- есть - 1 бал,
- нет - 0 баллов
"""

test['check_text_http'] = test['Text'].str.contains(r'(http|https):\/\/.+?(?=\/)\/\w') * 1

"""### сумируем check"""

test['sum_check'] = test['check_email'] + test['check_subject'] + test['check_body'] + test['check_text_http']

test.iloc[1]

test['Content-Type'].iloc[1]

train['Content-Type'][train['Content-Type'].str.contains('multipart/alternative')]

"""## Выгрузка"""

test[['X-UID', 'Label']].to_csv('Result.csv')

test['Content-Type'].value_counts()
