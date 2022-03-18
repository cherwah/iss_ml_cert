import nltk
import numpy as np
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

file = open('space_invaders.txt', encoding='utf-8')
doc = file.read()
file.close()

stop_words = stopwords.words('english')

porter = nltk.stem.PorterStemmer()
sentences = nltk.tokenize.sent_tokenize(doc)

# treat each sentence as a document
docs = []

punc = str.maketrans('','', string.punctuation)
for sent in sentences:
    sent_no_punc = sent.translate(punc)
    words_stemmed = [porter.stem(w) for w in sent_no_punc.lower().split()
                     if w not in stop_words]
    docs += [' '.join(words_stemmed)]    

tfidf_vec = TfidfVectorizer()
tfidf_wm = tfidf_vec.fit_transform(docs).toarray()
# print(tfidf_wm)

df_index = ['doc'+str(i) for i in range(len(docs))]
# print(df_index)

df_columns = tfidf_vec.get_feature_names()
# print(df_columns)

tfidf_df = pd.DataFrame(data=tfidf_wm, index=df_index, columns=df_columns)
tfidf_sum_by_docs = np.sum(tfidf_df, axis=1)

sorted_series = tfidf_sum_by_docs.sort_values(ascending=False)
top_series = sorted_series.head(10)
print("\n")

# remove the prefix 'doc' from indexes such as
# [doc10, doc24, doc11, ..., doc22]
num_only = [int(x[3:]) for x in top_series.index]
print('Top TF-IDF sentences:', num_only, '\n')

for i in sorted(num_only):
    print('[Line {}]'.format(i))
    print(sentences[i] + "\n")








































