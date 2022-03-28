import nltk
import numpy as np
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


'''
Ensure that all NTLK resources have been downloaded into
the local machine.
'''
def prep_nltk_data():
    try:
       nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


'''
Read our text file.
'''
def read_data(path):
    try:
        # use Python to read in our text file
        file = open(path, encoding='utf-8')
        doc = file.read()
        file.close()
        return doc
    except FileNotFoundError:
        return None

'''
Remove stopwords from our document. The processed document
is returned as a list of documents, where each sentence is 
a document and the entire document is now our corpus.
'''
def remove_stopwords(doc):
    # use the English stopwords provided by NLTK
    stop_words = stopwords.words('english')

    # get a Stemmer to convert words to root forms
    stemmer = nltk.stem.PorterStemmer()
    sentences = nltk.tokenize.sent_tokenize(doc)

    # treat each sentence as a document
    docs = []

    # map every punctuation found in string.punctuation
    # to an empty character (essentially removing them
    # from our corpus)
    punc = str.maketrans('','', string.punctuation)
    
    # perform stemming on each word, and convert each word to 
    # lowercase (so that words 'Bad', 'BAD', 'bad' are all treated 
    # the same)
    for sent in sentences:
        sent_no_punc = sent.translate(punc)
        words_stemmed = [stemmer.stem(w) 
            for w in sent_no_punc.lower().split()
                if w not in stop_words]
        docs += [' '.join(words_stemmed)]   

    # return an array of "documents" - the "documents" are
    # actually sentences in our original document
    return docs 


'''
Perform text featurization. Here, we use TF-IDF to convert text as 
feature-vectors. After that, we convert the generated feature-vectors 
into a Pandas dataframe to facilitate our computations later.
'''
def gen_features_as_dataframe(docs):
    # get a TF-IDF object from sklearn
    tfidf = TfidfVectorizer()

    # convert to NumPy array
    tfidf_vecs = tfidf.fit_transform(docs).toarray()

    # index for our dataframe
    df_index = ['doc'+str(i) for i in range(len(docs))]

    # columns for our dataframe
    df_columns = tfidf.get_feature_names()

    return pd.DataFrame(data=tfidf_vecs, 
        index=df_index, columns=df_columns)


'''
Get top N documents with the largest TF-IDF values.
'''
def get_top_tfidf_sums(df, n):
    # sum up the TF-IDF values for each document
    # axis=1 because the columns in our dataframe
    # are collapsed to give us a total value
    tfidf_sum_by_docs = np.sum(df, axis=1)

    # sort the computed sums in descending order
    # as we want the "documents" with the highest 
    # TF-IDF values
    sorted_series = tfidf_sum_by_docs.sort_values(ascending=False)
    
    # only returns the top N "documents"
    return sorted_series.head(n)


'''
Print out the sentences in the original document with the 
highest TF-IDF values.
'''
def print_summary(top_series, doc):
    # remove the prefix 'doc' from indexes such as
    # [doc10, doc24, doc11, ..., doc22]
    num_only = [int(x[3:]) for x in top_series.index]

    # we now have the top 10 "documents", but we want
    # our summary to show them as sentences as appear
    # in the same order as the original document
    sorted_num_only = sorted(num_only)

    # reference back to our original document
    sentences = nltk.tokenize.sent_tokenize(doc)

    # print out sentences, with top TF-IDF values,
    # in the order that they appear in the original
    # document
    for i in sorted_num_only:
        print('[Line {}]'.format(i))
        print(sentences[i] + "\n")


'''
Entry point of our program.
'''
def main():
    # ensure all NLTK resources are available
    prep_nltk_data()

    # reading in our dataset
    doc = read_data('dataset/space_invaders.txt')

    # perform data processing
    docs = remove_stopwords(doc)

    # generate TF-IDF features
    df = gen_features_as_dataframe(docs)

    # get top 10 documents by computed TF-IDF values
    top_series = get_top_tfidf_sums(df, 10)

    # show extracted sentences as summary
    print_summary(top_series, doc);


# running via "python summarization_tfidf_wshop_soln.py"
if __name__ == "__main__":
  main()





































