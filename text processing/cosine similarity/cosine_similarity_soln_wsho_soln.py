import nltk
import numpy as np
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

		# split our document based on 'newline' character
		docs = doc.split('\n')

		file.close()
		
		return docs
	except FileNotFoundError:
		return None


'''
Remove stopwords from our "documents" (each quote
is regarded as a "document" by our program)
'''
def preprocessing(docs):
    # use the English stopwords provided by NLTK
    stop_words = stopwords.words('english')

    # get a Stemmer to convert words to root forms
    stemmer = nltk.stem.PorterStemmer()
    # sentences = nltk.tokenize.sent_tokenize(doc)

    # to hold processed "documents"
    clean_docs = []

    # map every punctuation found in string.punctuation
    # to an empty character (essentially removing them
    # from our corpus)
    punc = str.maketrans('','', string.punctuation)
    
    # perform stemming on each word, and convert each word to 
    # lowercase (so that words 'Bad', 'BAD', 'bad' are all treated 
    # the same)
    for doc in docs:
        doc_no_punc = doc.translate(punc)
        words_stemmed = [stemmer.stem(w) 
            for w in doc_no_punc.lower().split()
                if w not in stop_words]
        clean_docs += [' '.join(words_stemmed)]   

    return clean_docs 


'''
Accepts a N_QUERY_DOCS X N_DOCS_IN_CORPUS matrix similarity matrix.
'''
def make_similarity_dataframe(similarity_matrix):
	# len(similarity_matrix[0]) gives N_DOCS_IN_CORPUS (no. of columns, 
	# which translates to number of documents being queried against)
	column_labels = ["doc" + str(i) for i in range(len(similarity_matrix[0]))]

	# len(similarity_matrix) gives N_QUERY_DOCS (no. of rows, which translates
	# to number of query strings)
	row_labels = ["query" + str(i) for i in range(len(similarity_matrix))]

	# creating a dataframe with the constructed labels
	return pd.DataFrame(similarity_matrix, index=row_labels, columns=column_labels)


'''
Print the quotes that is closely to the query
'''
def print_results(query, docs, series):
	# sort the series by values; larger values first
	sorted_series = series.sort_values(ascending=False)

	# filter away entries that have values of 0
	# '!= 0' means only want values that are not 0
	sorted_series = sorted_series[sorted_series != 0]
	
	print("Query: " + query)
	print("Results: ")

	for i in sorted_series.index:
		# discard the "doc" prefix in the row-label
		# only want the document-position in our corpus
		pos = int(i[3:])	

		# display the quotes and corresponding cosine similarity scores
		print(docs[pos] + ' [score:', sorted_series[i], ']', sep='')


'''
Entry point for our program.
'''
def main():
	# our query string
	query = 'life wise choices'

	# ensure all NLTK resources are available
	prep_nltk_data();

	# our dataset 
	docs = read_data('dataset/quotes.txt')

	# clean our data (e.g. remove stopwords, punctuations)
	docs_arr = preprocessing(docs)

	# get TF-IDF object from SKLearn and calculate
	# TF-IDF frequency-weightings
	tfidf = TfidfVectorizer()
	tfidf = tfidf.fit(docs_arr)

	# apply calculated TF-IDF frequency-weightings onto our 
	# "documents" (i.e. every quote is regarded as a document)
	docs_vecs = tfidf.transform(docs_arr).toarray()
	
	# clear our query
	query_arr = preprocessing([query])	

	# use the same TF-IDF frequency-weightings on our query string
	query_vec = tfidf.transform(query_arr).toarray()

	# gives us a N_QUERY_DOCS X N_DOCS matrix
	docs_similarity = cosine_similarity(query_vec, docs_vecs)

	# make a dataframe for easy viewing
	df = make_similarity_dataframe(docs_similarity)
	print(df)

	# show quotes that match the query as we only have one query, 
	# the first entry [0] corresponds to the cosine similarity scores 
	# for our query string 
	print_results(query, docs, df.iloc[0])


# running via "python cosine_similarity_wshop_soln.py"
if __name__ == "__main__":
  main()