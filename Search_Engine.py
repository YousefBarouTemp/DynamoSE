import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import qalsadi.lemmatizer
nltk.download("punkt")
nltk.download('stopwords')

class SearchEngine():
    """Search Engine class for making an arabic search engine
       This class is using an ARABIC NEWS DATASET https://www.sciencedirect.com/science/article/pii/S2352340919304305
       AUTHORS: Omar Einea, Ashraf Alnagar, Ridhwan Al Debsi
       Methods:
       find(query, choice): this the only interface of the class, you must provide the query and
       the searching type either using boolean model or Ranking search
    """
    def __init__(self):
        # Initializing and configuring all requirements to reduce computation
        
        self.__docs = pd.read_csv("assets/Arabic news preprocessed_2500.csv", index_col=0)
        
        with open("assets/term_document_matrix_data.pkl","rb") as file:
            self.__data = pickle.load(file)
            
        with open("assets/tfidf_vectorizer.pkl", "rb") as file:
            self.__tfidf_vectorizer = pickle.load(file)

        with open("assets/tfidf_doc_vec.pkl", 'rb') as file:
            self.__tfidf_docs_vec = pickle.load(file)
            
        with open("assets/docs_norm.pkl", "rb") as file:
            self.__tfidf_docs_norm = pickle.load(file)

        self.__voc = self.__data['Voc']
        self.__term_frequency_model = self.__data['Model']

    def __preprocess(self, query):
        # Removing Punctuations
        query = str(query)
        punct_query = re.sub(r'[^\w\s]', '', query)
    
        # Tokenization
        tokenized_query = word_tokenize(punct_query)
    
        # Removing stopwords
        arabic_stopwords = set(stopwords.words('arabic'))
        stopword_query = [word for word in tokenized_query if word not in arabic_stopwords]
        stopword_query = " ".join(stopword_query)
    
        # removing arabic diacritics
        arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        dia_query = re.sub(arabic_diacritics, '', stopword_query)
    
        # normalizing arabic words
        norm_query = re.sub("[إأآا]", "ا", dia_query)
        norm_query = re.sub("ى", "ي", norm_query)
        norm_query = re.sub("ؤ", "ء", norm_query)
        norm_query = re.sub("ئ", "ء", norm_query)
        norm_query = re.sub("ة", "ه", norm_query)
        norm_query = re.sub("گ", "ك", norm_query)
    
        # Lemmatization
        lemmer = qalsadi.lemmatizer.Lemmatizer()
        lemmatized_query = lemmer.lemmatize_text(norm_query)
    
        return ' '.join(lemmatized_query)

    def __extract_vocabulary(self):
        # Method used to extract the vocabulary of the documents
        # - Saved to reduce computations later
        list_docs = list(self.__docs['Preprocessed'].to_numpy())
        self.__voc = []
        for doc in list_docs:
            doc_temp = doc.split()
            for word in doc_temp:
                if word not in self.__voc:
                    self.__voc.append(word)

    def __term_document_matrix_indexing(self):
        # Method used to create the indexing of the boolean model for the documents
        # - Saved to reduce computations later
        list_docs = list(self.__docs['Preprocessed'].to_numpy())
        self.__term_frequency_model = []
        for d in list_docs:
            freq = []
            for word in self.__voc:
                if word in d:
                    freq.append(1)
                else:
                    freq.append(0)
            self.__term_frequency_model.append(freq)
            
    def __tfidf_indexing(self):
        # Method used to create the indexing of the ranking search for the documents
        # - Saved to reduce computation later
        list_docs = list(self.__docs['Preprocessed'].to_numpy())
        self.__tfidf_vectorizer = TfidfVectorizer()
        self.__tfidf_docs_vec = self.__tfidf_vectorizer.fit_transform(list_docs)

    def __term_document_matrix_query_transform(self,query):
        # Method used especially for the boolean model to vectorize the query depends on the
        # the boolean model created for the documents
        # :: Check if the vocabulary exists in the query on not (Vector)
        query_vector = []
        for word in self.__voc:
            if word in query:
                query_vector.append(1)
            else:
                query_vector.append(0)
        return query_vector

    def __tfidf_query_transform(self,query):
        # Method used especially for the ranking model to vectorize the query depends on the
        # the tf-idf model created for the documents
        query_vector = self.__tfidf_vectorizer.transform([query])
        return query_vector


    def __term_document_matrix_retrieve(self, query_vector):
        # Method used to retrieve all results based on the query (Vector) for the boolean model
        # :: Counts the number of matched words for ranking
        dict_res = {}
        for i in range(len(self.__term_frequency_model)):
            count = 0
            vec = self.__term_frequency_model[i]
            for t in range(len(vec)):
                if query_vector[t] == 1 and vec[t] == 1:
                    count += 1
            dict_res[i] = count
        return dict_res

    def __tfidf_retrieve(self,query_vector):
        # Method used to retrieve all results based on the query (Vector) for the ranking model
        # :: Computing the similarity between the query vector and the vector of each document
        similarity = np.dot(query_vector, self.__tfidf_docs_vec.T) / (norm(query_vector.toarray()) * self.__tfidf_docs_norm)
        return similarity


    def __term_document_matrix_ranking(self,query):
        # Method used to retrieve the top ten results of the boolean model
        # :: Collecting all private methods
        preprocessed_query = self.__preprocess(query)
        tokenized_preprocessed_query = preprocessed_query.split()
        query_vector = self.__term_document_matrix_query_transform(tokenized_preprocessed_query)
        result = self.__term_document_matrix_retrieve(query_vector)
        
        list_res = list(result.items())
        list_res.sort(reverse=True, key=lambda x:x[1])
        
        top_ten = []
        for Ix, _ in list_res[:10]:
            top_ten.append(self.__docs['Article'].iloc[Ix])
        
        del preprocessed_query,tokenized_preprocessed_query, query_vector, result, list_res
        return top_ten

    def __tfidf_ranking(self,query):
        # Method used to retrieve the top ten results of the ranking model
        # :: Collecting all private methods
        preprocessed_query = self.__preprocess(query)
        query_vector = self.__tfidf_query_transform(preprocessed_query)
        result = self.__tfidf_retrieve(query_vector)

        list_res = list(zip(self.__docs.index,result.toarray()[0]))
        list_res.sort(reverse=True, key=lambda x:x[1])

        top_ten = []
        for Ix, _ in list_res[:10]:
            top_ten.append(self.__docs['Article'].iloc[Ix])
        
        del preprocessed_query, query_vector, result, list_res
        return top_ten


    def find(self, query, choice):
        # The interface of the search engine which takes query and return the result
        # based on the choice either (1) for boolean model or (2) for ranking model -tfidf-
        if choice == 1:
            result = self.__term_document_matrix_ranking(query)
        elif choice == 2:
            result = self.__tfidf_ranking(query)

        return result