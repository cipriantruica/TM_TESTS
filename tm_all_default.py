# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2015, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.com"
__status__ = "Production"

from gensim.corpora import MmCorpus
from gensim.models import LsiModel, LdaModel, HdpModel, TfidfModel
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from time import time
import numpy as np

from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from os import listdir
from os.path import isfile, join
import codecs
import sys
#from stop_words import get_stop_words
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# import pymongo
import random
import doc2class



class TopicModeling:
    def __init__(self, id2word, corpus, doc2class=None, num_cores=-14):
        self.id2word = id2word
        self.corpus = corpus        
        self.doc2class = doc2class
        self.num_cores = num_cores
        self.doc2topicLSI = {}        
        self.doc2topicNMF = {}
        self.doc2topicKMeans = {}
        self.doc2topicLDA_gensim = {}
        self.doc2topicLDA_sklearn = {}

    def topicsLDA_gensim(self, num_topics=10, num_words=10, num_iterations=2000, chunksize=20000, decay=0.5):
        lda = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, chunksize=chunksize, iterations=num_iterations, alpha='auto', eta='auto', decay=decay)

        # documents for each topic
        if self.doc2class:
            doc_idx = 0
            for line in lda[self.corpus]:
                # get topic with maximum percentage
                if line:
                    topic_idx = max(line, key=lambda item:item[1])[0]
                else:
                    # if there is no topic assign a random one
                    topic_idx = random.randint(0, num_topics - 1)
                # make the dictionary                
                if self.doc2topicLDA_gensim.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicLDA_gensim[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicLDA_gensim[self.doc2class[doc_idx]][i] = 0
                self.doc2topicLDA_gensim[self.doc2class[doc_idx]][topic_idx] += 1
                doc_idx += 1
            print self.doc2topicLDA_gensim
        # return topics
        return lda.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

    def topicsLSI(self, num_topics=10, num_words=10, num_iterations=2000, chunksize=20000, decay=0.5, onepass=False):
        # LsiModel(corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=2, extra_samples=100)
        lsi = LsiModel(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, chunksize=chunksize, onepass=onepass, power_iters=num_iterations, decay=decay)

        # documents for each topic
        if self.doc2class:
            doc_idx = 0
            for line in lsi[self.corpus]:
                # get topic with maximum percentage
                if line:
                    topic_idx = max(line, key=lambda item:item[1])[0]
                else:
                    # if there is no topic assign a random one
                    topic_idx = random.randint(0, num_topics - 1)
                # make the dictionary
                if self.doc2topicLSI.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicLSI[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicLSI[self.doc2class[doc_idx]][i] = 0
                self.doc2topicLSI[self.doc2class[doc_idx]][topic_idx] += 1
                doc_idx += 1
            print self.doc2topicLSI

        # show_topics(num_topics=-1, num_words=10, log=False, formatted=True)
        # Return num_topics most significant topics (return all by default).
        # For each topic, show num_words most significant words (10 words by default).
        # The topics are returned as a list – a list of strings if formatted is True, or a list of (weight, word) 2-tuples if False.
        # If log is True, also output this result to log.                
        return lsi.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

        # show_topics(num_topics=-1, num_words=10, log=False, formatted=True)
        # Return num_topics most significant topics (return all by default).
        # For each topic, show num_words most significant words (10 words by default).
        # The topics are returned as a list – a list of strings if formatted is True, or a list of (weight, word) 2-tuples if False.
        # If log is True, also output this result to log.                
        return lsi.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

    def topicsHDP(self, num_topics=-1, num_words=10, chunksize=20000):
        # HdpModel(corpus, id2word, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None)
        hdp = HdpModel(corpus=self.corpus, id2word=self.id2word, chunksize=chunksize)

        # show_topics(topics=20, topn=20, log=False, formatted=True)
        # Print the topN most probable words for topics number of topics. Set topics=-1 to print all topics.
        # Set formatted=True to return the topics as a list of strings, or False as lists of (weight, word) pairs.
        # for elem in hdp[self.corpus_mm]:
        #   print elem
        return hdp.show_topics(topics=num_topics, topn=num_words, formatted=False)

    def topicsNMF(self, num_topics=10, num_words=10, num_iterations=2000):
        model = NMF(init="nndsvd", n_components=num_topics, max_iter=num_iterations)
        
        W = model.fit_transform(self.corpus)
        H = model.components_

        # Documents for each topic
        if self.doc2class:
            doc_idx = 0
            for line in W:
                topic_idx = np.where(line==line.max())[0][0]
                # make the dictionary
                if self.doc2topicNMF.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicNMF[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicNMF[self.doc2class[doc_idx]][i] = 0
                self.doc2topicNMF[self.doc2class[doc_idx]][topic_idx] += 1
                doc_idx += 1
            print self.doc2topicNMF
        
        # NMF topics
        topics = []
        for topic_index in range( H.shape[0] ):
            top_indices = np.argsort( H[topic_index,:] )[::-1][0:num_words]
            term_ranking = [(self.id2word[i], H[topic_index][i]) for i in top_indices]
            topics.append((topic_index, term_ranking))
        return topics

    def topicsLDA_sklearn(self, num_topics=10, num_words=10, num_iterations=2000, chunksize=20000, decay=0.5):
        lda_model = LatentDirichletAllocation(n_topics=num_topics, doc_topic_prior=None, topic_word_prior=None, learning_method='online', learning_decay=decay, learning_offset=10.0, max_iter=num_iterations, batch_size=chunksize, evaluate_every=-1, total_samples=chunksize, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=self.num_cores, verbose=0, random_state=None)
        lda_model.fit(self.corpus)
        H = lda_model.components_
        # print X
        if self.doc2class:
            doc_idx = 0
            for elem in lda_model.transform(self.corpus):
                topic_idx = np.where(elem==elem.max())[0][0]
                if self.doc2topicLDA_sklearn.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicLDA_sklearn[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicLDA_sklearn[self.doc2class[doc_idx]][i] = 0
                self.doc2topicLDA_sklearn[self.doc2class[doc_idx]][topic_idx] += 1
                doc_idx += 1
            print self.doc2topicLDA_sklearn

        topics = []
        for topic_index in range( H.shape[0] ):
            top_indices = np.argsort( H[topic_index,:] )[::-1][0:num_words]
            term_ranking = [(self.id2word[i], H[topic_index][i]) for i in top_indices]
            topics.append((topic_index, term_ranking))
        return topics

    def clustersKMeans(self, num_clusters=10, num_words=10, num_iterations=2000, n_init=10):
        kMeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=num_iterations, n_init=n_init, n_jobs=self.num_cores)
        kMeans.fit(self.corpus)
        order_centroids = kMeans.cluster_centers_.argsort()[:, ::-1]

        if self.doc2class:
            doc_idx = 0
            for elem in kMeans.predict(self.corpus):
                if self.doc2topicKMeans.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicKMeans[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_clusters):
                        self.doc2topicKMeans[self.doc2class[doc_idx]][i] = 0
                self.doc2topicKMeans[self.doc2class[doc_idx]][elem] += 1
                doc_idx += 1
            print self.doc2topicKMeans

        clusters  = []
        for i in range(num_clusters):
            terms = []
            for ind in order_centroids[i, :num_words]:
                terms.append((self.id2word[ind], order_centroids[i][ind]))
            clusters.append((i, terms)) 
        return clusters


def calMatrices(mydict):
    mat = []
    b = []
    a = []
    n = 0
    conv_mat = []
    for key in mydict:
        l = []
        for val in mydict[key]:
            l.append(mydict[key][val])

        a.append(sum(l))
        mat.append(l)
        conv_mat.append([key] + l + [sum(l)])
        n += 1

    for i in range(0, n):
        suma = 0
        for j in range(0, n):
            suma += mat[j][i]
        b.append(suma)
    conv_mat.append(['Total'] + b + [sum(b)])
    header =  ['class'] + ['T'+str(i) for i in range(0, n)] + ['Total']
    # print '\t'.join(str(elem) for elem in header)
    # for line in conv_mat:
    #     print '\t'.join(str(elem) for elem in line)
    return mat, a, b

def printMatrix(mat):
    format_mat = "{"
    i = 0
    n = len(mat)
    m = len(mat[0])
    for line in mat:
        format_mat += "{"
        j = 0
        for cell in line:
            if j != m-1:
                format_mat += str(cell)+".0,"
            else:
                format_mat += str(cell)+".0"
            j += 1
        if i != n-1:
            format_mat += "},"
        else:
            format_mat += "}"
        i += 1
    format_mat += "}"
    print format_mat

#list of stopwords
def stopWordsEN():
    # sw_stop_words = get_stop_words('en')
    sw_nltk = stopwords.words('english')
    sw_sklearn = list(ENGLISH_STOP_WORDS)
    sw_mallet = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'way', 'we', 'welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wonder', 'would', 'would', 'x', 'y', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']
    return list(set(sw_nltk + sw_mallet + sw_sklearn))

# reads the documents from file path
def readDocuments(mypath):
    sw_en = stopWordsEN()
    documents = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    documents = [""]*len(onlyfiles)
    print "len files", len(onlyfiles)
    for f in onlyfiles:
        with codecs.open(join(mypath, f), 'r',  "utf-8") as filename:
            lines = ""
            idx = int(f[:-4])
            for line in filename:
                lines += line.replace('\r', '').replace('\n', '') + ' '
            documents[idx] = [word.lower() for word in line.split(' ') if word.lower() not in sw_en]
    return documents

# prepare the corpus for Gensim
def prepareCorpusGemsin(documents):
    id2word_mm = corpora.Dictionary(documents)
    # creates a vectorize corpus
    corpus_count_mm  = [id2word_mm.doc2bow(text) for text in documents]
    tfidf_mm = TfidfModel(corpus_count_mm)
    corpus_tfidf_mm = tfidf_mm[corpus_count_mm]

    return id2word_mm, corpus_count_mm, corpus_tfidf_mm

# prepare the corpus for Sklearn
def prepareCorpusSklearn(documents):
    texts = [' '.join(document) for document in documents]

    tfidf_vect = TfidfVectorizer(lowercase=True, strip_accents="unicode", use_idf=True, norm="l2")
    corpus_tfidf_csr = tfidf_vect.fit_transform(texts)

    id2word_tfidf_csr = {}
    for term in tfidf_vect.vocabulary_.keys():
        id2word_tfidf_csr[ tfidf_vect.vocabulary_[term] ] = term

    count_vect = CountVectorizer(lowercase=True, strip_accents="unicode")
    corpus_count_csr = count_vect.fit_transform(texts)
   
    id2word_count_csr = {}
    for term in count_vect.vocabulary_.keys():
        id2word_count_csr[ count_vect.vocabulary_[term] ] = term

    return id2word_tfidf_csr, corpus_tfidf_csr, id2word_count_csr, corpus_count_csr

        
if __name__ == '__main__':
    in_filepath = sys.argv[1]
    num_topics = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    num_cores = int(sys.argv[4])
    
    documents = readDocuments(in_filepath)
    d2c = doc2class.doc2class_news
    id2word_mm, corpus_count_mm, corpus_tfidf_mm = prepareCorpusGemsin(documents)
    id2word_tfidf_csr, corpus_tfidf_csr, id2word_count_csr, corpus_count_csr = prepareCorpusSklearn(documents)
    
    print "NMF COUNT"
    start = time()
    topic_model = TopicModeling(id2word=id2word_count_csr, corpus=corpus_count_csr, doc2class=d2c, num_cores=num_cores)
    nmfTopics = topic_model.topicsNMF(num_topics=num_topics, num_iterations=num_iterations)
    for topic in nmfTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "NMF COUNT time", (end - start)

    print "NMF TFIDF"
    start = time()
    topic_model = TopicModeling(id2word=id2word_tfidf_csr, corpus=corpus_tfidf_csr, doc2class=d2c, num_cores=num_cores)
    nmfTopics = topic_model.topicsNMF(num_topics=num_topics, num_iterations=num_iterations)
    for topic in nmfTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "NMF TFIDF time", (end - start)

    print "KMEANS COUNT"
    start = time()
    topic_model = TopicModeling(id2word=id2word_count_csr, corpus=corpus_count_csr, doc2class=d2c, num_cores=num_cores)
    kMeansTopics = topic_model.clustersKMeans(num_clusters=num_topics, num_iterations=num_iterations)
    for topic in kMeansTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "KMEANS COUNT time", (end - start)
   
    print "KMEANS TFIDF"
    start = time()
    topic_model = TopicModeling(id2word=id2word_tfidf_csr, corpus=corpus_tfidf_csr, doc2class=d2c, num_cores=num_cores)
    kMeansTopics = topic_model.clustersKMeans(num_clusters=num_topics, num_iterations=num_iterations)
    for topic in kMeansTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "KMEANS TFIDF time", (end - start)
    
    print "LDA GENSIM COUNT"
    topic_model = TopicModeling(id2word=id2word_mm, corpus=corpus_count_mm, doc2class=d2c, num_cores=num_cores)
    start = time()
    ldaTopics = topic_model.topicsLDA_gensim(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in ldaTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "LDA GENSIM COUNT time", (end - start)
   
    print "LDA GENSIM TFIDF"
    start = time()
    topic_model = TopicModeling(id2word=id2word_mm, corpus=corpus_tfidf_mm, doc2class=d2c, num_cores=num_cores)
    ldaTopics = topic_model.topicsLDA_gensim(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in ldaTopics:
       wTopics = []
       for words in topic[1]:
           wTopics.append(words[0])
       print "Topic", topic[0], wTopics
    end = time()
    print "LDA GENSIM TFIDF time", (end - start)

    print "LDA SKLEARN COUNT"
    start = time()
    topic_model = TopicModeling(id2word=id2word_count_csr, corpus=corpus_count_csr, doc2class=d2c, num_cores=num_cores)
    ldaTopics = topic_model.topicsLDA_sklearn(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in ldaTopics:
        wTopics = []
        for words in topic[1]:
            wTopics.append(words[0])
        print "Topic", topic[0], wTopics
    end = time()
    print "LDA SKLEARN COUNT time", (end - start)

    print "LDA SKLEARN TFIDF"
    start = time()
    topic_model = TopicModeling(id2word=id2word_tfidf_csr, corpus=corpus_tfidf_csr, doc2class=d2c, num_cores=num_cores)
    ldaTopics = topic_model.topicsLDA_sklearn(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in ldaTopics:
        wTopics = []
        for words in topic[1]:
            wTopics.append(words[0])
        print "Topic", topic[0], wTopics
    end = time()
    print "LDA SKLEARN TFIDF time", (end - start)

    print "LSI GENSIM COUNT"
    start = time()
    topic_model = TopicModeling(id2word=id2word_mm, corpus=corpus_count_mm, doc2class=d2c, num_cores=num_cores)
    lsiTopics = topic_model.topicsLSI(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in lsiTopics:
        wTopics = []
        for words in topic[1]:
            wTopics.append(words[0])
        print "Topic", topic[0], wTopics
    end = time()
    print "LSI GENSIM COUNT time", (end - start)

    print "LSI GENSIM TFIDF"
    start = time()
    topic_model = TopicModeling(id2word=id2word_mm, corpus=corpus_tfidf_mm, doc2class=d2c, num_cores=num_cores)
    lsiTopics = topic_model.topicsLSI(num_topics=num_topics, num_iterations=num_iterations, chunksize=len(documents))
    for topic in lsiTopics:
        wTopics = []
        for words in topic[1]:
            wTopics.append(words[0])
        print "Topic", topic[0], wTopics
    end = time()
    print "LSI GENSIM TFIDF time", (end - start)
    


