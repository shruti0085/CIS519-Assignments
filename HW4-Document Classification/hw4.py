
# coding: utf-8

# # Homework 4: Document Classification
# 
# In this problem, you will implement several text classification systems using the naive Bayes algorithm and semi-supervised Learning. In addition to the algorithmic aspects -- you will implement a  naive Bayes classifier and semi-supervised learning protocols on top of it -- you will also get a taste of data preprocessing and feature extraction needed to do text classification.
# 
# Please note that we are leaving some of the implementation details to you. In particular, you can decide how to representation the data internally, so that your implementation of the algorithm is as efficient as possible. 
# <b>Note, however, that you are required to implement the algorithm yourself, and not use existing implementations, unless we specify it explicitly.</b>
# 
# Also note that you are free to add <b>optinal</b> parameters to the given function headers. However, please do not remove existing parameters or add additional non-optional ones. Doing so will cause Gradescope tests to fail.

# ## 1.1: Dataset
# 
# For the experiments, you will use the 20 newsgroups text dataset. It consists of ∼18000 newsgroups posts on 20 topics. We have provided 2 splits of the data. In the first split, 20news-bydate, we split the training and testing data by time, i.e. you will train a model on the data dated before a specified time and test that model on the data dated after that time. This split is realistic since in a real-world scenario, you will train your model on your current data and classify future incoming posts.

# ## 1.2: Preprocessing
# 
# In the first step, you need to preprocess all the documents and represent it in a form that can be used later by your classifier. We will use three ways to represent the data:
# 
# * Binary Bag of Words (B-BoW)
# * Count Bag of Words (C-BoW)
# * TF-IDF
# 
# We define these representations below. In each case, we define it as a matrix, where rows correspond to documents in the collection and columns correspond to words in the training data (details below). 
# 
# However, since the vocabulary size is too large, it will not be feasible to store the whole matrix in memory. You should use the fact that this matrix is really sparse, and store the document representation as a dictionary mapping from word index to the appropriate value, as we define below. For example, $\texttt{\{'doc1': \{'word1': val1, 'word2': val2,...\},...\}}$
# 
# <b><i>Please name the documents $<folder\_name>/<file\_name>$ in the dictionary.</i></b> i.e. 'talk.politics.misc/178761'
# <b>We would like you to do the preprocessing yourself, following the directions below. Do not use existing tokenization tools.</b>

# ### 1.2.1: Binary Bag of Words (B-BoW) Model
# 
# Extract a case-insensitive (that is, "Extract" will be represented as "extract") vocabulary set, $\mathcal{V}$, from the document collection $\mathcal{D}$ in the training data. Come up with a tokenization scheme - you can use simple space separation or more advanced Regex patterns to do this. You should lemmatize the tokens to extract the root word, and use a list of "stopwords" to ignore words like <i>the</i>, <i>a</i>, <i>an</i>, etc. <b>When reading files, make sure to include the 	exttt{errors='ignore'} option</b>
# 
# The set $\mathcal{V}$ of vocabulary extracted from the training data is now the set of features for your training. You will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# 
# \begin{equation*}
# \forall ~v \in  \mathcal{V}, ~\forall d \in  \mathcal{D}, ~~~~f[d,v]=
# \begin{cases}
#     1 & \text{if } v \in d \\
#     0 & \text{else}
# \end{cases}
# \end{equation*}
# 
# This should be a general function callable for any training data.

# In[1]:


import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import math
from itertools import product

import numpy as np


# In[2]:



import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[3]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\train')


# In[4]:


def b_bow(directory_name, use_adv_tokenization=False):
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    value 1 if it appears in the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: B-BoW Model
    '''
    

    lemmatizer = WordNetLemmatizer()
    stopW = set(stopwords.words('english'))
    word_dic ={}
    
    for folder in os.listdir(directory_name):

        for file in os.listdir(directory_name +'/'+folder):

            word_dic[folder +'/'+file] ={}
            with open(directory_name +'/'+folder+'/'+file , 'r', errors = 'ignore') as f:
                for line in f:
                    words = set([lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW])

                    for word in words:
                        word_dic[folder +'/'+file][word] = 1

                        
                              
    return word_dic
    #TODO
    pass


# In[55]:


# b_bow(os.getcwd(), use_adv_tokenization=False)


# ### 1.2.2: Count Bag of Words (C-BoW) Model
# 
# The first part of vocabulary extraction is the same as above.
# 
# Instead of using just the binary presence, you will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$, along with their counts. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# 
# \begin{equation*}
# f[d, v] = tf(d, v), ~~~~\forall v \in  \mathcal{V}, ~~~~\forall d \in  \mathcal{D},
# \end{equation*}
# 
# where, $tf(d,i)$ is the <i>Term-Frequency</i>, that is, number of times the word $v \in \mathcal{V}$ occurs in document $d$.

# In[5]:


def c_bow(directory_name, use_adv_tokenization=False):
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    number of times it appears in the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: C-BoW Model
    '''
    
    lemmatizer = WordNetLemmatizer()
    word_dic ={}
    stopW = stopwords.words('english')
    for folder in os.listdir(directory_name):

        for file in os.listdir(directory_name +'/'+folder):
            word_dic[folder +'/'+file] ={}
            with open(directory_name +'/'+folder+'/'+file , 'r', errors = 'ignore') as f:
                for line in f:
                    words = [lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW]
                    for word in words:

                        if word not  in word_dic[folder +'/'+file]:
                            word_dic[folder +'/'+file][word] = 1
                        else:
                            word_dic[folder +'/'+file][word] += 1
                              
    return word_dic   
    #TODO
    pass


# In[ ]:


# c_bow(os.listdir(os.getcwd()), use_adv_tokenization=False)


# ### 1.2.3: TF-IDF Model
# 
# The first part of vocabulary extraction is the same as above.
# 
# Given the Document collection $\mathcal{D}$, calculate the Inverse Document Frequency (IDF) for each word in the vocabulary $\mathcal{V}$. The IDF of the word $v$ is defined as the log (use base 10) of the multiplicative inverse of the fraction of documents in $\mathcal{D}$ that contain $v$. That is:
# $$idf(v) = \log\frac{|\mathcal{D}|}{|\{d \in \mathcal{D} ;v \in d\}|}$$
# 
# Similar to the representation above, you will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$, along with their <b>tf idf</b> value. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# \begin{equation*}
# f[d, v] = tf(d, v) \cdot idf(v, \mathcal{D}), ~~~~\forall v \in  \mathcal{V}, ~~~~\forall d \in  \mathcal{D},
# \end{equation*}
# 
# where, $tf(.)$ is the <i>Term-Frequency</i>, and $idf(.)$ is the <i>Inverse Document-Frequency</i> as defined above.

# In[6]:


def tf_idf(directory_name, use_adv_tokenization=False):
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    its TF-IDF value, with respect to the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: TF-IDF Model
    '''
    
    lemmatizer = WordNetLemmatizer()
    word_dic ={}
    idf_dic ={}
    stopW = stopwords.words('english')
    for folder in os.listdir(directory_name):
        for file in os.listdir(directory_name +'/'+folder):
            word_dic[folder +'/'+file] ={}
            with open(directory_name +'/'+folder+'/'+file , 'r', errors = 'ignore') as f:
                for line in f:
                    words = [lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW]
                    for word in words:

                        if word not in word_dic[folder +'/'+file]:
                            word_dic[folder +'/'+file][word] = 1
                        else:
                            word_dic[folder +'/'+file][word] += 1
            for word in word_dic[folder +'/' + file]:                            
                if word not in idf_dic:
                    idf_dic[word] = 1 
                else:
                    idf_dic[word] += 1
    D = len(word_dic) 
    tf_idf_dic ={}
    for folder in os.listdir(directory_name):                              
        for file in os.listdir(directory_name +'/'+folder): 
            tf_idf_dic[folder +'/'+file] ={}
  
            for word in word_dic[folder+'/'+file]: 
                tf = word_dic[folder +'/'+file][word]
                
                tf_idf_dic[folder +'/'+file][word] = tf * np.log10((D/idf_dic[word]))
    return tf_idf_dic                
    #TODO
    pass


# In[ ]:


# tf_idf(os.getcwd(), use_adv_tokenization=False)


# ## 1.3: Experiment 1
# 
# In this experiment, you will implement a simple (multiclass) Naive Bayes Classifier. That is, you will use the training documents to learn a model and then compute the most likely label among the 20 labels for a new document, using the Naive Bayes assumption. You will do it for all three document representations defined earlier. 
# 
# Note that, using Bayes rule, your prediction should be:
# 
# \begin{equation*}
# \hat{y}_d = \underset{y}{\mathrm{argmax}} P(d | y) \cdot P(y),
# \end{equation*}
# where $y$ ranges over the $20$ candidate labels.
# 
# Since we are using the Naive Bayes model, that is, we assume the independence of the features (words in a document) given the label (document type), we get: 
# 
# \begin{equation*}
# P(d | y) = \Pi_{v \in d}  P(v | y)
# \end{equation*}
# where $v$ is a word in document $d$ and $y$ is the label of document $d$. 
# 
# The question is now how to estimate the coordinate-wise conditional probabilities $P(v|y)$. We have suggested to use three different representations of the document, but in all cases, estimate this conditional probability as: 
# \begin{equation*}
# P(v | y) = \frac{\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',v]}{\sum_{w \in \mathcal{V}}\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',w]}
# \end{equation*}
# Notice that in class we used the same estimation for the <b>B-BOW</b> model, and here we generalize it to the other two representations. 
# 
# To do <b>$k$-laplace smoothing</b> (choose a suitable value of $k$), modify the above formula as follows: (You can think of 1-laplace smoothing as adding another document of class $y$ which contains all the words in the vocabulary.)
# 
# \begin{equation*}
# P(v | y) = \frac{\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',v] + k}{(\sum_{w \in \mathcal{V}}\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',w]) + |\mathcal{V}|k}.
# \end{equation*}
# You will also need to estimate the prior probability of each label, as:
# \begin{equation*}
# P(y) = \frac{\text{# of documents in }\mathcal{D} \text{ with label $y$}}{|\mathcal{D}|}
# \end{equation*}
# 
# In this experiment you will run the Naive Bayes algorithm for both datasets and all three the representations and provide an analysis of your results. Use <b>accuracy</b> to measure the performance.
# 
# </b>Important Implementation Detail:</b> Since we are multiplying probabilities, numerical underflow may occur while computing the resultant values. To avoid this, you should do your computation in <b>log space</b>. This means that you should take the log of the probability.
# <b>Also feel free to add in your own optional parameters for optimization purposes!</b>

# In[7]:


def naive_bayes(document, train_dir, model,prob_v_for_y = None, all_unique_word_counts =None,
                total_unique_words=None,prob_y=None,docs_all_label=None,doc_file =None, k=0.0):
    '''
    Uses naive bayes to predict the label of the given document.
    
    :param document: the path to the document whose label you need to predict
    :param train_dir: the directory with the training data
    :param model: the data representation model
    :param k: the k parameter in k-laplace smoothing
    :return: a tuple containing the predicted label of the document and log-probability of that label
    '''
    lemmatizer = WordNetLemmatizer()
    train_dic = {}
    all_unique_word_counts = {}
    total_unique_words  = {}
       
    train_dic = model
    D =len(train_dic)   

    if(prob_y is None or docs_all_label is None or doc_file is None or prob_v_for_y is None or
       all_unique_word_counts is None or total_unique_words is None): 
        prob_y = {}
        prob_v_for_y = {}
        docs_all_label = {}
        docs_all_label,doc_file =docs_each_label(train_dir)
        prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)

        prob_y = prob_y_try(train_dir, train_dic)    
        

                    
    
    stopW = stopwords.words('english')
    prob_d_y ={}                                                                     
    for folder in docs_all_label.keys():    
        prob_d_y[folder] =1
        with open(document , 'r', errors = 'ignore') as f:
            for line in f:
                words = [lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW]
                for word in words:
                    if word not in prob_v_for_y[folder]:
                        prob_v_for_y[folder][word] =  k/(total_unique_words[folder] + k*len(all_unique_word_counts[folder]))
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word]) 
                    else:
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word])   
    
    #prediction
    y_pred={}                        
    for folder in docs_all_label.keys(): 
        y_pred[folder] = prob_d_y[folder] + math.log10(prob_y[folder])

        
    y_max = max(y_pred, key=y_pred.get)  
    y_pred_final = y_pred[y_max]                                
    return y_max,y_pred_final                                                 
    
    #TODO
    pass


# In[8]:


def naive_bayes_X(document, train_dir, model, prob_v_for_y, all_unique_word_counts,
                total_unique_words,prob_y,docs_all_label,doc_file,k=0.0):
    '''
    Uses naive bayes to predict the label of the given document.
    
    :param document: the path to the document whose label you need to predict
    :param train_dir: the directory with the training data
    :param model: the data representation model
    :param k: the k parameter in k-laplace smoothing
    :return: a tuple containing the predicted label of the document and log-probability of that label
    '''
    lemmatizer = WordNetLemmatizer()
    train_dic = {}
    
         
    train_dic = model
      

    
    stopW = stopwords.words('english')
    prob_d_y ={}                                                                     
    for folder in docs_all_label.keys():    
        prob_d_y[folder] =1
        with open(document , 'r', errors = 'ignore') as f:
            for line in f:
                words = [lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW]
                for word in words:
                    if word not in prob_v_for_y[folder]:
                        prob_v_for_y[folder][word] =  k/(total_unique_words[folder] + k*len(all_unique_word_counts[folder]))
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word]) 
                    else:
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word])   
    
    #prediction
    y_pred={}                        
    for folder in docs_all_label.keys(): 
        y_pred[folder] = prob_d_y[folder] + math.log10(prob_y[folder])

        
    y_max = max(y_pred, key=y_pred.get)  
    y_pred_final = y_pred[y_max]                                
    return y_max,y_pred_final                                                 
    
    #TODO
    pass


# In[14]:


def prob_v_for_y_try(train_dir, model, k):
    train_dic = model
    docs_all_label,doc_file =docs_each_label(train_dir)
     #numerator p(v/y)   
    all_unique_word_counts = {}    
    for folder in docs_all_label.keys():    
        all_unique_word_counts[folder]={}
        for file in doc_file[folder]:
            for word in train_dic[folder+'/'+file]: 
                if word not in all_unique_word_counts[folder]:
                    all_unique_word_counts[folder][word] = train_dic[folder+'/'+file][word]
                else:
                    all_unique_word_counts[folder][word] += train_dic[folder+'/'+file][word]
                    
    #denominator p(v/y)                
    total_unique_words ={}    
    for folder in docs_all_label.keys():    
        total_unique_words[folder] =0 
        for word in all_unique_word_counts[folder]:
            total_unique_words[folder] += all_unique_word_counts[folder][word]
    
     #p(v/y)     
    prob_v_for_y = {}
    for folder in docs_all_label.keys():
        prob_v_for_y[folder] ={}
        for file in doc_file[folder]:
            for word in train_dic[folder+'/'+file]: 
                
                prob_v_for_y[folder][word] = (all_unique_word_counts[folder][word] + k)/(total_unique_words[folder] + k*len(all_unique_word_counts[folder]))
    return prob_v_for_y, all_unique_word_counts, total_unique_words


# In[15]:


def prob_y_try(train_dir, model):
    train_dic = model
    D =len(train_dic) 
    prob_y = {}
    docs_all_label = {}
    docs_all_label,doc_file =docs_each_label(train_dir)
    for label in docs_all_label.keys():
        prob_y[label] = docs_all_label[label]/D
    return prob_y    


# In[16]:


#function to get count of docs per label and create a dic mapping of docs in each label
def docs_each_label(train_dir):
    docs_label = {}
    doc_file ={}
    for folder in os.listdir(train_dir):
        docs_label[folder] = 0
        lis_f = []
        for files in os.listdir(train_dir + '/' + folder):
            docs_label[folder] += 1
            lis_f.append(files)
        doc_file[folder] = lis_f    
    return docs_label, doc_file        


# In[17]:


def test_naive_accuracy(test_dir, train_dir):
    tot_right = 0
    
    tot_files  = 0
    for folder in os.listdir(test_dir):
        tot_lab = 0
        for file in os.listdir(test_dir + '/'+folder):
            pred, log_prob = naive_bayes_X(test_dir + '/'+folder +'/'+file, train_dir, model, prob_v_for_y, all_unique_word_counts, total_unique_words, 
                prob_y, docs_all_label, doc_file, k=0.3)

            tot_files +=1

            if pred == folder:
                tot_right+=1
                tot_lab+=1

        print(folder)
        print(tot_lab)
        print(len(os.listdir(test_dir + '/'+folder)))
        print(tot_lab/(len(os.listdir(test_dir + '/'+folder))))
    print(tot_files)     
    print(tot_right)
    accuracy = tot_right/ tot_files
    return accuracy


# In[13]:


# model =b_bow(os.getcwd())


# In[65]:


# k=0.2
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# # prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)
# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[72]:


# k=0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)
# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[67]:


# k=1
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)
# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[69]:


# k=2
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)
# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[62]:


# train_dir = os.getcwd()
# k = 0.5
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)
# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[75]:


# model2 =c_bow(os.getcwd())


# In[76]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model2, k)
# prob_y = prob_y_try(train_dir, model2)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[77]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[78]:


# model3 =tf_idf(os.getcwd())


# In[79]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model3, k)
# prob_y = prob_y_try(train_dir, model3)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[80]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[96]:


# model =b_bow(os.getcwd())
# #without stopwords


# In[97]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[98]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[104]:


# model2 =c_bow(os.getcwd())
# print(os.getcwd())


# In[105]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model2, k)
# prob_y = prob_y_try(train_dir, model2)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[106]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[110]:


# model3 =tf_idf(os.getcwd())


# In[111]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model3, k)
# prob_y = prob_y_try(train_dir, model3)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[112]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',os.getcwd())


# In[81]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\train')


# In[83]:


# model3 =tf_idf(os.getcwd())


# In[84]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model3, k)
# prob_y = prob_y_try(train_dir, model3)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[85]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',os.getcwd())


# In[86]:


# model2 =c_bow(os.getcwd())


# In[89]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model2, k)
# prob_y = prob_y_try(train_dir, model2)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[90]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',os.getcwd())


# In[91]:


# model =b_bow(os.getcwd())


# In[92]:


# train_dir = os.getcwd()
# k = 0.3
# prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try(train_dir, model, k)
# prob_y = prob_y_try(train_dir, model)  
# docs_all_label,doc_file =docs_each_label(train_dir)


# In[93]:


# test_naive_accuracy(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',os.getcwd())


# In[94]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\train')


# ## 1.4: Experiment 2
# 
# In this experiment, you will use Semi-Supervised Learning to do document classification. 
# The underlying algorithm you will use is Naive Bayes with the <b>B-BOW</b> representation, as defined in Experiment 1.
# 
# See write-up for more details.

# ### 1.4.1: Top-m
# 
# Filter the top-m instances and augment them to the labeled data.

# In[18]:


def filter_by_mtop(predictions, m=0):
    '''
    Filter the top-m instances and augment them to the labeled data.
    
    :param predictions: dictionary mapping documents to tuples of (predicted label, log-probability)
                        i.e. {'doc1':('label1', prob1), 'doc2':('label2', prob2)}
    :param k: the number of predictions to augment to the labeled data
    :return: a tuple of (dictionary mapping documents to labels of new supervised data,
        
        list of documents in unsupervised data)
    '''
    
    predict = list(predictions.values())

    docs = list(predictions.keys())
    tot_list = list(zip(predict, docs))

    tot_list.sort(key= lambda temp:temp[0][1], reverse = True)

    top_m_dic = tot_list[0:m]
    left_dic = tot_list[m:]
    
    left_docs =[]
    for i in range(len(left_dic)):
        left_docs.append(left_dic[i][1])
        
    final_dic = {}
    
    for i in range(len(top_m_dic)):
        final_dic[top_m_dic[i][1]] = top_m_dic[i][0][0]
    #TODO
    
    return final_dic, left_docs
    pass


# ### 1.4.2: Threshold
# 
# Set a threshold. Augment those instances to the labeled data with confidence strictly higher than this threshold. You have to be careful here to make sure that the program terminates. If the confidence is never higher than the threshold, then the procedure will take forever to terminate. You can choose to terminate the program if there are 5 consecutive iterations where no confidence exceeded the threshold value. 

# In[19]:


def filter_by_threshold(predictions, threshold=0):
    '''
    Augment instances to the labeled data with confidence strictly higher than the given threshold.
    
    :param predictions: dictionary mapping documents to tuples of (predicted label, log-probability)
                        i.e. {'doc1':('label1', prob1), 'doc2':('label2', prob2)}
    :param threshold: the threshold to filter by
    :return: a tuple of (dictionary mapping documents to labels of new supervised data,
                         list of documents in unsupervised data)
    '''
    
    
    predict = list(predictions.values())
    print(predict)
    docs = list(predictions.keys())
    tot_list = list(zip(predict, docs))
    tot_list.sort(key= lambda temp:temp[0][1], reverse = True)
    
    
    left_docs =[]
    final_dic = {} 
    
    for i in range(len(tot_list)):
        if tot_list[i][0][1]> threshold:
            final_dic[tot_list[i][1]] = tot_list[i][0][0]
        else:
            left_docs.append(tot_list[i][1])
    #TODO
    
    return final_dic, left_docs
    #TODO
    pass


# In[20]:


import random


# In[23]:


def prob_v_for_y_try_X(train_S, model, k):
    train_dic = model
    docs_all_label,doc_file =docs_each_label_X(train_S)
     #numerator p(v/y)   
    all_unique_word_counts = {}    
    for folder in docs_all_label.keys():    
        all_unique_word_counts[folder]={}
        for file in doc_file[folder]:
            for word in train_dic[file]: 
                if word not in all_unique_word_counts[folder]:
                    all_unique_word_counts[folder][word] = train_dic[file][word]
                else:
                    all_unique_word_counts[folder][word] += train_dic[file][word]
                    
    #denominator p(v/y)                
    total_unique_words ={}    
    for folder in docs_all_label.keys():    
        total_unique_words[folder] =0 
        for word in all_unique_word_counts[folder]:
            total_unique_words[folder] += all_unique_word_counts[folder][word]
    
     #p(v/y)     
    prob_v_for_y = {}
    for folder in docs_all_label.keys():
        prob_v_for_y[folder] ={}
        for file in doc_file[folder]:
            for word in train_dic[file]: 
                
                prob_v_for_y[folder][word] = (all_unique_word_counts[folder][word] + k)/(total_unique_words[folder] + k*len(all_unique_word_counts[folder]))
    return prob_v_for_y, all_unique_word_counts, total_unique_words


# In[24]:


def prob_y_try_X(train_S, model):
    train_dic = model
    D =len(train_dic) 
    prob_y = {}
    docs_all_label = {}
    docs_all_label,doc_file =docs_each_label_X(train_S)
    for label in docs_all_label.keys():
        prob_y[label] = docs_all_label[label]/D
    return prob_y    


# In[25]:


#function to get count of docs per label and create a dic mapping of docs in each label, s= sx, sy-doc(label +/+file):label

def docs_each_label_X(train_S):

    new_dict = {}
    docs_label ={}
    doc_file = {}
    for k, v in train_S.items():
        new_dict.setdefault(v, []).append(k)    
   
    for label in new_dict.keys():
        docs_label[label] =len(new_dict.get(label)) 
        doc_file[label] = new_dict.get(label)
   
    return docs_label, doc_file        


# In[26]:


def naive_bayes_Y(document, train_S, model,prob_y,prob_v_for_y, all_unique_word_counts,
                  total_unique_words,docs_all_label, doc_file,k=0.0):
    '''
    Uses naive bayes to predict the label of the given document.
    
    :param document: the path to the document whose label you need to predict
    :param train_dir: the directory with the training data
    :param model: the data representation model
    :param k: the k parameter in k-laplace smoothing
    :return: a tuple containing the predicted label of the document and log-probability of that label
    '''
    lemmatizer = WordNetLemmatizer()
    train_dic = {}

    
         
    train_dic = model
      

    
    stopW = stopwords.words('english')
    prob_d_y ={}                                                                     
    for folder in docs_all_label.keys():    
        prob_d_y[folder] =1
        with open(document , 'r', errors = 'ignore') as f:
            for line in f:
                words = [lemmatizer.lemmatize(word) for word in line.lower().split() if word not in stopW]
                for word in words:
                    if word not in prob_v_for_y[folder]:
                        prob_v_for_y[folder][word] =  k/(total_unique_words[folder] + k*len(all_unique_word_counts[folder]))
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word]) 
                    else:
                        prob_d_y[folder] += math.log10(prob_v_for_y[folder][word])   
    
    #prediction
    y_pred={}                        
    for folder in docs_all_label.keys(): 
        y_pred[folder] = prob_d_y[folder] + math.log10(prob_y[folder])

        
    y_max = max(y_pred, key=y_pred.get)  
    y_pred_final = y_pred[y_max]                                
    return y_max,y_pred_final                                                 
    
    #TODO
    pass


# In[27]:


def docs_each_label2(train_S):
    pri_prob = {}

    for folder in train_S.keys():
        pri_prob[folder] = 0
   
        for files in train_S[folder]:
            pri_prob[folder] += 1
            
    return pri_prob  


# In[31]:


def semi_supervised(test_dir, train_dir, filter_function, p=0.05):
    '''
    Semi-supervised classifier with Naive Bayes and B-BoW representation.
    
    :param test_dir: directory with the test data
    :param train_dir: directory with the train data
    :param filter_function: the function we will use to filter data for augmenting the labeled data
    :param p: the proportion of the training data that starts off as labeled
    :return: a tuple containing the model trained on the supervised data, and the supervised dataset
             where the model is represented by a dict (same as B-BoW),
             and S is a mapping from documents to labels
    '''
    train_S ={}
    train_U ={}
    train_full ={}
    new_train ={}
    path = train_S
    
    Sx_arr = []
    
    print(train_dir)
    
    model_dic = {}
    S_train ={}
    U_all_files  =[]
    
    train_dic = b_bow(train_dir)
    i = int(len(train_dic)*p)
    Sx = random.sample(train_dic.keys(),i)

    for element in Sx:
        Sx_arr = element.split('/')
        Sy = Sx_arr[0]
        train_S[element] = Sy
    print(len(train_S))
    
    for file in train_dic.keys():
        if file not in train_S.keys():
            U_all_files.append(file)
        
    print("unsupervised")        


    print(len(U_all_files)) 
  
    doc_arr = []        
    pred_dic = {}
    acc = []
    while(len(U_all_files)!=0):
        print("check new u")
        print(len(U_all_files))
        b_bow_dic = {}
        for doc in train_S.keys():
            for file in train_dic.keys():
                if doc== file:
                    b_bow_dic[doc] = train_dic.get(file)
        print(len(b_bow_dic))   
        print("new dic")

        k=0.5
        prob_v_for_y, all_unique_word_counts, total_unique_words = prob_v_for_y_try_X(train_S, b_bow_dic, k)
        prob_y = prob_y_try_X(train_S, b_bow_dic)  
        docs_all_label,doc_file =docs_each_label_X(train_S)
        
        print(len(train_S))
        print(len(b_bow_dic))
        pred_dic = {}
        for doc in U_all_files:

                
            pred_label, pred_prob= naive_bayes_Y(train_dir +'/'+ doc, train_S,b_bow_dic,prob_y,prob_v_for_y,
                                                     all_unique_word_counts, total_unique_words,docs_all_label,doc_file,k=0.5)
            
#             
            pred_dic[doc] = pred_label, pred_prob

        
        print(len(pred_dic))
        print("new pred")
        
        accuracy_c =  test_acc_semi(test_dir,train_S,b_bow_dic,prob_y,prob_v_for_y,all_unique_word_counts,total_unique_words,docs_all_label,doc_file,k=0.5)
        acc.append(accuracy_c)
        
        S_from_U, U_left = filter_function(pred_dic)
        
    # s: doc1:label, u is only documents, train_s is label: list(docs)
        train_S.update(S_from_U)
        
        print("new S")
        print(len(train_S))
  
        U_all_files = U_left 
        if(len(S_from_U) is 0):
            return b_bow_dic, train_S
        print("left")
        print(len(U_all_files))

    b_bow_dic = {}
    for doc in train_S.keys():
        for file in train_dic.keys():
            if doc== file:
                b_bow_dic[doc] = train_dic.get(file)
    return b_bow_dic, train_S



    #TODO
    pass


# In[123]:


def test_acc_semi(test_dir, train_S,b_bow_dic,prob_y,prob_v_for_y,all_unique_word_counts,
                                        total_unique_words,docs_all_label,doc_file,k=0.5):
    tot_right = 0
    
    tot_files  = 0
    for folder in os.listdir(test_dir):
        tot_lab = 0
        for file in os.listdir(test_dir + '/'+folder):
            pred, log_prob = naive_bayes_Y(test_dir + '/'+folder +'/'+file, train_S, b_bow_dic, prob_y, prob_v_for_y, 
                                           all_unique_word_counts, total_unique_words, docs_all_label, doc_file, k=0.3)

            tot_files +=1

            if pred == folder:
                tot_right+=1
                tot_lab+=1

        print(folder)

        print(tot_lab/(len(os.listdir(test_dir + '/'+folder))))
    print(tot_files)     
    print(tot_right)
    accuracy = tot_right/ tot_files
    return accuracy


# In[30]:


# # Example of how to call the semi_supervised function:
# b_t,s_t,a_t = semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\train', filter_by_mtop, p=0.50)
# print(b_t)
# print(s_t)
# print(a_t)


# In[32]:


# b_t5,s_t5,a_t5 = semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\train', filter_by_mtop, p=0.10)
# print(b_t5)
# print(s_t5)
# print(a_t5)


# In[33]:


# print(a_t5)


# In[34]:


# b_t6,s_t6,a_t6 = semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-bydate-rm-metadata\train', filter_by_mtop, p=0.05)
# print(b_t6)
# print(s_t6)
# print(a_t6)


# In[ ]:


# # Example of how to call the semi_supervised function:
# b_t2, s_t2,a_t2 = semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\train', filter_by_mtop, p=0.50)
# print(b_t2)
# print(s_t2)
# print(a_t2)


# In[ ]:


# # Example of how to call the semi_supervised function:
# b_t3, s_t3,a_t3 = 
# semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\train', filter_by_mtop, p=0.10)
# print(b_t3)
# print(s_t3)
# print(a_t3)


# In[ ]:


# # Example of how to call the semi_supervised function:
# b_t4, s_t4,a_t4 = 
# semi_supervised(r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\test',r'C:\Users\Dell1\Desktop\CIS519\20news-random-rm-metadata\train', filter_by_mtop, p=0.05)
# print(b_t4)
# print(s_t4)
# print(a_t4)


# ## 1.5: (Optional: Extra Credit) Experiment 3
# 
# For experiment 2, initialize as suggested but, instead of the filtering, run EM. That is, for each data point in U, label it fractionally -- label data point d with label $l$, that has weight $p(l|d)$. Then, add all the (weighted) examples to S. Now use Naive Bayes to learn again on the augmented data set (but now each data point has a weight! That is, when you compute $P(f[d,i]|y)$, rather than simply counting all the documents that have label $y$, now <b>all</b> the documents have "fractions" of the label $y$), and use the model to relabel it (again, fractionally, as defined above.) Iterate, and determine a stopping criteria. 

# In[87]:


# Experiment 3


# ## Running your experiments
# 
# Use the cell below to run your code.
# 
# 
# Remember to comment it all out before submitting!
