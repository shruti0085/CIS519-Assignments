
# coding: utf-8

# In[10]:


import sklearn
import numpy
import os
import math


# In[11]:


def calc_f1(true_labels,predicted_labels):
    # true_labels - list of true labels (1/-1)
    # predicted_labels - list of predicted labels (1/-1)
    # return precision, recall and f1
    actual_positive = []
    predicted_positive = []
    count_true_positive= 0
    
    for i in range(len(true_labels)):
        
        if true_labels[i] == 1 and predicted_labels[i] == 1:
             count_true_positive = count_true_positive+1
            
    for k in range(len(predicted_labels)):
        if predicted_labels[k] == 1:
            predicted_positive.append(predicted_labels[k])
    for m in range(len(true_labels)):
        if true_labels[m] == 1:
            actual_positive.append(true_labels[m])
  
        
            
    precision = 0.0 
    recall = 0.0
    f1 = 0.0
    precision = count_true_positive/len(predicted_positive)
    recall = count_true_positive/len(actual_positive)
    f1 = (2*precision*recall)/(precision + recall)
    return precision, recall, f1


# In[12]:


class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged = False, eta = 1, alpha = 1):
        # Algorithm values can be Perceptron, Winnow, Adagrad, Perceptron-Avg, Winnow-Avg, Adagrad-Avg, SVM
        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        self.features = {feature for xi in x_train for feature in xi.keys()}
        
        if algorithm == 'Perceptron':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta*yi*value
                        self.w['bias'] = self.w['bias'] + eta*yi
                        
                    
                    

        elif algorithm == 'Winnow':
            self.w, self.w['bias'] = {feature:1.0 for feature in self.features}, -len(self.features)
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature]*(alpha**(yi*value))
                            
                        
            
        
        elif algorithm == 'Adagrad':
            #Initialize w, bias, gradient accumulator
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.grad = {feature:0.0 for feature in self.features}
            self.grad['bias'] = 0.0
            self.sum_grad_square = {feature:0.0 for feature in self.features}
            self.sum_grad_square['bias'] = 0.0
            
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    product_w = sum([self.w[feature]*value for feature, value in xi.items()]) 
                    
                    if yi*(product_w + self.w['bias']) <= 1:
                         
                        for feature, value in xi.items():
                            self.grad[feature] = (yi*value)*-1
                            self.sum_grad_square[feature] = self.sum_grad_square[feature] + ((self.grad[feature])**2)
                            self.w[feature] = self.w[feature] + (eta*yi*value)/((self.sum_grad_square[feature])**0.5)
                        self.grad['bias'] = yi*-1 
                        self.sum_grad_square['bias'] = self.sum_grad_square['bias'] + ((self.grad['bias'])**2)
                        self.w['bias'] = self.w['bias'] + (eta*yi*1)/((self.sum_grad_square['bias'])**0.5)
                    
        
   
        elif algorithm == 'Perceptron-Avg':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.cum_delta = {feature:0.0 for feature in self.features}
            self.cum_delta['bias'] = 0.0
            c=0
            # initialize another counter (c_total)
            c_total = 0
            new = iterations*len(x_train)
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                   
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta*yi*value    
                            self.cum_delta[feature] = self.cum_delta[feature] + (eta*yi*value)*c_total

                        self.w['bias'] = self.w['bias'] + eta*yi
                        self.cum_delta['bias'] = self.cum_delta['bias'] + (eta*yi)*c_total
                        c_total = c_total + c
                        c= 1
                        
                    else:
                        c= c+1
                    
            for feature in self.features:
                 self.w[feature] = self.w[feature] - (self.cum_delta[feature]/c_total)
            self.w['bias'] = self.w['bias'] - (self.cum_delta['bias']/c_total)
            
           
        
        elif algorithm == 'Winnow-Avg':
            self.w, self.w['bias'] = {feature:1.0 for feature in self.features}, -len(self.features)
            self.cum_w = {feature:0.0 for feature in self.features}
            c=0
            c_total = 0
            new = iterations*len(x_train)
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature in self.features:
                            if feature in xi:
                                self.w[feature] = self.w[feature]*(alpha**(yi*xi[feature]))
                            self.cum_w[feature] = self.cum_w[feature] + (self.w[feature]*c)
                        c_total = c_total + c
                        c=1
                    else:
                        c=c+1
            for feature in self.features:
                self.w[feature] = (self.cum_w[feature])/c_total
            
       
        elif algorithm == 'Adagrad-Avg':
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.grad = {feature:0.0 for feature in self.features}
            self.grad['bias'] = 0.0
            self.sum_grad_square = {feature:0.0 for feature in self.features}
            self.sum_grad_square['bias'] = 0.0
            self.cum_w, self.cum_w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            total = iterations*len(x_train)
            c=0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    product_w = sum([self.w[feature]*value for feature, value in xi.items()]) 
                    
                    if yi*(product_w + self.w['bias']) <= 1:

                        for feature in self.features:
                            if feature in xi:
                                self.grad[feature] = (yi*xi[feature])*-1
                                self.sum_grad_square[feature] = self.sum_grad_square[feature] + ((self.grad[feature])**2)
                                self.w[feature] = self.w[feature] + (eta*yi*xi[feature])/((self.sum_grad_square[feature])**0.5)
                            self.cum_w[feature] = self.cum_w[feature] + ((self.w[feature])*c)
                        self.grad['bias'] = yi*-1 
                        self.sum_grad_square['bias'] = self.sum_grad_square['bias'] + ((self.grad['bias'])**2)
                        self.w['bias'] = self.w['bias'] + (eta*yi*1)/((self.sum_grad_square['bias'])**0.5)
                        self.cum_w['bias'] = self.cum_w['bias'] + ((self.w['bias'])*c)
                        c=1
                    else:
                        c=c+1
            for feature in self.features:
                 self.w[feature] = self.cum_w[feature]/total
            self.w['bias'] = self.cum_w['bias']/total   
            
            

        elif algorithm == 'SVM':
            from sklearn.svm import LinearSVC
            from sklearn.datasets import make_classification
            from sklearn.feature_extraction import DictVectorizer

           
            self.vectorizer = DictVectorizer()
            x_vector = self.vectorizer.fit_transform(x_train)
            clf  = LinearSVC(penalty='l2', loss = 'hinge')
            self.svm = clf.fit(x_vector, y_train) 
                    
                    
            
        
        else:
            print('Unknown algorithm')
                
    def predict(self, x):
        s = sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1
    
    def predict_SVM(self, x):
        x = self.vectorizer.transform([x])
        return self.svm.predict(x)[0]


# In[13]:


#Parse the real-world data to generate features, 
#Returns a list of tuple lists
def parse_real_data(path):
    #List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path+filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# In[14]:


#Returns a list of labels
def parse_synthetic_labels(path):
    #List of tuples for each sentence
    labels = []
    with open(path+'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# In[15]:


#Returns a list of features
def parse_synthetic_data(path):
    #List of tuples for each sentence
    data = []
    with open(path+'x.txt') as file:
        features = []
        for line in file:
            #print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


# In[60]:



def extract_features_train(news_train_data):
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded)-3):
            news_train_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-1='+str(padded[i-1][0])
            feat2 = 'w+1='+str(padded[i+1][0])
            feat3 = 'w-2='+str(padded[i-2][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-3='+str(padded[i-3][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            train_features.update(feats)
            feats = {feature:1 for feature in feats}
            news_train_x.append(feats)

    return train_features, news_train_x, news_train_y


# In[61]:


def extract_features_dev(news_dev_data, train_features):
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_dev_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-1='+str(padded[i-1][0])
            feat2 = 'w+1='+str(padded[i+1][0])
            feat3 = 'w-2='+str(padded[i-2][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-3='+str(padded[i-3][0])
            feat6 = 'w+3='+str(padded[i+3][0]) 
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    return news_dev_x, news_dev_y


# In[7]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519')


# In[8]:


# print('Loading data...')

# ##Load data from folders.
# # #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')

# # #Load dense synthetic data
# syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/Train/')
# syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/Train/')
# syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_no_noise_data = parse_synthetic_data('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_dev_no_noise_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev_no_noise/')
   
# # #Load sparse synthetic data
# syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/Train/')
# syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Train/')
# syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/Dev/')
# syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Dev/')

# print('Data Loaded.')


# In[9]:


# # # Convert to sparse dictionary representations.

# print('Converting Synthetic data...')
# syn_dense_train = zip(*[({'x'+str(i): syn_dense_train_data[j][i]
#      for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1}, syn_dense_train_labels[j]) 
#          for j in range(len(syn_dense_train_data))])
# syn_dense_train_x, syn_dense_train_y = syn_dense_train
# syn_dense_dev = zip(*[({'x'+str(i): syn_dense_dev_data[j][i]
#      for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1}, syn_dense_dev_labels[j]) 
#          for j in range(len(syn_dense_dev_data))])
# syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

# # ## Similarly add code for the dev set with no noise and sparse data

# print('Done')


# In[13]:


# # # Convert to sparse dictionary representations.

# print('Converting Synthetic-Dev-No-noise data...')
# syn_dense_train = zip(*[({'x'+str(i): syn_dense_train_data[j][i]
#      for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1}, syn_dense_train_labels[j]) 
#          for j in range(len(syn_dense_train_data))])
# syn_dense_train_x, syn_dense_train_y = syn_dense_train
# syn_dense_dev_no_noise = zip(*[({'x'+str(i): syn_dense_dev_no_noise_data[j][i]
#      for i in range(len(syn_dense_dev_no_noise_data[j])) if syn_dense_dev_no_noise_data[j][i] == 1}, syn_dense_dev_no_noise_labels[j]) 
#          for j in range(len(syn_dense_dev_no_noise_data))])
# syn_dense_dev_no_noise_x, syn_dense_dev_no_noise_y = syn_dense_dev_no_noise

# # ## Similarly add code for the dev set with no noise and sparse data

# print('Done')


# In[24]:


# print('Converting Synthetic-Sparse data...')
# syn_sparse_train = zip(*[({'x'+str(i): syn_sparse_train_data[j][i]
#      for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1}, syn_sparse_train_labels[j]) 
#          for j in range(len(syn_sparse_train_data))])
# syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
# syn_sparse_dev = zip(*[({'x'+str(i): syn_sparse_dev_data[j][i]
#      for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1}, syn_sparse_dev_labels[j]) 
#          for j in range(len(syn_sparse_dev_data))])
# syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev
# print('Done')


# In[16]:


# syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')
# syn_dense_test_x=[({'x'+str(i): syn_dense_test_data[j][i]
#         for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1})
#             for j in range(len(syn_dense_test_data))]


# In[17]:


# syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/Test/')
# syn_sparse_test_x=[({'x'+str(i): syn_sparse_test_data[j][i]
#         for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1})
#             for j in range(len(syn_sparse_test_data))]


# In[62]:


# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# def extract_features_email_dev(email_dev_data, train_features):
#     email_dev_x = []
#     email_dev_y =[]
    
#     for sentence in email_dev_data:
#         padded = sentence[:]
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         for i in range(3,len(padded)-3):
#             email_dev_y.append(1 if padded[i][1]=='I' else -1)    
#             feat1 = 'w-1='+str(padded[i-1][0])
#             feat2 = 'w+1='+str(padded[i+1][0])
#             feat3 = 'w-2='+str(padded[i-2][0])
#             feat4 = 'w+2='+str(padded[i+2][0])
#             feat5 = 'w-3='+str(padded[i-3][0])
#             feat6 = 'w+3='+str(padded[i+3][0])
#             feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
#             feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
#             feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
#             feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
#             feats = {feature:1 for feature in feats if feature in train_features}
#             email_dev_x.append(feats)
#     return email_dev_x, email_dev_y            


# In[63]:


# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# def extract_features_email_test(email_test_data, train_features):
#     email_test_x = []
     
#     for sentence in email_test_data:
#         padded = sentence[:]
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         for i in range(3,len(padded)-3):
#             feat1 = 'w-1='+str(padded[i-1][0])
#             feat2 = 'w+1='+str(padded[i+1][0])
#             feat3 = 'w-2='+str(padded[i-2][0])
#             feat4 = 'w+2='+str(padded[i+2][0])
#             feat5 = 'w-3='+str(padded[i-3][0])
#             feat6 = 'w+3='+str(padded[i+3][0])
#             feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
#             feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
#             feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
#             feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
#             feats = {feature:1 for feature in feats if feature in train_features}
#             email_test_x.append(feats)
#     return email_test_x


# In[64]:


# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# def extract_features_news_test(news_test_data, train_features):
#     news_test_x =[]
#     for sentence in news_test_data:
#         padded = sentence[:]
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.insert(0, ('SSS', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         padded.append(('EEE', None))
#         for i in range(3,len(padded)-3):      
#             feat1 = 'w-1='+str(padded[i-1][0])
#             feat2 = 'w+1='+str(padded[i+1][0])
#             feat3 = 'w-2='+str(padded[i-2][0])
#             feat4 = 'w+2='+str(padded[i+2][0])
#             feat5 = 'w-3='+str(padded[i-3][0])
#             feat6 = 'w+3='+str(padded[i+3][0])
#             feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
#             feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
#             feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
#             feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
#             feats = {feature:1 for feature in feats if feature in train_features}
#             news_test_x.append(feats)
#     return news_test_x


# In[21]:


# # # Feature extraction
# # # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)


# In[23]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_email_dev(email_dev_data, train_features)

# y_predict_email_dev = []

# for i in range(len(email_dev_y)):
#     y_predict_email_dev.append(p.predict_SVM(email_dev_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(email_dev_y,y_predict_email_dev)
# print('Precision,Recall and f1 for Enron dev',precision_Con, recall_Con, f1_Con)


# In[25]:


# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_dev(email_dev_data, train_features)


# In[26]:


# p_per = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)

# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y=extract_features_dev(news_dev_data, train_features)
# y_predict_news_dev_per = []

# for i in range(len(news_dev_y)):
#     y_predict_news_dev_per.append(p_per.predict(news_dev_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(news_dev_y,y_predict_news_dev_per)
# print('Precision,Recall and f1 with percep for Conll dev',precision_Con, recall_Con, f1_Con)


# In[27]:


# p_per = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)

# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_email_dev(email_dev_data, train_features)
# y_predict_email_dev_per = []

# for i in range(len(email_dev_y)):
#     y_predict_email_dev_per.append(p_per.predict(email_dev_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(email_dev_y,y_predict_email_dev_per)
# print('Precision,Recall and f1 with percep for Enron',precision_Con, recall_Con, f1_Con)


# In[28]:


# p_per = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)

# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)


# y_predict_news_train_per = []

# for i in range(len(news_train_y)):
#     y_predict_news_train_per.append(p_per.predict(news_train_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(news_train_y,y_predict_news_train_per)
# print('Precision,Recall and f1 with percep for Conll train',precision_Con, recall_Con, f1_Con)


# In[29]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# y_predict_news_dev_svm = []

# for i in range(len(news_dev_y)):
#     y_predict_news_dev_svm.append(p.predict_SVM(news_dev_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(news_dev_y,y_predict_news_dev_svm)
# print('Precision,Recall and f1 with svm for Conll dev',precision_Con, recall_Con, f1_Con)


# In[30]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)

# y_predict_news_train_svm = []

# for i in range(len(news_train_y)):
#     y_predict_news_train_svm.append(p.predict_SVM(news_train_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(news_train_y,y_predict_news_train_svm)
# print('Precision,Recall and f1 with SVM for Conll train',precision_Con, recall_Con, f1_Con)


# In[35]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_test_x = extract_features_email_test(email_test_data, train_features)
# file_news = open('svm-enron.txt', 'w')
# y_label_email = []
# for i in range(len(email_test_x)):
#     y_label_email.append(p.predict_SVM(email_test_x[i]))
# for label in y_label_email:    
#     if label == 1:
#         file_news.write('I\n')
#     else:
#         file_news.write('O\n')


# In[39]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)

# news_test_x = extract_features_news_test(news_test_data, train_features)
# file_news = open('svm-conll.txt', 'w')
# y_label_news = []
# for i in range(len(news_test_x)):
#     y_label_news.append(p.predict_SVM(news_test_x[i]))
# for label in y_label_news:    
#     if label == 1:
#         file_news.write('I\n')
#     else:
#         file_news.write('O\n')


# In[34]:


# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/Test/')
# syn_sparse_test_x=[({'x'+str(i): syn_sparse_test_data[j][i]
#         for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1})
#             for j in range(len(syn_sparse_test_data))]

# file_sparse = open('svm-sparse.txt', 'w')
# y_label_sparse = []
# for i in range(len(syn_sparse_test_x)):
#     y_label_sparse.append(p.predict_SVM(syn_sparse_test_x[i]))
# for label in y_label_sparse:    
#     if label == 1:
#         file_sparse.write(str(1)+'\n')
#     else:
#         file_sparse.write(str(-1)+'\n')


# In[28]:


# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10)

# syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')
# syn_dense_test_x=[({'x'+str(i): syn_dense_test_data[j][i]
#         for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1})
#             for j in range(len(syn_dense_test_data))]
# file_dense = open('svm-dense.txt', 'w')
# y_label_dense = []
# for i in range(len(syn_dense_test_x)):
#     y_label_dense.append(p.predict_SVM(syn_dense_test_x[i]))
# for label in y_label_dense:    
#     if label == 1:
#         file_dense.write(str(1)+'\n')
#     else:
#         file_dense.write(str(-1)+'\n')


# In[25]:



# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_test_x = extract_features_email_test(email_test_data, train_features)
# file_news = open('p-enron.txt', 'w')
# y_label_email = []
# for i in range(len(email_test_x)):
#     y_label_email.append(p.predict(email_test_x[i]))
# for label in y_label_email:    
#     if label == 1:
#         file_news.write('I\n')
#     else:
#         file_news.write('O\n')


# In[35]:


# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_test_x = extract_features_news_test(news_test_data, train_features)

# file_news = open('p-conll.txt', 'w')
# y_label_news = []
# for i in range(len(news_test_x)):
#     y_label_news.append(p.predict(news_test_x[i]))
# for label in y_label_news:    
#     if label == 1:
#         file_news.write('I\n')
#     else:
#         file_news.write('O\n')


# In[36]:


# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)

# news_test_x = extract_features_news_test(news_test_data, train_features)
# file_news = open('p-conll.txt', 'w')
# y_label_news = []
# for i in range(len(news_test_x)):
#     y_label_news.append(p.predict(news_test_x[i]))
# for label in y_label_news:    
#     if label == 1:
#         file_news.write('I\n')
#     else:
#         file_news.write('O\n')


# In[30]:


# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged = True)
# syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/Test/')
# syn_sparse_test_x=[({'x'+str(i): syn_sparse_test_data[j][i]
#         for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1})
#             for j in range(len(syn_sparse_test_data))]

# file_sparse = open('p-sparse.txt', 'w')
# y_label_sparse = []
# for i in range(len(syn_sparse_test_x)):
#     y_label_sparse.append(p.predict(syn_sparse_test_x[i]))
# for label in y_label_sparse:    
#     if label == 1:
#         file_sparse.write(str(1)+'\n')
#     else:
#         file_sparse.write(str(-1)+'\n')


# In[33]:


# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged = True)

# syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')
# syn_dense_test_x=[({'x'+str(i): syn_dense_test_data[j][i]
#         for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1})
#             for j in range(len(syn_dense_test_data))]
# file_dense = open('p-dense.txt', 'w')
# y_label_dense = []
# for i in range(len(syn_dense_test_x)):
#     y_label_dense.append(p.predict(syn_dense_test_x[i]))
# for label in y_label_dense:    
#     if label == 1:
#         file_dense.write(str(1)+'\n')
#     else:
#         file_dense.write(str(-1)+'\n')


# In[32]:


# print('\nPerceptron Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[16]:


# print('\nWinnow Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.001)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[33]:


# print('\nPerceptron Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn sparse Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn sparse Train Accuracy:', accuracy_train)


# In[35]:


# print('\nWinnow Accuracy')
# # # Test Perceptron on Dense Synthetic
# alpha_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]
# winnow_para = {}
# winnow_train = {}
# for val in alpha_values:
#     p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = val)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     winnow_para[val] = accuracy
#     print('Syn Dense Dev Accuracy:', accuracy)
#     accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
#     print('Syn Dense Train Accuracy:', accuracy_train)
#     winnow_train[val] = accuracy_train
# print('For dev',winnow_para) 
# print('For train' , winnow_train)


# In[36]:


# print('\nPerceptron Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# print('Syn Dense Dev No Noise Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[37]:


# alpha_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]
# winnow_para = {}
# winnow_train = {}
# for val in alpha_values:
#     p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha = val)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     winnow_para[val] = accuracy
#     print('Syn sparse Dev Accuracy:', accuracy)
#     accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
#     print('Syn sparse Train Accuracy:', accuracy_train)
#     winnow_train[val] = accuracy_train
# print('For dev',winnow_para) 
# print('For train' , winnow_train)


# In[38]:


# alpha_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]
# winnow_para = {}
# winnow_train = {}
# for val in alpha_values:
#     p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = val)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
#     winnow_para[val] = accuracy
# #     print('Syn dense Dev no noise Accuracy:', accuracy)
#     accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# #     print('Syn dense no noise Train Accuracy:', accuracy_train)
#     winnow_train[val] = accuracy_train
# print('For dev no noise',winnow_para) 
# print('For train' , winnow_train)


# In[40]:


# print('\nAdagrad Accuracy')
# # # Test Perceptron on Dense Synthetic
# eta_val = [1.5, 0.25, 0.03, 0.005, 0.001]
# ada_acc = {}
# ada_train = {}
# for val in eta_val:
#     p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = val)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# #     print('Syn Dense Dev Accuracy:', accuracy)
#     ada_acc[val] = accuracy
#     accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# #     print('Syn Dense Train Accuracy:', accuracy_train)
#     ada_train[val] = accuracy_train
# print('For dev', ada_acc) 
# print('For train' , ada_train)


# In[56]:


# print('\nAdagrad Accuracy')
# # # Test Perceptron on Dense Synthetic
# eta_val = [1.5, 0.25, 0.03, 0.005, 0.001]
# ada_acc = {}
# ada_train = {}
# for val in eta_val:
#     p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10, eta = val)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# #     print('Syn Dense Dev Accuracy:', accuracy)
#     ada_acc[val] = accuracy
#     accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# #     print('Syn Dense Train Accuracy:', accuracy_train)
#     ada_train[val] = accuracy_train
# print('For dev sparse', ada_acc) 
# print('For train sparse' , ada_train)


# In[42]:


# print('\nAdagrad Accuracy')
# # # Test Perceptron on Dense Synthetic
# eta_val = [1.5, 0.25, 0.03, 0.005, 0.001]
# ada_acc = {}
# ada_train = {}
# for val in eta_val:
#     p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = val)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# #     print('Syn Dense Dev Accuracy:', accuracy)
#     ada_acc[val] = accuracy
#     accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# #     print('Syn Dense Train Accuracy:', accuracy_train)
#     ada_train[val] = accuracy_train
# print('For dev no noise', ada_acc) 
# print('For train' , ada_train)


# In[36]:


# print('\nWinnow Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# In[43]:


# print('\nPerceptron-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)
   


# In[44]:


# print('\nPerceptron-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# print('Syn Dense Dev no noise Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)
   


# In[45]:


# print('\nPerceptron-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn sparse Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn sparse Train Accuracy:', accuracy_train)
   


# In[46]:


# print('\nSVM Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict_SVM(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[48]:


# print('\nSVM Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict_SVM(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# print('Syn Dense Dev no noise Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict_SVM(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[50]:


# print('\nSVM Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn sparse Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict_SVM(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn sparse Train Accuracy:', accuracy_train)


# In[28]:


# print('\nSVM Accuracy for real data')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict_SVM(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
# print('News Dev Accuracy:', accuracy)


# In[34]:


# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# dense_train_x_list = []
# dense_train_y_list = []

# for k in range(len(sets)):
#     dense_train_x_list.append(syn_dense_train_x[0:sets[k]])
#     dense_train_y_list.append(syn_dense_train_y[0:sets[k]])
# plot_ylabels =[]
# plot_xlabels =[]
# for j in range(len(sets)):
#     plot_xlabels.append(sets[j])
   
#     p = Classifier('Winnow', dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = False, alpha = 1.005
#                   )
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     plot_ylabels.append(accuracy)
# print(plot_ylabels)    


# In[46]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# y_predict = []

# for i in range(len(news_dev_y)):
#     y_predict.append(p.predict_SVM(news_dev_x[i]))

# precision_Con, recall_Con, f1_Con = calc_f1(news_dev_y,y_predict)
# print('Precision,Recall and f1 for ConLL',precision_Con, recall_Con, f1_Con)


# In[51]:


# print('\nWinnow-Avg Accuracy')
# # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[27]:


# print('\nWinnow-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha = 1.005, averaged = True)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Train Accuracy:', accuracy_train)


# In[53]:


# print('\nWinnow-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# print('Syn Dense Dev no noise Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[97]:


# print('\nWinnow-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.1)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# In[65]:


# print('\nPerceptron-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_email_dev(email_dev_data, train_features)
# accuracy = sum([1 for i in range(len(email_dev_y)) if p.predict(email_dev_x[i]) == email_dev_y[i]])/len(email_dev_y)*100
# print('Email Dev Accuracy:', accuracy)


# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)


# news_dev_x, news_dev_y=extract_features_dev(news_dev_data, train_features)
# accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
# print('news Dev Accuracy:', accuracy)


# In[66]:


# print('\nSVM Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged = True)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_email_dev(email_dev_data, train_features)
# accuracy = sum([1 for i in range(len(email_dev_y)) if p.predict_SVM(email_dev_x[i]) == email_dev_y[i]])/len(email_dev_y)*100
# print('Email Dev Accuracy:', accuracy)


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10, averaged = True)
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)


# news_dev_x, news_dev_y=extract_features_dev(news_dev_data, train_features)
# accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict_SVM(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
# print('news Dev Accuracy:', accuracy)


# In[54]:


# print('\nAdagrad-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[55]:


# print('\nAdagrad-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5, averaged = True)
# accuracy = sum([1 for i in range(len(syn_dense_dev_no_noise_y)) if p.predict(syn_dense_dev_no_noise_x[i]) == syn_dense_dev_no_noise_y[i]])/len(syn_dense_dev_no_noise_y)*100
# print('Syn Dense Dev no noise Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Train Accuracy:', accuracy_train)


# In[57]:


# print('\nAdagrad-Avg Accuracy')
# # # Test Perceptron on Dense Synthetic
# p = Classifier('Adagrad-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, eta = 1.5, averaged = True)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)
# accuracy_train = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn sparse Train Accuracy:', accuracy_train)


# In[39]:


# import numpy as np
# import matplotlib.pyplot as plt

# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# dense_train_x_list = []
# dense_train_y_list = []
# plot_xlabels =[]
# for k in range(len(sets)):
#     dense_train_x_list.append(syn_dense_train_x[0:sets[k]])
#     dense_train_y_list.append(syn_dense_train_y[0:sets[k]])
# models = [('Perceptron', False, 1, 1,'r'), ('Winnow', False, 1.0005, 1, 'g' ), ('Adagrad', False, 1, 1.5, 'b'),
#           ('Perceptron-Avg', True, 1, 1, 'c'), ('Winnow-Avg', True, 1.0005, 1, 'm'), ('Adagrad-Avg', True, 1, 1.5, 'k'), ('SVM', False, 1, 1, 'y')]  
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
# plot_xlabels = sets
# ax.set_xlim(500,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# for model in models:
#     plot_ylabels =[]
#     xi = np.arange(0,11,1)
#     for j in range(len(sets)):
#         if (model[0]=='SVM'):
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy) 
#         else:    
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1],
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y))
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
            
#     print('accuracy for dense'+ str(model),plot_ylabels)
#     ax.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
#     ax2.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])

# plt.xlabel("Training set size")       

# plt.legend(loc='lower right')   
# plt.show()         
# plt.savefig("curve.png")


# In[37]:


# import numpy as np
# import matplotlib.pyplot as plt

# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# dense_train_x_list = []
# dense_train_y_list = []
# plot_xlabels =[]
# for k in range(len(sets)):
#     dense_train_x_list.append(syn_dense_train_x[0:sets[k]])
#     dense_train_y_list.append(syn_dense_train_y[0:sets[k]])
# models = [('Perceptron', False, 1, 1,'r'), ('Winnow', False, 1.1, 1, 'g' ), ('Adagrad', False, 1, 1.5, 'b'),
#           ('Perceptron-Avg', True, 1, 1, 'c'), ('Winnow-Avg', True, 1.1, 1, 'm'), ('Adagrad-Avg', True, 1, 1.5, 'k'), ('SVM', False, 1, 1, 'y')]  
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
# plot_xlabels = sets
# ax.set_xlim(500,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# for model in models:
#     plot_ylabels =[]
#     xi = np.arange(0,11,1)
#     for j in range(len(sets)):
#         if (model[0]=='SVM'):
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy) 
#         elif(model[0]=='Winnow') :
#             p = Classifier('Winnow', dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.1)
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
#         elif(model[0]=='Winnow-Avg'):
#             p = Classifier('Winnow-Avg', dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.1)
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
        
#         else:    
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1],
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y))
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
            
#     print('accuracy for dense'+ str(model),plot_ylabels)
#     ax.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
#     ax2.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])

# plt.xlabel("Training set size")       

# plt.legend(loc='lower right')   
# plt.show()         
# plt.savefig("curve.png")


# In[26]:


# import numpy as np
# import matplotlib.pyplot as plt

# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# sparse_train_x_list = []
# sparse_train_y_list = []
# plot_xlabels =[]
# for k in range(len(sets)):
#     sparse_train_x_list.append(syn_sparse_train_x[0:sets[k]])
#     sparse_train_y_list.append(syn_sparse_train_y[0:sets[k]])
# models = [('Perceptron', False, 1, 1,'r'), ('Winnow', False, 1.1, 1, 'g' ), ('Adagrad', False, 1, 1.5, 'b'),
#           ('Perceptron-Avg', True, 1, 1, 'c'), ('Winnow-Avg', True, 1.1, 1, 'm'), ('Adagrad-Avg', True, 1, 1.5, 'k'), ('SVM', False, 1, 1, 'y')]  
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
# plot_xlabels = sets
# ax.set_xlim(500,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# for model in models:
#     plot_ylabels =[]
#     xi = np.arange(0,11,1)
#     for j in range(len(sets)):
#         if (model[0]=='SVM'):
#             p = Classifier(model[0], sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy) 
#         elif(model[0]=='Winnow') :
#             p = Classifier('Winnow', sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.1)
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
#         elif(model[0]=='Winnow-Avg'):
#             p = Classifier('Winnow-Avg', sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.1)
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
        
#         else:    
#             p = Classifier(model[0], sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1],
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y))
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
            
#     print('accuracy for dense'+ str(model),plot_ylabels)
#     ax.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
#     ax2.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])

# plt.xlabel("Training set size")       

# plt.legend(loc='lower right')   
# plt.show()         
# plt.savefig("curve.png")


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# sparse_train_x_list = []
# sparse_train_y_list = []
# plot_xlabels =[]
# for k in range(len(sets)):
#     sparse_train_x_list.append(syn_sparse_train_x[0:sets[k]])
#     sparse_train_y_list.append(syn_sparse_train_y[0:sets[k]])
# models = [('Perceptron', False, 1, 1,'r'), ('Winnow', False, 1.005, 1, 'g' ), ('Adagrad', False, 1, 1.5, 'b'),
#           ('Perceptron-Avg', True, 1, 1, 'c'), ('Winnow-Avg', True, 1.005, 1, 'm'), ('Adagrad-Avg', True, 1, 1.5, 'k'), ('SVM', False, 1, 1, 'y')]  
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
# plot_xlabels = sets
# ax.set_xlim(500,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# for model in models:
#     plot_ylabels =[]
#     xi = np.arange(0,11,1)
#     for j in range(len(sets)):
#         if (model[0]=='SVM'):
#             p = Classifier(model[0], sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy) 
#         elif(model[0]=='Winnow') :
#             p = Classifier('Winnow', sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.005)
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
#         elif(model[0]=='Winnow-Avg'):
#             p = Classifier('Winnow-Avg', sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.005)
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) 
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
        
#         else:    
#             p = Classifier(model[0], sparse_train_x_list[j], sparse_train_y_list[j], iterations=10, averaged = model[1],
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_sparse_dev_y))
#                             if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#             plot_ylabels.append(accuracy)
            
#     print('accuracy for dense'+ str(model),plot_ylabels)
#     ax.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
#     ax2.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])

# plt.xlabel("Training set size")       

# plt.legend(loc='lower right')   
# plt.show()         
# plt.savefig("curve.png")


# In[28]:


# import numpy as np
# import matplotlib.pyplot as plt

# sets = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# dense_train_x_list = []
# dense_train_y_list = []
# plot_xlabels =[]
# for k in range(len(sets)):
#     dense_train_x_list.append(syn_dense_train_x[0:sets[k]])
#     dense_train_y_list.append(syn_dense_train_y[0:sets[k]])
# models = [('Perceptron', False, 1, 1,'r'), ('Winnow', False, 1.1, 1, 'g' ), ('Adagrad', False, 1, 1.5, 'b'),
#           ('Perceptron-Avg', True, 1, 1, 'c'), ('Winnow-Avg', True, 1.1, 1, 'm'), ('Adagrad-Avg', True, 1, 1.5, 'k'), ('SVM', False, 1, 1, 'y')]  
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
# plot_xlabels = sets
# ax.set_xlim(500,5000)
# ax2.set_xlim(5000,50000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# for model in models:
#     plot_ylabels =[]
#     xi = np.arange(0,11,1)
#     for j in range(len(sets)):
#         if (model[0]=='SVM'):
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy) 
#         elif(model[0]=='Winnow') :
#             p = Classifier('Winnow', dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.005)
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
#         elif(model[0]=='Winnow-Avg'):
#             p = Classifier('Winnow-Avg', dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1], 
#                            alpha = 1.005)
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y)) 
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
        
#         else:    
#             p = Classifier(model[0], dense_train_x_list[j], dense_train_y_list[j], iterations=10, averaged = model[1],
#                            alpha = model[2], eta = model[3])
#             accuracy = sum([1 for i in range(len(syn_dense_dev_y))
#                             if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#             plot_ylabels.append(accuracy)
            
#     print('accuracy for dense'+ str(model),plot_ylabels)
#     ax.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
#     ax2.plot(plot_xlabels, plot_ylabels,label = model[0], color = model[4])
# plt.ylabel("Accuracy")
# plt.xlabel("Training set size")       

# plt.legend(loc='lower right')   
# plt.show()         
# plt.savefig("curve.png")

