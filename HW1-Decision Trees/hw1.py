
# coding: utf-8

# # Homework 1 Template
# This is the template for the first homework assignment.
# Below are some function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[1]:


# Uncomment and run this code if you want to verify your `sklearn` installation.
# If this cell outputs 'array([1])', then it's installed correctly.

#from sklearn import tree
#X = [[0, 0], [1, 1]]
#y = [0, 1]
#clf = tree.DecisionTreeClassifier(criterion='entropy')
#clf = clf.fit(X, y)
#clf.predict([[2, 2]])


# In[2]:


# Uncomment this code to see how to visualize a decision tree. This code should
# be commented out when you submit to the autograder.
# If this cell fails with
# an error related to `pydotplus`, try running `pip install pydotplus`
# from the command line, and retry. Similarly for any other package failure message.
# If you can't get this cell working, it's ok - this part is not required.
#
# This part should be commented out when you submit it to Gradescope

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus

#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,
#                feature_names=['feature1', 'feature2'],
#                class_names=['0', '1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())


# In[6]:


# This code should be commented out when you submit to the autograder.
# This cell will possibly download and unzip the dataset required for this assignment.
# It hasn't been tested on Windows, so it will not run if you are running on Windows.

#import os

#if os.name != 'nt':  # This is the Windows check
#    if not os.path.exists('badges.zip'):
        # If your statement starts with "!", then the command is run in bash, not python
#        !wget https://www.seas.upenn.edu/~cis519/fall2018/assets/HW/HW1/badges.zip
#        !mkdir -p badges
#        !unzip badges.zip -d badges
#        print('The data has saved in the "badges" directory.')
#else:
#    print('Sorry, I think you are running on windows. '
#          'You will need to manually download the data')


# In[1]:


import string
def compute_features(name):
    """
    Compute all of the features for a given name. The input
    name will always have 3 names separated by a space.
    
    Args:
        name (str): The input name, like "bill henry gates".
    Returns:
        list: The features for the name as a list, like [0, 0, 1, 0, 1].
    """
    char = list((string.ascii_lowercase))
    first, middle, last = name.split()
    if len(first) < 5:
        first = first + '&'*(5 - len(first))
    if len(middle) < 5:
        middle = middle + '&'*(5- len(middle))
    if len(last) <5:
        last = last + '&'*(5-len(last))
    new_name = first[:5] + middle[:5] + last[:5]
    full_feature_list = []
    
    for i in new_name:
        feature_list1 =[]
        for j in range(26):
            feature_list1.append(0)
        if i in char:
            feature_list1[char.index(i)] = 1
            full_feature_list.extend(feature_list1)
 
        else:
            feature_list2=[]
            for k  in range(26):
                feature_list2.append(0)
            full_feature_list.extend(feature_list2)
   
   
    return(full_feature_list)
   


# In[2]:


# result = compute_features('shai kumar sinha')
# print(result)
# print(len(result))


# In[3]:



from sklearn.tree import DecisionTreeClassifier

# The `max_depth=None` construction is how you specify default arguments
# in python. By adding a default argument, you can call this method in a couple of ways:
#     
#     train_decision_tree(X, y)
#     train_decision_tree(X, y, 4) or train_decision_tree(X, y, max_depth=4)
#
# In the first way, max_depth is automatically set to `None`, otherwise it is 4.

        
def train_decision_tree(X, y, max_depth=None):
    """
    Trains a decision tree on the input data using the information gain criterion
    (set the criterion in the constructor to 'entropy').
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        max_depth (int): The maximum depth the decision tree is allowed to be. If
                         `None`, then the depth is unbounded.
    Returns:
        DecisionTreeClassifier: the learned decision tree.
    """
   

    clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = max_depth)
    n_clf = clf.fit(np.array(X),y)
    return n_clf
    


# In[4]:


from sklearn.linear_model import SGDClassifier
import numpy as np

def train_sgd(X, y, learning_rate='optimal', alpha=0.0001, tol=None):
    """
    Trains an `SGDClassifier` using 'log' loss on the input data.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        learning_rate (str): The learning rate to use. See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    Returns:
        SGDClassifier: the learned classifier.
        
    """
    
    clf = SGDClassifier(loss="log", alpha=alpha, tol=tol)
    s_clf = clf.fit(np.array(X),y)
    return s_clf


# In[5]:


from sklearn.model_selection import train_test_split
import numpy as np
def train_sgd_with_stumps(X, y):
    """
    Trains an `SGDClassifier` using 'log' loss on the input data. The classifier will
    be trained on features that are computed using decision tree stumps.
    
    This function will return two items, the `SGDClassifier` and list of `DecisionTreeClassifier`s
    which were used to compute the new feature set. If `sgd` is the name of your `SGDClassifier`
    and `stumps` is the name of your list of `DecisionTreeClassifier`s, then writing
    `return sgd, stumps` will return both of them at the same time.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
    Returns:
        SGDClassifier: the learned classifier.
        List[DecisionTree]: the decision stumps that were used to compute the features
                            for the `SGDClassifier`.
    """
    # This is an example for how to return multiple arguments
    # in python. If you write `a, b = train_sgd_with_stumps(X, y)`, then
    # a will be 1 and b will be 2.
    
    #generates stumps
    
    stumps = []
    X_fin=[]
    X_1= []
    for i in range(200):
        new_x= []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        dt_stump = DecisionTreeClassifier(max_depth = 8,criterion = 'entropy').fit(X_train,y_train)
        stumps.append(dt_stump)     
        
        new_x = dt_stump.predict(X)
        
        X_1.append(new_x)
        
    X_fin = np.transpose(X_1)
    
    sgd = SGDClassifier(loss = 'log').fit(np.array(X_fin),y)           
    return sgd, stumps


# In[6]:


# The input to this function can be an `SGDClassifier` or a `DecisionTreeClassifier`.
# Because they both use the same interface for predicting labels, the code can be the same
# for both of them.
def predict_clf(clf, X):
    """
    Predicts labels for all instances in `X` using the `clf` classifier. This function
    will be the same for `DecisionTreeClassifier`s and `SGDClassifier`s.
    
    Args:
        clf: (`SGDClassifier` or `DecisionTreeClassifier`): the trained classifier.
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
    Returns:
        List[int]: the predicted labels for each instance in `X`.
    """
        
    return clf.predict(X)

    


# In[7]:


# The SGD-DT classifier can't use the same function as the SGD or decision trees
# because it requires an extra argument
import numpy as np
def predict_sgd_with_stumps(sgd, stumps, X):
    """
    Predicts labels for all instances `X` using the `SGDClassifier` trained with
    features computed from decision stumps. The input `X` will be a matrix of the
    original features. The stumps will be used to map `X` from the original features
    to the features that the `SGDClassifier` were trained with.
    
    Args:
        sgd (`SGDClassifier`): the classifier that was trained with features computed
                               using the input stumps.
        stumps (List[DecisionTreeClassifier]): a list of `DecisionTreeClassifier`s that
                                               were used to train the `SGDClassifier`.
        X (list of lists): The features that were used to train the stumps (i.e. the original
                           feature set).
    Returns:
        List[int]: the predicted labels for each instance in `X`.
    """
         
    X_fin =[]
    X_1 = []
    for stump in stumps:
        new_x = []
        new_x = stump.predict(X)
        X_1.append(new_x)
        
    X_fin = np.transpose(X_1)
    return predict_clf(sgd, X_fin)
    


# In[8]:


# Write the rest of your code here. Anything from here down should be commented
# out when you submit to the autograder


# In[9]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519')


# In[68]:


# def splitting_files(fold_number):
#     X_lists = []
#     y_lists = []
#     feature_names = []
#     names = []
#     with open('train.fold-'+str(fold_number)+'.txt' , 'r') as f:
#         for line in f:
#             content = line.strip()
#             word = content.split()
#             if word[0] == '+':
#                 y_lists.append(1)
#             else:
#                 y_lists.append(0)  
#             feature_names.append(word[1]+ " " + word[2] + " " + word[3])
            
#         for every_name in feature_names:
#             feature_name =compute_features(every_name)
#             X_lists.append(feature_name) 
#     return(X_lists,y_lists)
       


# In[69]:


# X0,y0 = splitting_files(0)
# X1,y1 = splitting_files(1)
# X2,y2 = splitting_files(2)
# X3,y3 = splitting_files(3)
# X4,y4 = splitting_files(4)
# X_full = []
# y_full =[]
# X_splits = [X0,X1,X2,X3,X3]
# y_splits= [y0,y1,y2,y3,y4]

    


# In[70]:


# fold0_X = X1+X2+X3+X4
# fold0_y = y1+y2+y3+y4
# test0 = X0
# fold1_X = X0+X2+X3+X4
# fold1_y = y0+y2+y3+y4
# test1 = X1
# fold2_X= X0+X1+X3+X4
# fold2_y= y0+y1+y3+y4
# test2 = X2
# fold3_X =X0+X1+X2+X4
# fold3_y= y0+y1+y2+y4
# test3 = X3
# fold4_X = X0+X1+X2+X3
# fold4_y= y0+y1+y2+y3
# test4 = X4
     
    


# In[71]:


# train_sgd(X0,y0,alpha = 0.0001,tol = 0.005)


# In[72]:


# train_sgd(X0,y0,alpha = 0.0001)


# In[15]:


# from sklearn.metrics import accuracy_score
# folds_X = [fold0_X,fold1_X,fold2_X,fold3_X,fold4_X]
# folds_y = [fold0_y,fold1_y,fold2_y,fold3_y,fold4_y]
# tests = [test0,test1,test2,test3,test4]
# X_splits = [X0,X1,X2,X3,X4]
# y_splits= [y0,y1,y2,y3,y4]


# def sgd_score_vary_alpha(folds_X,folds_y,alp):
#     pa_list2 = []
#     ta_list2 = []
#     total_pa = 0
#     total_ta = 0
#     for i in range(5):
#         new_clf = train_sgd(folds_X[i],folds_y[i],alpha = alp)
#         y_test_pred =predict_clf(new_clf,X_splits[i])
#         pa_score = accuracy_score(y_test_pred,y_splits[i])
#         pa_list2.append(pa_score)
#         y_train_pred = predict_clf(new_clf,folds_X[i])
#         ta_score = accuracy_score(y_train_pred,folds_y[i])
#         ta_list2.append(ta_score)
        
#         total_ta = total_ta + ta_score
#         total_pa = total_pa + pa_score
    
#     pa = total_pa/5 
#     ta = total_ta/5
#     return pa, ta, pa_list2, ta_list2


# In[20]:


# pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_score_vary_alpha(folds_X,folds_y,0.001)
# print(pa_sgd)
# print(pa_sgd_score, ta_sgd_score)


# In[19]:


# pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = sgd_score_vary_alpha(folds_X,folds_y,0.0001)
# print(pa_sgd1)
# print(pa_sgd_score1, ta_sgd_score1)


# In[18]:


# pa_sgd_score2, ta_sgd_score2, pa_sgd2, ta_sgd2 = sgd_score_vary_alpha(folds_X,folds_y,5e-4)
# print(pa_sgd2)
# print(pa_sgd_score2, ta_sgd_score2)


# In[21]:


# pa_sgd_score3, ta_sgd_score3, pa_sgd3, ta_sgd3 = sgd_score_vary_alpha(folds_X,folds_y,1e-5)
# print(pa_sgd3)
# print(pa_sgd_score3, ta_sgd_score3)


# In[22]:


# pa_sgd_score4, ta_sgd_score4, pa_sgd4, ta_sgd4 = sgd_score_vary_alpha(folds_X,folds_y,5e-3)
# print(pa_sgd4)
# print(pa_sgd_score4, ta_sgd_score4)


# In[23]:


# pa_sgd_score5, ta_sgd_score5, pa_sgd5, ta_sgd5 = sgd_score_vary_alpha(folds_X,folds_y,1e-2)
# print(pa_sgd5)
# print(pa_sgd_score5, ta_sgd_score5)


# In[31]:



# def sgd_score_vary_tol(folds_X,folds_y,tol):
#     pa_list2 = []
#     ta_list2 = []
#     total_pa = 0
#     total_ta = 0
#     for i in range(5):
#         new_clf = train_sgd(folds_X[i],folds_y[i],tol = tol)
#         y_test_pred =predict_clf(new_clf,X_splits[i])
#         pa_score = accuracy_score(y_test_pred,y_splits[i])
#         pa_list2.append(pa_score)
#         y_train_pred = predict_clf(new_clf,folds_X[i])
#         ta_score = accuracy_score(y_train_pred,folds_y[i])
#         ta_list2.append(ta_score)
        
#         total_ta = total_ta + ta_score
#         total_pa = total_pa + pa_score
   
#     pa = total_pa/5 
#     ta = total_ta/5
#     return pa, ta, pa_list2, ta_list2


# In[32]:


# pa_sgd_score_1, ta_sgd_score_1, pa_sgd_1, ta_sgd_1 = sgd_score_vary_tol(folds_X,folds_y,1e-3)
# print(pa_sgd_1)
# print(pa_sgd_score_1, ta_sgd_score_1)


# In[39]:


# pa_sgd_score_2, ta_sgd_score_2, pa_sgd_2, ta_sgd_2 = sgd_score_vary_tol(folds_X,folds_y,1e-4)
# print(pa_sgd_2)
# print(pa_sgd_score_2, ta_sgd_score_2)


# In[42]:


# pa_sgd_score_3, ta_sgd_score_3, pa_sgd_3, ta_sgd_3 = sgd_score_vary_tol(folds_X,folds_y,1e-5)
# print(pa_sgd_3)
# print(pa_sgd_score_3, ta_sgd_score_3)


# In[43]:


# pa_sgd_score_7, ta_sgd_score_7, pa_sgd_7, ta_sgd_7 = sgd_score_vary_tol(folds_X,folds_y,1e-2)
# print(pa_sgd_7)
# print(pa_sgd_score_7, ta_sgd_score_7)


# In[36]:


# pa_sgd_score_4, ta_sgd_score_4, pa_sgd_4, ta_sgd_4 = sgd_score_vary_tol(folds_X,folds_y,5e-1)
# print(pa_sgd_4)
# print(pa_sgd_score_4, ta_sgd_score_4)


# In[187]:


# pa_sgd_score_5, ta_sgd_score_5, pa_sgd_5, ta_sgd_5 = sgd_score_vary_tol(folds_X,folds_y,0.01)
# print(pa_sgd_5)
# print(pa_sgd_score_5, ta_sgd_score_5)


# In[189]:


# pa_sgd_score_6, ta_sgd_score_6, pa_sgd_6, ta_sgd_6 = sgd_score_vary_tol(folds_X,folds_y,1e-1)
# print(pa_sgd_6)
# print(pa_sgd_score_6, ta_sgd_score_6)


# In[49]:


# folds_X = [fold0_X,fold1_X,fold2_X,fold3_X,fold4_X]
# folds_y = [fold0_y,fold1_y,fold2_y,fold3_y,fold4_y]
# tests = [test0,test1,test2,test3,test4]
# X_splits = [X0,X1,X2,X3,X4]
# y_splits= [y0,y1,y2,y3,y4]
# def varydepth_score_dt2(folds_X,folds_y,max_dep):
#     total_pa2 = 0
#     total_ta2 = 0
#     pa_list2 = []
#     pa_score2= 0
#     ta_score2 = 0
#     ta_list2 = []
#     for i in range(0,5):
#         new_clf2 = train_decision_tree(folds_X[i], folds_y[i], max_depth=max_dep)
#         y_test_pred2 =predict_clf(new_clf2,X_splits[i])
#         pa_score2 = accuracy_score(y_test_pred2,y_splits[i])
#         pa_list2.append(pa_score2)
#         new_clf3 = train_decision_tree(folds_X[i],folds_y[i],max_depth = max_dep)
#         y_train_pred = predict_clf(new_clf3,folds_X[i])
#         ta_score2 = accuracy_score(y_train_pred,folds_y[i])
#         #ta_score2 = DecisionTreeClassifier(max_depth = 4).fit(folds_X[i],folds_y[i]).score(folds_X[i],folds_y[i])
#         ta_list2.append(ta_score2)
#         total_ta2 = total_ta2 + ta_score2
#         total_pa2= total_pa2 + pa_score2
#     print(new_clf2)
#     pa2 = total_pa2/5 
#     ta2 = total_ta2/5
#     return pa2, ta2, pa_list2, ta_list2 


# In[50]:


# pa_dt_score, ta_dt_score, pa_dt, ta_dt = varydepth_score_dt2(folds_X,folds_y,None)
# print(pa_dt)
# print(pa_dt_score,ta_dt_score)
# print(ta_dt)


# In[51]:


# pa_score_dt4, ta_score_dt4 ,pa_dt4 , ta_dt4=varydepth_score_dt2(folds_X,folds_y,4)
# print(pa_dt4)
# print(pa_score_dt4,ta_score_dt4)
# print(ta_dt4)


# In[52]:



# pa_score_dt8, ta_score_dt8 ,pa_dt8 , ta_dt8 = varydepth_score_dt2(folds_X,folds_y,8)
# print(pa_dt8)
# print(ta_dt8)
# print(pa_score_dt8,ta_score_dt8)


# In[135]:


# a, b = train_sgd_with_stumps(fold0_X, fold0_y)
# print(a,b)
    


# In[136]:


# predict_sgd_with_stumps(a, b, X_splits[0])


# In[53]:


# from sklearn.metrics import accuracy_score
# folds_X = [fold0_X,fold1_X,fold2_X,fold3_X,fold4_X]
# folds_y = [fold0_y,fold1_y,fold2_y,fold3_y,fold4_y]
# tests = [test0,test1,test2,test3,test4]
# X_splits = [X0,X1,X2,X3,X4]
# y_splits= [y0,y1,y2,y3,y4]
# total_pa = 0
# total_ta = 0
# def stumps_score(folds_X,folds_y):
#     total_pa3 = 0
#     total_ta3 = 0
#     ta_list3 =[]
#     pa_list3= []
#     pa_score3 =0
#     ta_score3=0
#     a3= 0
#     b3 =0
#     for i in range(5):
#         a3, b3= train_sgd_with_stumps(folds_X[i], folds_y[i])
#         y_test_pred3 = predict_sgd_with_stumps(a3, b3, X_splits[i])
#         pa_score3 = accuracy_score(y_test_pred3,y_splits[i])
#         total_pa3 = total_pa3 + pa_score3
#         pa_list3.append(pa_score3)
#         y_train_pred3 = predict_sgd_with_stumps(a3, b3, folds_X[i])
#         ta_score3 = accuracy_score(y_train_pred3,folds_y[i])
#         total_ta3 = total_ta3 + ta_score3
#         ta_list3.append(ta_score3)
#     print(a3)   
#     pa3 = total_pa3/5
#     ta3 = total_ta3/5
#     return pa3, ta3, pa_list3, ta_list3


# In[54]:


# pa_score_stumps, ta_score_stumps, pa_stumps, ta_stumps = stumps_score(folds_X,folds_y)
# print(pa_score_stumps, ta_score_stumps)
# print(pa_stumps)
# print(ta_stumps)


# In[55]:


# test_X=[]
# a_test = []
# names = []
# def get_test_X(): 
#     with open('test.unlabeled.txt' , 'r') as f:
#         for line in f:
#             content = line.strip()
#             word = content.split()
#             #print(word)
#             names.append(word[0]+" " + word[1]+ " " + word[2])
                  
#         for name in names:
#             a_test =compute_features(name)
#             test_X.append(a_test)
#     return test_X


# In[56]:


# import numpy as np
# import math
# def calc_confidence(pa_score_list):
#     ac = pa_score_list
#     x_bar = np.mean(ac)
#     S = np.std(ac)
#     n = len(ac)
#     ci_lower = x_bar - 4.604*(S/math.sqrt(n))
#     ci_upper = x_bar + 4.604*(S/math.sqrt(n))
#     return ci_upper, ci_lower    


# In[58]:


# ci_for_dt = calc_confidence(pa_dt)
# ci_for_dt4 =calc_confidence(pa_dt4)
# ci_for_dt8 =calc_confidence(pa_dt8)
# ci_for_sgd =calc_confidence(pa_sgd1)
# ci_for_stumps = calc_confidence(pa_stumps)
# print(ci_for_dt)
# print(ci_for_dt4)
# print(ci_for_dt8)
# print(ci_for_sgd)
# print(ci_for_stumps)


# In[59]:


# X_train_full = []
# y_train_full = []
# a_train_full = []
# names_train_full = []
# with open('train.txt' , 'r') as f:
#         for line in f:
#             content = line.strip()
#             word = content.split()
#             names_train_full.append(word[1]+ " " + word[2] + " " + word[3])
#             if word[0] == '+':
#                 y_train_full.append(1)
#             else:
#                 y_train_full.append(0)  
#         for name in names_train_full:
#             a_train_full =compute_features(name)
#             X_train_full.append(a_train_full)
# print(X_train_full,y_train_full)


# In[167]:



# def y_pred_final(clf):
#     y_unlabel =predict_clf(clf, get_test_X())
#     y_final_test = []
#     for label in y_unlabel:
#         if(label == 1):
#             y_final_test.append('+')
#         elif(label == 0):
#             y_final_test.append('-')
#     return y_final_test    
    


# In[169]:


# import numpy
# import pandas 
# import csv
# y_unlabel_sgd = y_pred_final(train_sgd(X_train_full,y_train_full,alpha = 0.001))
# y_unlabel_dt = y_pred_final(train_decision_tree(X_train_full,y_train_full))
# y_unlabel_dt4 = y_pred_final(train_decision_tree(X_train_full,y_train_full,4))
# y_unlabel_dt8 = y_pred_final(train_decision_tree(X_train_full,y_train_full,8))


# with open("sgd.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in y_unlabel_sgd:
#         writer.writerow([val])
# with open("dt.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in y_unlabel_dt:
#         writer.writerow([val])        
# with open("dt-4.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in y_unlabel_dt4:
#         writer.writerow([val])
# with open("dt-8.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in y_unlabel_dt8:
#         writer.writerow([val])
       
#     print(y_unlabel_sgd)       



# In[170]:


# def y_pred_final_stumps():
#     te, xe =train_sgd_with_stumps(X_train_full,y_train_full)
#     y_unlabel_stumps = predict_sgd_with_stumps(te, xe, get_test_X())
#     y_final_stumps = []
#     for label in y_unlabel_stumps:
#         if(label == 1):
#             y_final_stumps.append('+')
#         elif(label == 0):
#             y_final_stumps.append('-')
#     return y_final_stumps


# In[171]:


# import csv

# y_fin_unlabel_sgdt = y_pred_final_stumps()
# with open("sgd-dt.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in y_fin_unlabel_sgdt:
#         writer.writerow([val])

