
# coding: utf-8

# # Homework 3 Template
# This is the template for the third homework assignment.
# Below are some class and function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[1]:


# import os
# os.chdir(r'C:\Users\Dell1\Desktop\CIS519')


# In[2]:


############################################################
# Imports
############################################################
# Include your imports here, if any are used.
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms


# In[3]:


def extract_data(x_data_filepath, y_data_filepath):
    X = np.load(x_data_filepath)
    y = np.load(y_data_filepath)
    return X, y


# In[4]:


############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    """CIFAR-10 image dataset."""
    
    def __init__(self, X, y, transformations=None):
        self.len = len(X)           
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# In[5]:


############################################################
# Feed Forward Neural Network
############################################################
class FeedForwardNN(nn.Module):
    """ 
        (1) Use self.fc1 as the variable name for your first fully connected layer
        (2) Use self.fc2 as the variable name for your second fully connected layer
    """
    
    def __init__(self):
        
        super(FeedForwardNN, self).__init__()
        
        self.fc1 = nn.Linear(32*32*3, 1500)
        self.fc2 = nn.Linear(1500, 10)
        

    def forward(self, x):
        
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        
        return out
       

    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()


# In[6]:


############################################################
# Convolutional Neural Network
############################################################
class ConvolutionalNN(nn.Module):
    """ 
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool as the variable name for your pooling layer
        (3) User self.conv2 as the variable name for your second convolutional layer
        (4) Use self.fc1 as the variable name for your first fully connected layer
        (5) Use self.fc2 as the variable name for your second fully connected layer
        (6) Use self.fc3 as the variable name for your third fully connected layer
    """
    def __init__(self):
        
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 3, 1, 0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(7, 16, 3, 1, 0)
        self.fc1 = nn.Linear(16*13*13, 130)
        self.fc2 = nn.Linear(130, 72)
        self.fc3 = nn.Linear(72, 10)
        

    def forward(self, x):
        
        
        out = F.relu(self.conv1(x))
        
        out = self.pool(out)
      
        out = F.relu(self.conv2(out))
        out = out.view(-1, 16 * 13 * 13)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        
        out = F.sigmoid(self.fc3(out))
       
        return out
    
    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_conv1_params(self):
        return self.conv1.__repr__()
    
    def get_pool_params(self):
        return self.pool.__repr__()

    def get_conv2_params(self):
        return self.conv2.__repr__()
    
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()
    
    def get_fc3_params(self):
        return self.fc3.__repr__()


# In[17]:


############################################################
# Hyperparameterized Feed Forward Neural Network
############################################################
class HyperParamsFeedForwardNN(nn.Module):
    def __init__(self):
        
        super(HyperParamsFeedForwardNN, self).__init__()
        
        self.fc1 = nn.Linear(32*32*3, 2000)
        self.fc2 = nn.Linear(2000, 10)
       
      
        
        

    def forward(self, x):
        
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        
        return out


# In[8]:


############################################################
# Hyperparameterized Convolutional Neural Network
############################################################
class HyperParamsConvNN(nn.Module):
    def __init__(self, kernel_size=3, img_size=32):
        super(HyperParamsConvNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 7, 3, 1, 0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(7, 16, 3, 1, 0)
        self.fc1 = nn.Linear(16*13*13, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 10)
       
        

    def forward(self, x):

        
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 16 * 13 * 13)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        
        
        return out
        
        


# In[9]:


############################################################
# Run Experiment
############################################################
def run_experiment(neural_network, train_loader, test_loader, loss_function, optimizer):
    """
    Runs experiment on the model neural network given a train and test data loader, loss function and optimizer.

    Args:
        neural_network (NN model that extends torch.nn.Module): For example, it should take an instance of either
                                                                FeedForwardNN or ConvolutionalNN,
        train_loader (DataLoader),
        test_loader (DataLoader),
        loss_function (torch.nn.CrossEntropyLoss),
        optimizer (optim.SGD)
    Returns:
        tuple: First position, testing accuracy.
               Second position, training accuracy.
               Third position, training loss.

               For example, if you find that
                            testing accuracy = 0.76,
                            training accuracy = 0.24
                            training loss = 0.56

               This function should return (0.76, 0.24, 0.56)
    """
    neural_net = neural_network
    train_loader = train_loader
    test_loader = test_loader
    criterion = loss_function
    optimizer = optimizer(neural_net.parameters(), lr=0.001, momentum=0.9)

    max_epochs = 100

    loss_np = np.zeros((max_epochs))
    epoch_accuracy = np.zeros((max_epochs))
    test_accuracy = np.zeros((max_epochs))
    

    
    for epoch in range(max_epochs):
        loss_list= []
        batch_accuracy =[]
        test_acc =[]
        for i, data in enumerate(train_loader, 0):
            # Get inputs and labels from data loader 
            inputs, labels = data
            if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
                inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

            elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
                inputs, labels = Variable(inputs), Variable(labels)    
            
            
            # zero gradient
            optimizer.zero_grad()
        
            # Feed the input data into the network 
            y_pred = neural_net(inputs)
        
            # Calculate the loss using predicted labels and ground truth labels
            loss = criterion(y_pred, labels)
            loss_f = float(loss)
            # backpropogates to compute gradient
            loss.backward()
        
            # updates the weghts
            optimizer.step()
           
            y_pred_np = y_pred.data.numpy()
        
            pred_np = np.empty(len(y_pred_np))
            for m in range(len(y_pred_np)):
                pred_np[m] = np.argmax(y_pred_np[m])
                
            pred_np = pred_np.reshape(len(pred_np), 1)

            label_np = labels.data.numpy().reshape(len(labels), 1)
            
            correct = 0
            for j in range(y_pred_np.shape[0]):
                if pred_np[j] == label_np[j]:
                    correct += 1
            batch_accuracy.append(float(correct)/float(len(label_np)))    
            loss_list.append(loss.data.numpy()) 
          
        loss_np[epoch] =  np.mean(np.asarray(loss_list))  
        epoch_accuracy[epoch] = np.mean(np.asarray(batch_accuracy))
        print("epoch: ", epoch, "loss: ", loss_np[epoch], "epoch_acc", epoch_accuracy[epoch])
   


        
        for data in test_loader:
        
        # Get inputs and labels from data loader 
            test_inputs, test_labels = data
           
            if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
                test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
            elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
                test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
            y_pred_test = neural_net(test_inputs)
                                            
            y_pred_test_np = y_pred_test.data.numpy()
            pred_test_np = np.empty(len(y_pred_test_np))
        
            for k in range(len(y_pred_test_np)):
                pred_test_np[k] = np.argmax(y_pred_test_np[k])

            pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

            label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

            correct_test = 0
            for j in range(y_pred_test_np.shape[0]):
                if pred_test_np[j] == label_test_np[j]:
                    correct_test += 1
            test_acc.append(float(correct_test) / float(len(label_test_np)))
        test_accuracy[epoch] = np.mean(np.asarray(test_acc))    
        print("epoch", epoch, "test acc",test_accuracy[epoch])    
    
    print("Test Accuracy: ", test_accuracy[max_epochs-1])  
    print("final training accuracy: ", epoch_accuracy[max_epochs-1])
    

    epoch_number = np.arange(0,max_epochs,1)


# Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, epoch_accuracy)
    plt.title('training accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    
    
        # Plot the loss over epoch
    plt.figure()
    plt.plot(epoch_number, loss_np)
    plt.title('loss over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    
    
    plt.figure()
    plt.plot(epoch_number, epoch_accuracy)
    plt.title('training /test accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    plt.plot(epoch_number, test_accuracy)
#     plt.title('test accuracy over epoches')
    plt.show()

    return test_accuracy[max_epochs-1], epoch_accuracy[max_epochs-1], loss_np[max_epochs-1]
   


# In[10]:


def normalize_image(image):
    """
    Normalizes the RGB pixel values of an image.

    Args:
        image (3D NumPy array): For example, it should take in a single 3x32x32 image from the CIFAR-10 dataset
    Returns:
        tuple: The normalized image
    """
    
    img_red = image[0]
    img_green = image[1]
    img_blue = image[2]
        
    mean_red = np.mean(img_red).astype(np.float64)
    std_red =  np.std(img_red).astype(np.float64)
    
    mean_green = np.mean(img_green).astype(np.float64)
    std_green =  np.std(img_green).astype(np.float64)
    
    mean_blue = np.mean(img_blue).astype(np.float64)
    std_blue =  np.std(img_blue).astype(np.float64)
        
    img_r_norm = (img_red - mean_red) / std_red  
    img_g_norm = (img_green - mean_green) / std_green
    img_b_norm = (img_blue - mean_blue) / std_blue
        
    image[0] = img_r_norm
    image[1] = img_g_norm
    image[2] = img_b_norm
        
    return image
   


# In[11]:


# class Dataset_test(Dataset):
#     """CIFAR-10 image dataset."""
    
#     x_data_test = 'cifar10-data/test_images.npy'
#     y_data_test = 'cifar10-data/test_labels.npy'

#     X_test, y_test = extract_data(x_data_test, y_data_test)
#     def __init__(self, X_test, y_test, transformations=None):
#         self.len = len(X)           
#         self.x_data = torch.from_numpy(X).float()
#         self.y_data = torch.from_numpy(y).long()
    
#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         return self.x_data[idx], self.y_data[idx]


# In[22]:


# x_data_train = 'cifar10-data/train_images.npy'
# y_data_train = 'cifar10-data/train_labels.npy'

# X, y = extract_data(x_data_train, y_data_train)

# x_data_test = 'cifar10-data/test_images.npy'
# y_data_test = 'cifar10-data/test_labels.npy'

# X_test, y_test = extract_data(x_data_test, y_data_test)


# In[20]:


# net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[20]:


# def normalizeflip_image(image):
#     """
#     Normalizes the RGB pixel values of an image.

#     Args:
#         image (3D NumPy array): For example, it should take in a single 3x32x32 image from the CIFAR-10 dataset
#     Returns:
#         tuple: The normalized image
#     """
    
#     mean = np.mean(image,axis=(1,2,3))
#     std = np.std(image,axis=(1,2,3))
#     transform = transforms.Compose([transforms.RandomHorizontalFlip(),tranforms.Normalize(mean=torch.from_numpy(mean),std=torch.from_numpy(std))])
#     return image


# In[21]:


# from numpy import array
# norm_images_train_flip = []
# for i in range(len(X)):
#     norm_images_train_flip.append(normalizeflip_image(X[i]))
# norm_images_x_train_flip = array(norm_images_train) 

# norm_images_test_flip = []
# for i in range(len(X_test)):
#     norm_images_test_flip.append(normalizeflip_image(X_test[i]))
# norm_images_x_test_flip = array(norm_images_test_flip) 


# In[ ]:


# # net = ConvolutionalNN()


# # net = ConvolutionalNN()
# neural_net = ConvolutionalNN()
# dataset_train3 = Dataset(norm_images_x_train_flip, y)
# train_loader = DataLoader(dataset=dataset_train3,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test3 = Dataset(norm_images_x_test_flip, y_test)
# test_loader = DataLoader(dataset=dataset_test3,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()


# # criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD




# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)


# max_epochs = 100

# loss_np_norm_flip = np.zeros((max_epochs))
# epoch_accuracy_norm_flip = np.zeros((max_epochs))
# test_accuracy_norm_flip = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_norm_flip= []
#     batch_accuracy_norm_flip =[]
#     test_acc_norm_flip =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_norm_flip.append(float(correct)/float(len(label_np)))    
#         loss_list_norm_flip.append(loss.data.numpy()) 
          
#     loss_np_norm_flip[epoch] =  np.mean(np.asarray(loss_list_norm_flip))  
#     epoch_accuracy_norm_flip[epoch] = np.mean(np.asarray(batch_accuracy_norm_flip))
#     print("epoch: ", epoch, "loss: ", loss_np_norm_flip[epoch], "epoch_acc", epoch_accuracy_norm_flip[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_norm_flip.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_norm_flip[epoch] = np.mean(np.asarray(test_acc_norm_flip))    
#     print("epoch", epoch, "test acc",test_accuracy_norm_flip[epoch])    
    
# print("Test Accuracy: ", test_accuracy_norm_flip[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_norm_flip[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_norm_flip)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_norm_flip)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_norm_flip)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_norm_flip)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[ ]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_norm_flip, color = 'b', label = 'train accuracy norm flip')#
# lns3 = ax1.plot(epoch_number, test_accuracy_norm_flip, color = 'orange', label = 'test accuracy norm flip')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_norm_flip, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for CNN normalised by horizontal flipping images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[23]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = ConvolutionalNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.Adagrad(neural_net.parameters(), lr=0.001)


# max_epochs = 100

# loss_np_ada = np.zeros((max_epochs))
# epoch_accuracy_ada = np.zeros((max_epochs))
# test_accuracy_ada = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_ada= []
#     batch_accuracy_ada =[]
#     test_acc_ada =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_ada.append(float(correct)/float(len(label_np)))    
#         loss_list_ada.append(loss.data.numpy()) 
          
#     loss_np_ada[epoch] =  np.mean(np.asarray(loss_list_ada))  
#     epoch_accuracy_ada[epoch] = np.mean(np.asarray(batch_accuracy_ada))
#     print("epoch: ", epoch, "loss: ", loss_np_ada[epoch], "epoch_acc", epoch_accuracy_ada[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_ada.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_ada[epoch] = np.mean(np.asarray(test_acc_ada))    
#     print("epoch", epoch, "test acc",test_accuracy_ada[epoch])    
    
# print("Test Accuracy: ", test_accuracy_ada[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_ada[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_ada)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_ada)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_ada)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_ada)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[24]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_ada, color = 'b', label = 'train accuracy adagrad')#
# lns3 = ax1.plot(epoch_number, test_accuracy_ada, color = 'orange', label = 'test accuracy adagrad')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_ada, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for CNN by using Adagrad optimiser')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[ ]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = ConvolutionalNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.Adam(neural_net.parameters(), lr=0.001)


# max_epochs = 100

# loss_np_adam = np.zeros((max_epochs))
# epoch_accuracy_adam = np.zeros((max_epochs))
# test_accuracy_adam = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_adam= []
#     batch_accuracy_adam =[]
#     test_acc_adam =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_adam.append(float(correct)/float(len(label_np)))    
#         loss_list_adam.append(loss.data.numpy()) 
          
#     loss_np_adam[epoch] =  np.mean(np.asarray(loss_list_adam))  
#     epoch_accuracy_adam[epoch] = np.mean(np.asarray(batch_accuracy_adam))
#     print("epoch: ", epoch, "loss: ", loss_np_adam[epoch], "epoch_acc", epoch_accuracy_adam[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_adam.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_adam[epoch] = np.mean(np.asarray(test_acc_adam))    
#     print("epoch", epoch, "test acc",test_accuracy_adam[epoch])    
    
# print("Test Accuracy: ", test_accuracy_adam[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_adam[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_adam)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_adam)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_adam)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_ada)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[24]:


# net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[14]:


# from numpy import array
# norm_images_train = []
# for i in range(len(X)):
#     norm_images_train.append(normalize_image(X[i]))
# norm_images_x_train = array(norm_images_train) 

# norm_images_test = []
# for i in range(len(X_test)):
#     norm_images_test.append(normalize_image(X_test[i]))
# norm_images_x_test = array(norm_images_test) 


# In[22]:


# net = ConvolutionalNN()
# dataset_train2 = Dataset(norm_images_x_train, y)
# train_loader = DataLoader(dataset=dataset_train2,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test2 = Dataset(norm_images_x_test, y_test)
# test_loader = DataLoader(dataset=dataset_test2,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[15]:


# net = ConvolutionalNN()
# dataset_train2 = Dataset(norm_images_x_train, y)
# train_loader = DataLoader(dataset=dataset_train2,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test2 = Dataset(norm_images_x_test, y_test)
# test_loader = DataLoader(dataset=dataset_test2,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[18]:


# net = FeedForwardNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[50]:


# net = HyperParamsFeedForwardNN()
# #2000,100 0.001
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[53]:


# net = HyperParamsFeedForwardNN()
# #2000,100 0.005
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[56]:


# net = HyperParamsFeedForwardNN()
# #2000,10 0.001,128-yes
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[61]:


# net = HyperParamsFeedForwardNN()
# #1000,100 0.001,128
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[75]:


# net = HyperParamsFeedForwardNN()
# #1500,100 0.001,128
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[79]:


# net = HyperParamsFeedForwardNN()
# #2000,100 0.005,128
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[83]:


# net = HyperParamsFeedForwardNN()
# #2000,500,10 0.001,64
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[57]:


# net = HyperParamsFeedForwardNN()
# #2000,100 0.001,256
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=256,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=256,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[20]:


# net = HyperParamsConvNN()
# #0.01,250/100
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[46]:


# net = HyperParamsConvNN()
# #0.005,90/30
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[26]:


# net = HyperParamsConvNN()
# #0.005,250/100-yes
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[32]:


# net = HyperParamsConvNN()
# #0.005,250/100,128
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[33]:


# net = HyperParamsConvNN()
# #0.005,250/100,256
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=256,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=256,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[38]:


# net = HyperParamsConvNN()
# #0.005,250/100,64, 16*10*10(5)
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[40]:


# net = HyperParamsConvNN()
# #0.005,250/100,64, 16*7*7(7)
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[30]:


# net = HyperParamsConvNN()
# #0.003,250/100, 128
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[57]:


# net = FeedForwardNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)


# In[16]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_ff, color = 'b', label = 'train accuracy raw')#
# lns3 = ax1.plot(epoch_number, test_accuracy_ff, color = 'orange', label = 'test accuracy raw')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_ff, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for Baseline Feed Forward raw images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[15]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = FeedForwardNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)


# max_epochs = 100

# loss_np_ff = np.zeros((max_epochs))
# epoch_accuracy_ff = np.zeros((max_epochs))
# test_accuracy_ff = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_ff= []
#     batch_accuracy_ff =[]
#     test_acc_ff =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
#         inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
# #         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_ff.append(float(correct)/float(len(label_np)))    
#         loss_list_ff.append(loss.data.numpy()) 
          
#     loss_np_ff[epoch] =  np.mean(np.asarray(loss_list_ff))  
#     epoch_accuracy_ff[epoch] = np.mean(np.asarray(batch_accuracy_ff))
#     print("epoch: ", epoch, "loss: ", loss_np_ff[epoch], "epoch_acc", epoch_accuracy_ff[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
#         test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
# #         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_ff.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_ff[epoch] = np.mean(np.asarray(test_acc_ff))    
#     print("epoch", epoch, "test acc",test_accuracy_ff[epoch])    
    
# print("Test Accuracy: ", test_accuracy_ff[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_ff[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_ff)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_ff)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_ff)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_ff)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[18]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=256,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=256,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = HyperParamsFeedForwardNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)


# max_epochs = 100

# loss_np_hyff = np.zeros((max_epochs))
# epoch_accuracy_hyff = np.zeros((max_epochs))
# test_accuracy_hyff = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_hyff= []
#     batch_accuracy_hyff =[]
#     test_acc_hyff =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
#         inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
# #         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_hyff.append(float(correct)/float(len(label_np)))    
#         loss_list_hyff.append(loss.data.numpy()) 
          
#     loss_np_hyff[epoch] =  np.mean(np.asarray(loss_list_hyff))  
#     epoch_accuracy_hyff[epoch] = np.mean(np.asarray(batch_accuracy_hyff))
#     print("epoch: ", epoch, "loss: ", loss_np_hyff[epoch], "epoch_acc", epoch_accuracy_hyff[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
#         test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
# #         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_hyff.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_hyff[epoch] = np.mean(np.asarray(test_acc_hyff))    
#     print("epoch", epoch, "test acc",test_accuracy_hyff[epoch])    
    
# print("Test Accuracy: ", test_accuracy_hyff[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_hyff[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_hyff)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_hyff)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_hyff)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_hyff)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[19]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_hyff, color = 'b', label = 'train accuracy raw')#
# lns3 = ax1.plot(epoch_number, test_accuracy_hyff, color = 'orange', label = 'test accuracy raw')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_hyff, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for HyperParameter Feed forward for raw images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[13]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = HyperParamsConvNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.005, momentum=0.9)


# max_epochs = 100

# loss_np_hycn = np.zeros((max_epochs))
# epoch_accuracy_hycn = np.zeros((max_epochs))
# test_accuracy_hycn = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_hycn= []
#     batch_accuracy_hycn =[]
#     test_acc_hycn =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_hycn.append(float(correct)/float(len(label_np)))    
#         loss_list_hycn.append(loss.data.numpy()) 
          
#     loss_np_hycn[epoch] =  np.mean(np.asarray(loss_list_hycn))  
#     epoch_accuracy_hycn[epoch] = np.mean(np.asarray(batch_accuracy_hycn))
#     print("epoch: ", epoch, "loss: ", loss_np_hycn[epoch], "epoch_acc", epoch_accuracy_hycn[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_hycn.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_hycn[epoch] = np.mean(np.asarray(test_acc_hycn))    
#     print("epoch", epoch, "test acc",test_accuracy_hycn[epoch])    
    
# print("Test Accuracy: ", test_accuracy_hycn[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_hycn[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_hycn)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_hycn)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_hycn)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_hycn)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[14]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_hycn, color = 'b', label = 'train accuracy raw')#
# lns3 = ax1.plot(epoch_number, test_accuracy_hycn, color = 'orange', label = 'test accuracy raw')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_hycn, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for HyperParameter CNN raw images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[64]:


# # net = ConvolutionalNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD



# neural_net = ConvolutionalNN()
# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)


# max_epochs = 100

# loss_np_raw = np.zeros((max_epochs))
# epoch_accuracy_raw = np.zeros((max_epochs))
# test_accuracy_raw = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_raw= []
#     batch_accuracy_raw =[]
#     test_acc_raw =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_raw.append(float(correct)/float(len(label_np)))    
#         loss_list_raw.append(loss.data.numpy()) 
          
#     loss_np_raw[epoch] =  np.mean(np.asarray(loss_list_raw))  
#     epoch_accuracy_raw[epoch] = np.mean(np.asarray(batch_accuracy_raw))
#     print("epoch: ", epoch, "loss: ", loss_np_raw[epoch], "epoch_acc", epoch_accuracy_raw[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_raw.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_raw[epoch] = np.mean(np.asarray(test_acc_raw))    
#     print("epoch", epoch, "test acc",test_accuracy_raw[epoch])    
    
# print("Test Accuracy: ", test_accuracy_raw[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_raw[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_raw)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_raw)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_raw)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_raw)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[65]:


# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_raw)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_raw)
# #     plt.title('test accuracy over epoches')
# plt.show()


# In[66]:


# from numpy import array
# norm_images_train = []
# for i in range(len(X)):
#     norm_images_train.append(normalize_image(X[i]))
# norm_images_x_train = array(norm_images_train) 

# norm_images_test = []
# for i in range(len(X_test)):
#     norm_images_test.append(normalize_image(X_test[i]))
# norm_images_x_test = array(norm_images_test) 


# In[67]:


# # net = ConvolutionalNN()


# # net = ConvolutionalNN()
# neural_net = ConvolutionalNN()
# dataset_train2 = Dataset(norm_images_x_train, y)
# train_loader = DataLoader(dataset=dataset_train2,
#                           batch_size=64,
#                           shuffle=True)
# dataset_test2 = Dataset(norm_images_x_test, y_test)
# test_loader = DataLoader(dataset=dataset_test2,
#                           batch_size=64,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()


# # criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD




# # train_loader = train_loader
# # test_loader = test_loader
# # criterion = loss_function
# optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)


# max_epochs = 100

# loss_np_norm = np.zeros((max_epochs))
# epoch_accuracy_norm = np.zeros((max_epochs))
# test_accuracy_norm = np.zeros((max_epochs))
    
# #     test_accuracy = np.zeros((max_epochs))
# #     epoch_number = np.arange(0,max_epochs,1)

# #     # Plot the loss over epoch
# #     plt.figure()
# #     plt.plot(epoch_number, loss_np)
# #     plt.title('loss over epoches')
# #     plt.xlabel('Number of Epoch')
# #     plt.ylabel('Loss')
    
# for epoch in range(max_epochs):
#     loss_list_norm= []
#     batch_accuracy_norm =[]
#     test_acc_norm =[]
#     for i, data in enumerate(train_loader, 0):
#             # Get inputs and labels from data loader 
#         inputs, labels = data
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels)

# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         inputs, labels = Variable(inputs), Variable(labels)    
            
            
#             # zero gradient
#         optimizer.zero_grad()
        
#             # Feed the input data into the network 
#         y_pred = neural_net(inputs)
        
#             # Calculate the loss using predicted labels and ground truth labels
#         loss = criterion(y_pred, labels)
#         loss_f = float(loss)
#             # backpropogates to compute gradient
#         loss.backward()
        
#             # updates the weghts
#         optimizer.step()
           
#         y_pred_np = y_pred.data.numpy()
        
#         pred_np = np.empty(len(y_pred_np))
#         for m in range(len(y_pred_np)):
#             pred_np[m] = np.argmax(y_pred_np[m])
                
#         pred_np = pred_np.reshape(len(pred_np), 1)

#         label_np = labels.data.numpy().reshape(len(labels), 1)
            
#         correct = 0
#         for j in range(y_pred_np.shape[0]):
#             if pred_np[j] == label_np[j]:
#                 correct += 1
#         batch_accuracy_norm.append(float(correct)/float(len(label_np)))    
#         loss_list_norm.append(loss.data.numpy()) 
          
#     loss_np_norm[epoch] =  np.mean(np.asarray(loss_list_norm))  
#     epoch_accuracy_norm[epoch] = np.mean(np.asarray(batch_accuracy_norm))
#     print("epoch: ", epoch, "loss: ", loss_np_norm[epoch], "epoch_acc", epoch_accuracy_norm[epoch])
   


        
#     for data in test_loader:
        
#         # Get inputs and labels from data loader 
#         test_inputs, test_labels = data
    
# #         if(isinstance(neural_net,FeedForwardNN) or isinstance(neural_net,HyperParamsFeedForwardNN)):
# #             test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels)
# #         elif(isinstance(neural_net,ConvolutionalNN) or isinstance(neural_net,HyperParamsConvNN)):
#         test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            
#         y_pred_test = neural_net(test_inputs)
                                            
#         y_pred_test_np = y_pred_test.data.numpy()
#         pred_test_np = np.empty(len(y_pred_test_np))
        
#         for k in range(len(y_pred_test_np)):
#             pred_test_np[k] = np.argmax(y_pred_test_np[k])

#         pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

#         label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

#         correct_test = 0
#         for j in range(y_pred_test_np.shape[0]):
#             if pred_test_np[j] == label_test_np[j]:
#                 correct_test += 1
#         test_acc_norm.append(float(correct_test) / float(len(label_test_np)))
#     test_accuracy_norm[epoch] = np.mean(np.asarray(test_acc_norm))    
#     print("epoch", epoch, "test acc",test_accuracy_norm[epoch])    
    
# print("Test Accuracy: ", test_accuracy_norm[max_epochs-1])  
# print("final training accuracy: ", epoch_accuracy_norm[max_epochs-1])
    

# epoch_number = np.arange(0,max_epochs,1)


# # Plot the training accuracy over epoch
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_norm)
# plt.title('training accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
    
    
#         # Plot the loss over epoch
# plt.figure()
# plt.plot(epoch_number, loss_np_norm)
# plt.title('loss over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('Loss')
    
    
# plt.figure()
# plt.plot(epoch_number, epoch_accuracy_norm)
# plt.title('training /test accuracy over epoches')
# plt.xlabel('Number of Epoch')
# plt.ylabel('accuracy')
# plt.plot(epoch_number, test_accuracy_norm)
# #     plt.title('test accuracy over epoches')
# plt.show()

    
   


# In[70]:


# # accuracyTrainRaw 
# # accuracyTrain
# # accuracyTestRaw
# # accuracyTest
# # loss_npRaw
# # loss_np


# loss_np_raw 
# epoch_accuracy_raw 
# test_accuracy_raw 
# loss_np_norm 
# epoch_accuracy_norm 
# test_accuracy_norm 
# max_epochs = 100

# epoch_number = np.arange(0,max_epochs,1)

# # Plot the training accuracy over epoch
# fig, ax1 = plt.subplots(facecolor = 'w')
# lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_raw, color = 'b', label = 'train accuracy raw')#
# lns3 = ax1.plot(epoch_number, test_accuracy_raw, color = 'orange', label = 'test accuracy raw')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_raw, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns2+lns3+lns4+lns5+lns6
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 3: training, testing accuracy & loss for CNN raw and normalized images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[84]:


# fig, ax1 = plt.subplots(facecolor = 'w')
# # lns2 = ax1.plot(epoch_number, epoch_accuracy_norm, color = 'b',label = 'train accuracy normalized', linestyle = 'dashdot')
# # lns4 = ax1.plot(epoch_number, test_accuracy_norm, color = 'orange', label = 'test accuracy normalized', linestyle = 'dashdot')#

# lns1 = ax1.plot(epoch_number, epoch_accuracy_raw, color = 'b', label = 'train accuracy raw')#
# lns3 = ax1.plot(epoch_number, test_accuracy_raw, color = 'orange', label = 'test accuracy raw')#
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# # lns6 = ax2.plot(epoch_number, loss_np_norm, color = 'g', label = 'loss normalized',linestyle = 'dashdot')#
# lns5 = ax2.plot(epoch_number, loss_np_raw, color = 'g', label = 'loss raw')#
# #ax2.set_ylim(1, 4)
# ax2.set_ylabel('Loss')
# ax1.set_xlabel('Number of epochs')

# # group stuff
# lns = lns1+lns3+lns5
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Experiment 2: training, testing accuracy & loss for Baseline CNN raw images')
# plt.xlabel('Number of Epochs', fontsize = 'xx-large')
# #plt.ylabel('Accuracy',fontsize = 'xx-large')
# plt.show()


# In[65]:


# net = HyperParamsConvNN()
# dataset_train = Dataset(X, y)
# train_loader = DataLoader(dataset=dataset_train,
#                           batch_size=128,
#                           shuffle=True)
# dataset_test = Dataset(X_test, y_test)
# test_loader = DataLoader(dataset=dataset_test,
#                           batch_size=128,
#                           shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD
# u, t, o = run_experiment(net, train_loader, test_loader, criterion, optimizer)

