#!/usr/bin/env python
# coding: utf-8

# In[25]:


import glob
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

from sklearn.model_selection import StratifiedKFold

from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


# In[2]:


from  torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_auc_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay 


# In[3]:



#Takes location of pickled files, and returns two dictionaries containing
#probability arrays for HE and TFF3.
class dataloading:
    
    #Path contains the HE and TFF3 folders#    
    def __init__(self, path):
        self.filenames_HE = (glob.glob(path+'he/*.p'))
        self.filenames_TFF3 = (glob.glob(path+'tff3/*.p'))
        
        self.Names_HE = []
        self.Arrays_HE = []
        self.Dict_HE = {}
        
        self.Names_TFF3 = []
        self.Arrays_TFF3 = []
        self.Dict_TFF3 = {}
        
        self.n_HE_classes = 4
        self.n_TFF3_classes = 3
        
      
    def unpickle(self, filename):
        
        infile = open(filename, 'rb')
        output_file = pickle.load(infile)
        infile.close
        return output_file 
    
    def to_Array_HE(self, Data_HE):
        dimx, dimy = Data_HE['maskSize'][0], Data_HE['maskSize'][1]
        A = np.empty([dimx, dimy, self.n_HE_classes]) #Constructing an empty 3-D array of the correct size
        for m in range(dimy):
            for n in range(dimx):
                if 'prediction' in Data_HE['tileDictionary'][(n, m)]:
                    A[n, m] = Data_HE['tileDictionary'][(n, m)]['prediction'] # If a prediciton key exists, adding in the probabilities at each x, y position for a particular array.             
                else:     
                    A[n, m] = [0.0, 0.0, 0.0, 0.0] #Otherwise, filling in the space with zeros
        return A    
    
    def to_Array_TFF3(self, Data_TFF3):
        dimx, dimy = Data_TFF3['maskSize'][0], Data_TFF3['maskSize'][1]
        A = np.empty([dimx, dimy, self.n_TFF3_classes]) #Constructing an empty 3-D array of the correct size
        for m in range(dimy):
            for n in range(dimx):
                if 'prediction' in Data_TFF3['tileDictionary'][(n, m)]:
                    A[n, m] = Data_TFF3['tileDictionary'][(n, m)]['prediction'] # If a prediciton key exists, adding in the probabilities at each x, y position for a particular array.             
                else:     
                    A[n, m] = [0.0, 0.0, 0.0] #Otherwise, filling in the space with zeros
        return A            
    
    
    def extract_to_Dict(self):
        
        for filename in self.filenames_HE:      
        
            Data_HE = self.unpickle(filename) # unpickle the file
            Name_HE = os.path.splitext(os.path.basename(filename))[0] 
            Array_HE = self.to_Array_HE(Data_HE)

            self.Names_HE.append(Name_HE)
            self.Arrays_HE.append(Array_HE)

            self.Dict_HE[Name_HE] = Array_HE
            
        for filename in self.filenames_TFF3:      
        
            Data_TFF3 = self.unpickle(filename) # unpickle the file
            Name_TFF3 = os.path.splitext(os.path.basename(filename))[0] 
            Array_TFF3 = self.to_Array_TFF3(Data_TFF3)

            self.Names_TFF3.append(Name_TFF3)
            self.Arrays_TFF3.append(Array_TFF3)

            self.Dict_TFF3[Name_TFF3] = Array_TFF3
           


# In[4]:


##Extract Ground Truths into Pandas array, and allows for 
##onces to be called from the object 
class ground_truth():
    def __init__(self, path):
        self.DataFrame_ground_truths = pd.read_csv(path)
        self.DataFrame_ground_truths.insert(0, 
                                       'Patient_identifier', 
                                       [self.DataFrame_ground_truths['Case'][i].split('_')[0] for i in range(len(self.DataFrame_ground_truths))]) 
        self.DataFrame_ground_truth_endoscopy = self.DataFrame_ground_truths[['Patient_identifier', 'Endoscopy (at least C1 or M3) + Biopsy (IM)']].copy()
    


# In[5]:


##Find and/or plot the n highest probability points for each
##class type
class n_squares(dataloading):
    def __init__(self, model):
        pass

        
    def find_highest_n_points(self, array, n, pad):
        
        self.array_pad = np.pad(array, [(pad, pad), (pad, pad)], mode='constant')        
        indices =  np.argpartition(self.array_pad.flatten(), -n)[-n:]    
        self.indices = np.vstack(np.unravel_index(indices, self.array_pad.shape)).T
    
    def plot_highest_n_points(self, array, n):
        indices =  np.argpartition(array.flatten(), -n)[-n:]    
        indices = np.vstack(np.unravel_index(indices, array.shape)).T
        
        x, y = indices[:, 0], indices[:, 1]
        z = [array[i, j] for i, j in indices]

        axs1 = plt.axes()
        axs1.scatter(x, y, c = z, cmap='cool')
        axs1.set_title(str(n) + ' largest points \n 2D')
        axs1.set_xlabel('X')
        axs1.set_ylabel('Y')
        plt.show()

        axs2 = plt.axes(projection="3d")
        axs2.scatter3D(x, y, z, c=z, cmap='cool')
        axs2.set_xlabel('X')
        axs2.set_title('3D')
        axs2.set_ylabel('Y')
        axs2.set_zlabel('Probabilitity')
        plt.show()

        return 

    def plot_cumulative_freq(self, array, n):
        indices =  np.argpartition(array.flatten(), -n)[-n:]    
        indices = np.vstack(np.unravel_index(indices, array.shape)).T
        x, y = indices[:, 0], indices[:, 1]
        z = [array[i, j] for i, j in indices]
        d = sorted(z)

        plt.plot(d)
    


# In[6]:


class preprocessing_n_arrays():
    def __init__(self, data, n, window_size):
        pad = int((window_size-1)/2) # Calculate padding size. Window_size should be odd.
        self.processed_data = {} # Empty dict to store data
        self.window_size = window_size
        classes = ['Equivocal', 'Negative', 'Positive'] # class labels
        data_n = n_squares(data) #call class to allow calculation of points
        self.n = n
        
        for names in data.Names_TFF3: # iterate through each patient
            array_pad = np.pad(data.Dict_TFF3[names], # pad array
                               [(2, 2), (2, 2),(0, 0)], 
                               mode='constant')
            dict_data = {} # create dict to store data for each class

            for z in range(data.Dict_TFF3[names].shape[2]):
                data_n.find_highest_n_points(data.Dict_TFF3[names][:,:,z], 100, pad)  
                list_data = []

                for i, j in data_n.indices:
                    list_data.append(array_pad[i-pad:i+pad+1, j-pad:j+pad+1, 0:3])
 
                dict_data[classes[z]] = sorted(list_data, key=lambda x: np.sum(x), reverse=True)
            
            dict_data['Ground_truth'] = ground_truth1.DataFrame_ground_truths[ground_truth1.DataFrame_ground_truths['Case'] == names]['Endoscopy (at least C1 or M3) + Biopsy (IM)'].bool()
            self.processed_data[names] = dict_data


# In[7]:


## Autoencoders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


# In[8]:


##Define LINEAR AUTOENCODER
class AE_linear(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=40
        )
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=40, out_features=20
                                              )
        self.encoder_output_layer = nn.Linear(
            in_features=20, out_features=10
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=10, out_features=20
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=20, out_features=40)
        self.decoder_output_layer = nn.Linear(
            in_features=40, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        code1 = self.encoder_output_layer(activation)#this as code
        code2 = torch.relu(code1)#or after through ReLU
        activation = self.decoder_hidden_layer1(code2)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code2


# In[9]:


class TilesDatasetLabelled_linear(Dataset):
    def __init__(self, data, patient_id):
        self.data = data
        self.id = patient_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.id[idx], self.data[idx]  


# In[10]:


def data_split_linear(data, patient_id, batch_size):
    dataset = TilesDatasetLabelled_linear(data, patient_id)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    return all_loader


# In[11]:


class TilesDatasetLabelled_conv(Dataset):
    def __init__(self, data, patient_id):
        self.data = data
        self.id = patient_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return(self.id[idx], self.data[idx].transpose((2, 0, 1)))


# In[12]:


def data_split_conv(data, patient_id, batch_size):
    dataset = TilesDatasetLabelled_conv(data, patient_id)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    return all_loader


# In[29]:


def Linear_encoder_trainer(epochs, squares, patient_id,
                           batch_size, lr,
                           optimizer, criterion, classes):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = int(np.prod(squares.shape)/len(squares))
    all_loader = data_split_linear(squares, patient_id, batch_size)
    
    model_linear = AE_linear(input_shape=input_shape).to(device)
    if optimizer == 'Adam':
        optimizer = optim.Adam(model_linear.parameters(), lr=lr)
    if criterion == 'MSE':
        criterion = nn.MSELoss() 
    
    codes_list = []
    patient_id_list = []
    
    for epoch in range(epochs):
        loss = 0      
      
        for patient in all_loader:
            patient_id, data = patient
            data = data.view(-1, input_shape).float().to(device)
            optimizer.zero_grad()
            outputs, code = model_linear(data)

            codes_np = code.data.cpu().numpy()
            codes_list.append(codes_np)
            patient_id_list.append(patient_id)

            train_loss = criterion(outputs, data)
            train_loss.backward()

            optimizer.step()
            loss += train_loss.item()
            # compute the epoch training loss
          
        loss = loss / len(all_loader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, class = {}".format(epoch + 1, epochs, loss, classes))
        
    show_encoder_linear(all_loader, model_linear)    
    
    return patient_id_list, codes_list


# In[14]:


def Conv_encoder_trainer(epochs, squares, patient_id,
                           batch_size, lr,
                           optimizer, criterion, classes):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_loader = data_split_conv(squares, patient_id, batch_size)
    model_conv = AE_conv().to(device)
    if optimizer == 'Adam':
        optimizer = optim.Adam(model_conv.parameters(), lr=lr)
    if criterion == 'MSE':
        criterion = nn.MSELoss() 
    
    codes_list = []
    patient_id_list = []
    
    for epoch in range(epochs):
        loss = 0      
      
        for patient in all_loader:
            patient_id, data = patient
            data = data.float()
            optimizer.zero_grad()
            outputs, code = model_conv(data)

            codes_np = code.data.cpu().numpy()
            codes_list.append(codes_np)
            patient_id_list.append(patient_id)

            train_loss = criterion(outputs, data)
            train_loss.backward()

            optimizer.step()
            loss += train_loss.item()
            # compute the epoch training loss
          
        loss = loss / len(all_loader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, class = {}".format(epoch + 1, epochs, loss, classes))
        
    show_encoder_conv(all_loader, model_conv)    
    
    return patient_id_list, codes_list


# In[15]:


def show_encoder_linear(all_loader, model_linear):
        
            # obtain one batch of validation images
    dataiter = iter(all_loader)
    patient_id, images = dataiter.next()

    outputs, code = model_linear(images.view(-1, 75).float())
    outputs = torch.reshape(outputs, (10, 5, 5, 3))
    outputs = outputs.detach().numpy()

    images = images.numpy() # convert images to numpy for display# obtain one batch of test images

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(1, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(outputs[idx])


    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(1, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx])  


# In[16]:


def show_encoder_conv(all_loader, model_conv):
        
            # obtain one batch of validation images
    dataiter = iter(all_loader)
    patient_id, images = dataiter.next()

    images = images.float()
    outputs, code = model_conv(images)
    
    outputs = outputs.view(10, 3, 5, 5)
    # use detach when it's an output that requires_grad
    outputs = outputs.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(1, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(outputs[idx].transpose(1, 2, 0))

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(1, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx].permute(1, 2, 0))  


# In[17]:


def Linear_encoder(data, batch_size, epochs, optimizer, criterion, lr):
    
    TFF_3_classes = ['Equivocal', 'Negative', 'Positive']
    data_size = data.n * len(data.processed_data)
    
    patient_id = np.array([names for names in data.processed_data for _ in range(data.n)])
    dataframe_codes = pd.DataFrame({'Patient_ID_linear': patient_id})
    
    for classes in TFF_3_classes:
        #extract data from object
        squares = np.array([i for names in data.processed_data for i in data.processed_data[names][classes]])   
        
        #train model and extract codes and reconstructed images
        patient_id_list, codes_list = Linear_encoder_trainer(epochs, squares, patient_id,
                                                             batch_size, lr,
                                                             optimizer, criterion, classes)
        
        #flatten codes and patient ids
        codes_list_flat = [item for sublist in codes_list for item in sublist]
        patient_id_list_flat = [item for sublist in patient_id_list for item in sublist]

        #take the last epoch only
        interval = int(len(codes_list_flat)/epochs)
        
        #add to each class codes to master df
        dataframe_codes['Patient_ID' + str(classes)] = patient_id_list_flat[interval*(epochs-1):interval*(epochs)]
        dataframe_codes[classes] = codes_list_flat[interval*(epochs-1):interval*(epochs)] 
         
    return dataframe_codes


# In[18]:


def Conv_encoder(data, batch_size, epochs, optimizer, criterion, lr):
    
    TFF_3_classes = ['Equivocal', 'Negative', 'Positive']
    data_size = data.n * len(data.processed_data)
    
    patient_id = np.array([names for names in data.processed_data for _ in range(data.n)])
    df_codes = pd.DataFrame({'Patient_ID_conv': patient_id})
    
    for classes in TFF_3_classes:
        #extract data from object
        squares = np.array([i for names in data.processed_data for i in data.processed_data[names][classes]])   
        
        #train model and extract codes and reconstructed images
        patient_id_list, codes_list = Conv_encoder_trainer(epochs, squares, patient_id,
                                                             batch_size, lr,
                                                             optimizer, criterion, classes)
        
        #flatten codes and patient ids
        codes_list_flat = [item for sublist in codes_list for item in sublist]
        patient_id_list_flat = [item for sublist in patient_id_list for item in sublist]

        #take the last epoch only
        interval = int(len(codes_list_flat)/epochs)
        
        #add to each class codes to master df
        df_codes['Patient_ID' + str(classes)] = patient_id_list_flat[interval*(epochs-1):interval*(epochs)]
        df_codes[classes] = codes_list_flat[interval*(epochs-1):interval*(epochs)] 
         
        for names in data.processed_data:
            data.processed_data[names][str(classes)+'_code'] = df_codes[classes][df_codes['Patient_ID' + str(classes)] == names].to_numpy()
    
            
    return df_codes


# In[19]:


import torch.nn.functional as F

# define the conv NN architecture
class AE_conv(nn.Module):
    def __init__(self):
        super(AE_conv, self).__init__()
        ## encoder layers ##
        # conv layer  
        self.conv1 = nn.Conv2d(3, 50, 3, padding=0)  
        # conv layer
        self.conv2 = nn.Conv2d(50, 20, 3, padding=0)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(20, 50, 3)
        self.t_conv2 = nn.ConvTranspose2d(50, 3, 3)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        # add second hidden layer
        conv_code = F.relu(self.conv2(x))
        #conv_code = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        decode = F.relu(self.t_conv1(conv_code))
        # output layer (with sigmoid for scaling from 0 to 1)
        decode = F.sigmoid(self.t_conv2(decode))
                
        return decode, conv_code


# In[ ]:





# In[20]:


path_data = '/Users/prabhvirmarway/Documents/Working Notebooks/Data/tile-dicts/'
path_ground_truth = '/Users/prabhvirmarway/Documents/Working Notebooks/Data/anon_gt_TFF3.csv'


# In[21]:


data1 = dataloading(path_data)
data1.extract_to_Dict()
# load data from pickled files and extract to dict


# In[22]:


ground_truth1 = ground_truth(path_ground_truth)
# load ground truths in dataframe


# In[23]:


data1p = preprocessing_n_arrays(data1, n = 100, window_size = 5)
# n = number of highest points to grab for each class for each patient
# window_size = size of window taken around each point. Should be odd. 


# In[30]:


codesL = Linear_encoder(data1p, 
               batch_size = 10,
               epochs = 5,
               optimizer = 'Adam',
               lr = 0.001,
               criterion = 'MSE')
# autoencoder to extact a code from the n windows for each class for each patient;
# stored in a dataframe


# In[31]:


codesC = Conv_encoder(data1p, 
               batch_size = 10,
               epochs = 5,
               optimizer = 'Adam',
               lr = 0.001,
               criterion = 'MSE')


# In[32]:


data1p.processed_data['2fe614a9_TFF3_1'].keys()


# In[33]:


def LSMT_preprocess(data):
    rnn_patients = np.squeeze(np.array([names for names in data.processed_data]))
    rnn_targets = [data.processed_data[names]['Ground_truth'] for names in data.processed_data]
    rnn_targets = np.array(rnn_targets, dtype=bool)
    
    n_patients = len(data.processed_data)
    n_codes = data.n
    
    rnn_equiv_codes = np.array([i for names in data.processed_data for code in data.processed_data[names]['Equivocal_code'] for i in code])
    rnn_neg_codes = np.array([i for names in data.processed_data for code in data.processed_data[names]['Negative_code'] for i in code])
    rnn_pos_codes = np.array([i for names in data.processed_data for code in data.processed_data[names]['Positive_code'] for i in code])
    rnn_equiv_codes = np.reshape(rnn_equiv_codes, (n_patients, n_codes, -1))
    rnn_neg_codes = np.reshape(rnn_neg_codes, (n_patients, n_codes, -1))
    rnn_pos_codes = np.reshape(rnn_equiv_codes, (n_patients, n_codes, -1))
    
    data.Dict_rnn = {'Patients': rnn_patients,
                     'Equivocal': rnn_equiv_codes,
                     'Negative': rnn_neg_codes, 
                     'Positive': rnn_pos_codes,
                     'Truth': rnn_targets}
    
    return data.Dict_rnn


# In[34]:


LSMT_preprocess(data1p)


# In[ ]:





# In[35]:


def LSTM_k_fold(features_train, targets_train, 
                features_test, targets_test,
                batch_size, input_dim, seq_dim):
    
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
    train = TensorDataset(featuresTrain,targetsTrain)
    
    test = TensorDataset(featuresTest,targetsTest)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    
    num = random.randint(0,711)
    plt.imshow(inputs[6].reshape(input_dim,seq_dim))
    plt.axis("off")
    plt.title(str(targets[6]))
    plt.savefig('graph.png')
    plt.show()

    print('Length of training data', len(train_loader.dataset))
    print('Length of test data', len(test_loader.dataset))
    
    return features_train, train_loader, test_loader


# In[36]:


def LSTM_loader(inputs, targets, test_size, batch_size, input_dim, seq_dim):
    features_train, features_test, targets_train, targets_test = train_test_split(inputs,
                                                                                  targets,
                                                                                  test_size = 0.2,
                                                                                  random_state = 42)
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
    train = TensorDataset(featuresTrain,targetsTrain)
    
    test = TensorDataset(featuresTest,targetsTest)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    
    num = random.randint(0,711)
    plt.imshow(inputs[6].reshape(input_dim,seq_dim))
    plt.axis("off")
    plt.title(str(targets[6]))
    plt.savefig('graph.png')
    plt.show()

    print('Length of training data', len(train_loader.dataset))
    print('Length of test data', len(test_loader.dataset))
    
    return features_train, train_loader, test_loader
    


# In[37]:


def LSTM(inputs, targets, test_size, batch_size, n_iters,
                input_dim, hidden_dim, layer_dim, output_dim, seq_dim,
                error, lr, optimizer, k_fold, n_splits):
   
    
    if k_fold == False:
    
        features_train, train_loader, test_loader = LSTM_loader(inputs, targets, test_size, 
                                                                batch_size, input_dim, seq_dim)
        
        num_epochs = n_iters / (len(features_train) / batch_size)
        num_epochs = int(num_epochs)
        
        LSTM_trainer(input_dim, hidden_dim, layer_dim, output_dim, seq_dim,
                    error, lr, optimizer, num_epochs, train_loader, test_loader)

    
    if k_fold == True:
        skf = StratifiedKFold(n_splits=n_splits, shuffle =True, random_state = 42)
        skf.get_n_splits(inputs, targets)   
        
        fold_n = 0
        for train_index, test_index in skf.split(inputs, targets):
            features_train, features_test = inputs[train_index], inputs[test_index]
            targets_train, targets_test = targets[train_index], targets[test_index]
            
            features_train, train_loader, test_loader = LSTM_k_fold(features_train, targets_train, 
                                                                    features_test, targets_test,
                                                                    batch_size, input_dim, seq_dim)
            
            fold_n += 1
            print('Fold number', fold_n, 'out of', n_splits)
            num_epochs = n_iters / (len(features_train) / batch_size)
            num_epochs = int(num_epochs)
            print("Epochs to be run: ",num_epochs) 
            
            LSTM_trainer(input_dim, hidden_dim, layer_dim, output_dim, seq_dim,
                         error, lr, optimizer, num_epochs, train_loader, test_loader)
            
          
            
            
        


# In[38]:


def LSTM_trainer(input_dim, hidden_dim, layer_dim, output_dim, seq_dim,
                error, lr, optimizer, num_epochs, train_loader, test_loader):    
    
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    if error == 'BCELogit':
        error = nn.BCEWithLogitsLoss() 
    else:    
        error = nn.BCEWithLogitsLoss()
        
    if optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
             
    
    loss_list_test = []
    loss_list_train = []
    iteration_list = []
    accuracy_list = []
    count = 0
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_dim, input_dim).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            labels = labels.unsqueeze(1)

            loss_train = error(outputs, labels.float())

            # Getting gradients
            loss_train.backward()

            # Updating parameters
            optimizer.step()


            count += 1

            if count % 500 == 0:
                # Calculate Accuracy
                predictions_list = []
                labels_list = []
                outputs_list = []

                correct = 0
                total = 0
                size = 0
                loss_test = 0
                for images, labels in test_loader:
                    images = images.view(-1, seq_dim, input_dim)
                    # Forward pass only to get logits/output
                    outputs = model(images)
                    labels = labels.unsqueeze(1)
                    loss_test = error(outputs, labels.float())

                    loss_test += loss_test.data.item()*labels.shape[0]
                    size += labels.shape[0]




                    predictions = torch.round(torch.sigmoid(outputs))
                    predictions_list.append(predictions.detach().numpy())
                    labels_list.append(labels.detach().numpy())
                    outputs_list.append(outputs.detach().numpy())

                loss_list_test.append(loss_test/size)
                outputs_list = np.vstack(outputs_list)
                predictions_list = np.vstack(predictions_list)
                labels_list = np.concatenate(labels_list)

                prec, recall, thresholds = precision_recall_curve(labels_list, outputs_list)
                pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, average_precision = 0, estimator_name = 'LSTM' ).plot()

                fpr, tpr, thresholds = roc_curve(labels_list, outputs_list)
                plt.plot(fpr, tpr, label = roc_auc_score(labels_list, outputs_list))

                target_spec = 0.90
                spec = [1-i for i in fpr]
                idx_spec = min(enumerate(spec), key=lambda x: abs(x[1]-target_spec))


                #print('Specificity:' , spec[idx_spec[0]-2], 'Sensitivity:' , tpr[idx_spec[0]-2]) 
                print('Specificity:' , spec[idx_spec[0]-1], 'Sensitivity:' , tpr[idx_spec[0]-1])    
                print('Specificity:' , spec[idx_spec[0]], 'Sensitivity:' , tpr[idx_spec[0]])
                print('Specificity:' , spec[idx_spec[0]+1], 'Sensitivity:' , tpr[idx_spec[0]+1])
                #print('Specificity:' , spec[idx_spec[0]+2], 'Sensitivity:' , tpr[idx_spec[0]+2]) 

                print(classification_report(labels_list, predictions_list))
                print('Matthews_corrcoef' , matthews_corrcoef(labels_list, predictions_list))
                print('roc_auc_score' , roc_auc_score(labels_list, predictions_list))

                loss_list_train.append(loss_train.data.item())
                iteration_list.append(count)

                # Print Loss
                print('Iteration: {}. Loss_train: {}. Loss_test {}.'.format(count, loss_train.data.item(), loss_test/size))


# In[39]:


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 100, 27
        # out[:, -1, :] --> last time step hidden states 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 2
        return out


# In[40]:


inputs = data1p.Dict_rnn['Equivocal']
targets = data1p.Dict_rnn['Truth']


# In[ ]:


inputs.shape


# In[41]:


## Implement k-fold validation - works
## Identify errors
## multi-class data input for LSTM
## raw square inpit for LSTM
## Hyperparameter - number of squares, LSTM architecture and params
##set-up github data record re: Steve, ECL2 Amazon, Github repos etc. 
##Speak to Peter?


# In[42]:


LSTM(inputs = inputs, targets = targets, test_size = 0.2, batch_size = 10, n_iters = 4000,
    input_dim = 100, hidden_dim = 200, layer_dim = 3, output_dim = 1, seq_dim = 20,
    error = 'Adadelta', lr = 0.01, optimizer = 'BCELost', k_fold = True, n_splits = 5)


# In[ ]:





# In[ ]:



    
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




