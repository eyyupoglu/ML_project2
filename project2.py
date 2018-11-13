
# coding: utf-8

# TODOS:
# 1. Standartasation of hot encodeds,- I dont think we need it but maybe very at the end we can do it.
# 2. Generalisation error to linear regression -Done 

# ### Change the directory to the repository folder in your computer depends on wherever it is.

# In[1]:


cd /home/mehmet/Desktop/DTU_Fall_018/Introduction_machine_learning/project2/ML_project2/


# In[2]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]


# # PCA

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_excel('DataSet.xlsx')
df[-5:]

import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
from datetime import date

X = df.as_matrix(columns=df.columns[df.columns!='Date'])
N = len(df.columns) -1

std = np.std(X, axis=0)
std = np.expand_dims(std,axis=0)

X = X/std
# Subtract mean value from data

Y = X - np.ones((N))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V = V.T

Z = Y @ V[:,:2]


data = pd.DataFrame({'PCA-1':Z[:,0],'PCA-2':Z[:,1]})  

ax = sns.scatterplot(x="PCA-1", y="PCA-2",
                      s=100, 
                     data=data)
ax.set_title('Projection of the centred data on the PC space', fontsize = 20)
sns.set(rc={'figure.figsize':(1,1)})
ax = ax.get_figure()
ax.savefig('projection_data.png')


# In[51]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import pandas as pd
import numpy as np
df = pd.read_excel('DataSet.xlsx')

df = df[(df.temp<=1) & (df.temp>0) & (df.atemp<=1) & (df.atemp>=0) & (df.hum<=1) & (df.hum>=0) & (df.windspeed <=1) ]
orig_y = df['new']
# df['new'] = np.log(((df['new'] - df['new'].mean()) / df['new'].std()).abs()).abs()
# np.log(((df['new'] - df['new'].mean()) / df['new'].std()).abs()).abs().hist()
df.head()
# plt.plot(df['new'])


# # Linear Model without generalisation

# In[52]:


import numpy as np
from sklearn.linear_model import LinearRegression
sub_df = df[[ 'season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
cols = [ 'season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

for each in cols:
    dummies = pd.get_dummies(sub_df[each], prefix=each)
    sub_df = sub_df.drop(each, axis = 1)
    sub_df = pd.concat([ dummies, sub_df], axis=1)
    
sub_df = pd.concat([ df[['hum']], sub_df], axis=1)
sub_df = pd.concat([ df[['temp']], sub_df], axis=1)
sub_df = pd.concat([ df[['atemp']], sub_df], axis=1)
sub_df = pd.concat([ df[['windspeed']], sub_df], axis=1)


X = sub_df.as_matrix()
y = np.asarray(df['new'])



reg = LinearRegression().fit(X , y)
reg.coef_


# In[53]:


import math
#Standardize the 1 out of K encoded variables
pd.options.mode.chained_assignment = None
df1 = sub_df[[ 'weathersit_1', 'weathersit_2', 'weathersit_3', 'workingday_0', 'workingday_1', 'windspeed', 'atemp', 'temp', 'hum',
            'mnth_1', 'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12',
            'season_1', 'season_2', 'season_3', 'season_4', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
            'weekday_6', 'holiday_0', 'holiday_1']]
for i in ('weathersit_1', 'weathersit_2', 'weathersit_3'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(3)
for i in ('workingday_0', 'workingday_1'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(2)
for i in ('weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(7)
for i in ('holiday_0', 'holiday_1'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(2)
for i in ('mnth_1', 'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(12)
for i in ('season_1', 'season_2', 'season_3', 'season_4'):
    df1[i] = (((df1[i] - df1[i].mean()) / df1[i].std()).abs())/math.sqrt(4)
sub_df1 = df1[['windspeed', 'atemp', 'temp', 'hum', 'holiday_0', 'holiday_1']]
sub_df = df1

reg = LinearRegression().fit(X , y)
reg.coef_


# In[55]:


sub_df.head()


# In[56]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
from matplotlib.pyplot import  plot, legend
predicted = plt.plot(range(len(X)), reg.predict(X), label = 'predicted')
original = plt.plot(range(len(X)), y, label = 'original')


plt.savefig('predicted_and_original.png')


# # Linear Model with Generalisation

# In[57]:


from sklearn import linear_model

def get_ridge_lm(alpha = 0.1):
    reg = linear_model.Ridge (alpha = alpha).fit(X[:200], y[:200])
    return reg


# In[58]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 20]
from matplotlib.pyplot import  plot, legend

X = sub_df.as_matrix()
y = np.asarray(df['new'])



alphas = [1, 4, 10, 100, 500, 1000]
for i in range(6):
    plt.subplot(3, 2, i+1)
    predicted = plot(get_ridge_lm(alpha = int(alphas[i])).predict(X[:200]), label = 'predicted')
    legend('predicted')
    original = plot(y[:200], label = 'original')
    legend('original')
    title = 'alpha = ' + str(alphas[i])
    plt.title(title, fontsize = 25)

plt.savefig('predicted_and_original2.png')

attributeNames = [name for name in list(sub_df)]

print(attributeNames)
print(list(np.around(get_ridge_lm(alpha = 1).coef_, decimals=2)))



# In[11]:


asd cd /home/mehmet/Desktop/DTU_Fall_018/Introduction_machine_learning/02450Toolbox_Python/Tools


# In[12]:


pwd


# # 2.Forward selection linear regression

# In[59]:


# exercise 6.2.1
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, savefig
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from Tools.toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

# Load data from matlab file

attributeNames = [name for name in list(sub_df)]
N, M = X.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 4
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.Ridge (fit_intercept=True, alpha = 1).fit(X_train, y_train)
    
#     m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation, display=textout)
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.Ridge (fit_intercept=True, alpha = 1).fit(X_train[:,selected_features], y_train)
        
        some_features = [attributeNames[int(el)] for el in selected_features]
        print(some_features)
        some_coefs = m.coef_
        some_coefs = np.around(some_coefs, decimals=3)
        print(some_coefs)
#         m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        plt.savefig('MSE_cross_validation.png')
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
#     print('Train indices: {0}'.format(train_index))
#     print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
plt.savefig('forward_selection2.png')

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=1 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.Ridge (fit_intercept=True, alpha = 1).fit(X[:,ff], y)
#     m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    y_est= m.predict(X[:,ff])
    residual=y-y_est
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')
    
savefig('Forward_selection.png')
show()

print('Ran Exercise 6.2.1')


# In[37]:


selected_list = [1,4,6,7,14,15,16,17, 26, 27,30, 31]
selected_list


# In[38]:


Features = np.zeros(34)
Features[selected_list] = 1
Features


# In[47]:


[attributeNames[ind] for ind in selected_list]


# In[48]:


m = lm.Ridge (fit_intercept=True, alpha = 10).fit(X[:500,selected_list], y[:500])

print([attributeNames[ind] for ind in selected_list])
m.coef_


# # Regression question 4

# In[49]:


# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats


attributeNames = [name for name in list(sub_df)]
N, M = X.shape
C = 2

# Normalize data
X = stats.zscore(X);
                
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = [ 4, 8, 16, 32]     # number of hidden units
n_train = len(n_hidden_units)             # number of networks trained in each k-fold
learning_goal = 1     # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates

# K-fold crossvalidation
K = 10                   # only three folds to speed up this example
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
best_errors = list()
best_hidden_n_list = list()
k=0


def glm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.Ridge(fit_intercept=True, alpha = 100).fit(X_train, y_train)
        validation_error[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        f=f+1
    return validation_error.mean()
    

for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    best_train_error = np.inf
    best_hidden_n = None
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units[i], 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            best_hidden_n = n_hidden_units[i]
            error_hist[range(len(train_error)),k] = train_error
    print('BEST HIDDEN UNIT AMOUNT: {0}...'.format(best_hidden_n))
    best_hidden_n_list.append(best_hidden_n)
    best_errors.append(best_train_error)
    
    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    k+=1
    #break

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,7));
subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
savefig('ann_regression.png')
show()

print('Ran Exercise 8.2.6')

#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer


# # Regression Question 4

# In[18]:


# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats

# Load data from matlab file
mat_data = X

y = X[:,-3].squeeze()
attributeNames = [name for name in list(sub_df)]
N, M = X.shape
C = 2

# Normalize data
X = stats.zscore(X);
                
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = [ 4, 8, 16, 32]     # number of hidden units
n_train = len(n_hidden_units)             # number of networks trained in each k-fold
learning_goal = 1     # stop criterion 1 (train mse to be reached)
max_epochs = 200         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates

# K-fold crossvalidation
K = 10                   # only three folds to speed up this example
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)*np.nan
errors_lm = np.zeros(K) * np.nan
errors_baseline = np.zeros(K) * np.nan

error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
best_errors = list()
best_hidden_n_list = list()
k=0


def glm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.Ridge(fit_intercept=True, alpha = 100).fit(X_train, y_train)
        validation_error[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        f=f+1
    return validation_error.mean()
    

for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    best_train_error = np.inf
    best_hidden_n = None
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units[i], 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            best_hidden_n = n_hidden_units[i]
            error_hist[range(len(train_error)),k] = train_error
    print('BEST HIDDEN UNIT AMOUNT: {0}...'.format(best_hidden_n))
    best_hidden_n_list.append(best_hidden_n)
    best_errors.append(best_train_error)
    

    y_est = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    #errors for linear regression model
    m = linear_model.Ridge(fit_intercept=True, alpha = 100).fit(X_train, y_train)    
    errors_lm[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    errors_baseline[k] = np.square(np.mean(y_train) - m.predict(X_test)).sum()/y_test.shape[0]
    
    print('Best ANN error: {0}...'.format(best_train_error))
    print('Best LM error: {0}...'.format(errors_lm[k]))
    print('Best Baseline error: {0}...'.format(errors_baseline[k]))
    
    k+=1
    #break

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,7));
#plotting the errors for linear regression
bar(range(0,K),errors_lm); title('Mean-square errors linear regression');

subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
savefig('ann_regression.png')
show()

print('Ran Exercise 8.2.6')

#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer


# In[25]:


from scipy import stats
# plt.bar(range(0,K), errors_baseline); title('Base line errors');
plt.bar(range(0,K), errors_lm); title('Mean-square errors linear regression');
plt.bar(range(0,K), errors); title('Mean-square errors');



show()

# [tstatistic, pvalue] = stats.ttest_ind(errors_lm, errors)
# print('errors_lm, errors')
# print(pvalue)

[tstatistic, pvalue] = stats.ttest_ind(errors_baseline, errors)
print('errors_baseline, errors')
print(pvalue)

[tstatistic, pvalue] = stats.ttest_ind(errors_lm, errors_baseline)
print('errors_lm, errors_baseline')
print(pvalue)

