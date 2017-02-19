
# coding: utf-8

# In[1]:

import numpy as np
from numpy.linalg import inv


# In[2]:

#normalises training data and returns normalized training data along with the computed mean and sd
def train_normalize(x):
    mean_values = []
    std_values = []
    x_norm = x
    for i in range(x.shape[1]):
        m = np.mean(x[:,i])
        s = np.std(x[:,i])
        mean_values.append(m)
        std_values.append(s)
        x_norm[:,i] = (x_norm[:,i] - m) / s
    return x_norm,mean_values,std_values

#normalizes training data based on the mean and sd computed from the training data
def test_normalize(x,m,s):
    x_norm = x
    for i in range(x.shape[1]):
        x_norm[:,i] = (x_norm[:,i] - m[i]) / s[i]
    return x_norm


#needs to be modified for making it better #numpy library for norm can be tried to make it faster
def gradient_descent_p_norm(phi,y,alpha,lamb,threshold,p_norm):
    print("Computing weights using gradient_descent....")
    converged = False
    M = phi.shape[0]
    weight = np.random.rand(phi.shape[1])
    iterations = 0
    error = (np.sum((np.dot(phi,weight) - y)**2))/(2.0 * M)
    prev_error = error + lamb*np.sum(np.power(np.absolute(weight),p_norm))
    
    while not converged:
        iterations = iterations + 1
        loss = np.dot(phi,weight) - y
        grad = (np.dot(phi.T,loss))/M + (p_norm * lamb * (np.power(np.absolute(weight),p_norm-1)))
        weight = weight - (alpha * grad)
        cur_error = (np.sum((np.dot(phi,weight) - y)**2))/(2.0 * M)
        new_error = cur_error + lamb*np.sum(np.power(np.absolute(weight),p_norm))
        if(abs(prev_error-new_error) < threshold):
            converged = True
        prev_error = new_error
        
    print("Done... Total Iterations:"+ str(iterations))        
    return weight


#process of feature engineering
#any features to be adeded should be added here
def increase_features(features):
    features=np.absolute(features)
    to_add1=np.power(features,1.8)
    to_add2=np.power(features,0.3)
    features=np.column_stack((features,to_add1))
    features=np.column_stack((features,to_add2))
    return features





# In[3]:

#returns training data as phi and y and data for cross validation
def get_data(learning_percent):
    data=np.genfromtxt("data/train.csv",delimiter="," ,skip_header=1)
    test_data=np.genfromtxt("data/test.csv",delimiter="," ,skip_header=1)
    
    learn_end=int(learning_percent*data.shape[0])
    train_features=data[:learn_end,1:-1] #excluding the ids and the actual output
    train_output=np.array(data[:learn_end,-1:] )# actual output or y
    cross_features=data[learn_end:,1:-1]
    cross_output=data[learn_end:,-1:]

    test_ids=test_data[:,:1]
    test_features=test_data[:,1:]
    
    #feature engineering
    print("Feature Engineering on Data..")
    train_features=increase_features(train_features)
    test_features=increase_features(test_features)
    
    #normalizing the data
    print("Normalizing Data....")
    train_features,mean_values,std_values=train_normalize(train_features)
    test_features=test_normalize(test_features,mean_values,std_values)
    
    train_features=np.column_stack((np.ones((train_features.shape[0],1)),train_features))
    test_features=np.column_stack((np.ones((test_features.shape[0],1)),test_features))
    return train_features,train_output,cross_features,cross_output,test_features,test_ids

#Functions produces the result based on the weight passed 
#make sure that the test_ids and test_features are not modified
def get_result(test_features,weights,test_ids,file_name):
    predicted = np.matmul( test_features,weights)
    output=np.column_stack((test_ids,predicted))
    np.savetxt(file_name,output,delimiter=",",fmt="%d,%.2f",header="ID,MEDV",comments ='')


def closed_form(phi,y,lamb):
    data=phi
    Y=y
    weights=np.dot(np.dot(inv(np.dot(data.T,data) + lamb*np.identity(data.shape[1])),data.T),Y)
    return weights


# In[6]:
print("Starting....")
print("Extracting Data....")
train_features,train_output,cross_features,cross_output,test_features,test_ids=get_data(0.4)
#print(train_features.shape,train_output.shape,cross_features.shape,cross_output.shape,test_features.shape)

phi=train_features
y=train_output
trainY=( val[0] for val in y);
trainY=list(trainY);
y=np.array(trainY)

#weights = closed_form(phi,y,0.0001)
#print(weights)

print("Processing for L2 Norm....")
#def gradient_descent_p_norm(phi,y,alpha,lamb,threshold,p_norm)
weights=gradient_descent_p_norm(phi,y,0.01,0.00009,0.000001,2)
print ("weights")
print(weights)
print ("Writing the predicted values to csv")
get_result(test_features,weights,test_ids,"output.csv")

print("Processing for L1.2 Norm....")
weights=gradient_descent_p_norm(phi,y,0.01,0.00009,0.000001,1.2)
print ("weights")
print(weights)
print ("Writing the predicted values to csv")
get_result(test_features,weights,test_ids,"output_p1.csv")

print("Processing for L1.3 Norm....")
weights=gradient_descent_p_norm(phi,y,0.01,0.00009,0.000001,1.3)
print ("weights")
print(weights)
print ("Writing the predicted values to csv")
get_result(test_features,weights,test_ids,"output_p2.csv")

print("Processing for L1.4 Norm....")
weights=gradient_descent_p_norm(phi,y,0.01,0.00009,0.000001,1.4)
print ("weights")
print(weights)
print ("Writing the predicted values to csv")
get_result(test_features,weights,test_ids,"output_p3.csv")




