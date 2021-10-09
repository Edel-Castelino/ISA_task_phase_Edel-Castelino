'''
Author: Edel Castelino
ISA taskphase :week 1
Implement Linear Regression
dataset:https://www.kaggle.com/aungpyaeap/fish-market/

'''
import pandas as pd                 # linear algebra
import numpy as np                  # data processing

fr= open("FishList.csv","r")            #open the file
data_df = pd.read_csv(fr)               #reads the data from the file

#print(data_df.head())              # this line prints the first 5 rows

# dependent variable is Species
# indpendent variables are Weight, Length1, Length2, Length3, Height, Width
'''
x= data_df.drop(['Species'],axis=1).values
y= data_df['Species'].values
'''
fish_map = {'Bream':1 , 'Roach':2, 'Whitefish':3, 'Parkki':4, 'Perch':5, 'Pike':6, 'Smelt':7}
        #giving an int value to the type of fish

data_df = data_df.replace(to_replace=fish_map)

col_name = data_df.columns[0]
data_df.rename(columns={col_name:'Species'},inplace=True)

'''
print(data_df.head)
input()
'''

x= np.array(data_df.drop(['Species'],1))
y= np.array(data_df['Species'])

#split data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=0)
#print(x_train)
#print(y_train)

#train the model on the training set
from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train, y_train)

#predict the test set results
y_pred= ml.predict(x_test)
print(y_pred)

#to further evaluate the accuracy of the model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))         

#plot the results
import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("actual vs predicted")
plt.show()








