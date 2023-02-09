import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
le = LabelEncoder()

dataset = pd.read_csv('Dataset/ElectricityTheft.csv')
dataset.fillna(0, inplace = True)

dataset['client_id'] = pd.Series(le.fit_transform(dataset['client_id'].astype(str)))
dataset['label'] = dataset['label'].astype('uint8')
print(dataset.info())
dataset.drop(['creation_date'], axis = 1,inplace=True)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
Y = Y.astype('uint8')
print(X)
print(Y)

unique, frequency = np.unique(Y, return_counts = True)
print(str(unique)+" "+str(frequency))

#X = normalize(X)
print(X.shape)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = Y.astype('uint8')

Y = to_categorical(Y)
print(Y)
Y = Y.astype('uint8')
counts = np.bincount(Y[:, 0])
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
'''
class_weight = {0: weight_for_0, 1: weight_for_1}
cnn_model = Sequential() #creating RNN model object
cnn_model.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
cnn_model.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
cnn_model.add(Dense(Y.shape[1], activation='softmax',kernel_initializer = "uniform")) #after building model need to predict two classes such as normal or Dyslipidemia disease
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
print(cnn_model.summary()) #display rnn details
hist = cnn_model.fit(X, Y, epochs=20, batch_size=64,class_weight=class_weight)
cnn_model.save_weights('model/model_weights.h5')            
model_json = cnn_model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()

'''
with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
classifier.load_weights("model/model_weights.h5")
classifier._make_predict_function()   
print(classifier.summary())
cnn_model = classifier

test = pd.read_csv('Dataset/test.csv')
test.fillna(0, inplace = True)
#test['client_id'] = pd.Series(le.fit_transform(test['client_id'].astype(str)))
#test.drop(['creation_date'], axis = 1,inplace=True)
test = test.values
#test = normalize(test)
count = 0
predict = cnn_model.predict(X)
Y = []
for i in range(len(predict)):
    val = np.argmax(predict[i])
    Y.append(val)
    if val == 0 and count < 10:
        print(X[i])
        count = count + 1
        
Y = np.asarray(Y)

extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
X = extract.predict(X)
rfc = RandomForestClassifier(n_estimators=200, random_state=0)
rfc.fit(X, Y)

test = extract.predict(test)
print(test.shape)
print(X.shape)
predict = rfc.predict(test)
print(predict)

