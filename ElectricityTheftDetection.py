from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Electricity Theft Detection in Power Grids with Deep Learning and Random Forests") 
main.geometry("1000x650")

global filename
global cnn_model
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X, Y
    global le
    global dataset
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset['client_id'] = pd.Series(le.fit_transform(dataset['client_id'].astype(str)))
    dataset['label'] = dataset['label'].astype('uint8')
    print(dataset.info())
    dataset.drop(['creation_date'], axis = 1,inplace=True)
    text.insert(END,str(dataset.head())+"\n\n")
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype('uint8')
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = Y.astype('uint8')
    text.insert(END,"Total records found in dataset to train Deep Learning : "+str(X.shape[0])+"\n\n")

   
def runCNN():
    global X, Y
    text.delete('1.0', END)
    global cnn_model
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    Y1 = to_categorical(Y)
    Y1 = Y1.astype('uint8')
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_model = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_model.load_weights("model/model_weights.h5")
        cnn_model._make_predict_function()   
        print(cnn_model.summary())
        
    else:
        counts = np.bincount(Y1[:, 0])
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]
        class_weight = {0: weight_for_0, 1: weight_for_1}
        cnn_model = Sequential() #creating RNN model object
        cnn_model.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
        cnn_model.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
        cnn_model.add(Dense(Y.shape[1], activation='softmax',kernel_initializer = "uniform")) #after building model need to predict two classes such as normal or Dyslipidemia disease
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
        print(cnn_model.summary()) #display rnn details
        hist = cnn_model.fit(X, Y1, epochs=20, batch_size=64,class_weight=class_weight)
        cnn_model.save_weights('model/model_weights.h5')            
        model_json = cnn_model.to_json()
        with open("model/model.json", "w") as json_file:
          json_file.write(model_json)
        json_file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2, random_state=0)
    y_test = np.argmax(y_test, axis=1)
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"CNN Precision : "+str(p)+"\n")
    text.insert(END,"CNN Recall    : "+str(r)+"\n")
    text.insert(END,"CNN FMeasure  : "+str(f)+"\n")
    text.insert(END,"CNN Accuracy  : "+str(f)+"\n\n")

def runCNNRF():
    global classifier
    global X, Y
    global cnn_model
    predict = cnn_model.predict(X)
    YY = []
    for i in range(len(predict)):
        val = np.argmax(predict[i])
        YY.append(val)
    YY = np.asarray(YY)
    extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
    XX = extract.predict(X)
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(XX, YY)
    classifier = rfc
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=0)
    predict = rfc.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"CNN with Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"CNN with Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"CNN with Random Forest FMeasure  : "+str(f)+"\n")
    text.insert(END,"CNN with Random Forest Accuracy  : "+str(f)+"\n\n")
    
def runCNNSVM():
    global X, Y
    global cnn_model
    predict = cnn_model.predict(X)
    YY = []
    for i in range(len(predict)):
        val = np.argmax(predict[i])
        YY.append(val)
    YY = np.asarray(YY)
    extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
    XX = extract.predict(X)
    rfc = svm.SVC()
    rfc.fit(XX, YY)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=0)
    predict = rfc.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"CNN with SVM Precision : "+str(p)+"\n")
    text.insert(END,"CNN with SVM Recall    : "+str(r)+"\n")
    text.insert(END,"CNN with SVM FMeasure  : "+str(f)+"\n")
    text.insert(END,"CNN with SVM Accuracy  : "+str(f)+"\n\n")
    

def runRandomForest():
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FMeasure  : "+str(f)+"\n")
    text.insert(END,"Random Forest Accuracy  : "+str(f)+"\n\n")

def runSVM():
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    rfc = svm.SVC()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall    : "+str(r)+"\n")
    text.insert(END,"SVM FMeasure  : "+str(f)+"\n")
    text.insert(END,"SVM Accuracy  : "+str(f)+"\n\n")    

def predict():
    global classifier
    global cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    test = test.values
    data = test
    extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
    test = extract.predict(test)
    predict = classifier.predict(test)
    for i in range(len(predict)):
        if predict[i] == 1:
            text.insert(END,str(data[i])+" ===> record detected as ENERGY THEFT\n\n")
        if predict[i] == 0:
            text.insert(END,str(data[i])+" ===> record NOT detected as ENERGY THEFT\n\n")     
    
def graph():
    df = pd.DataFrame([['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','F1 Score',fscore[0]],['CNN','Accuracy',accuracy[0]],
                       ['CNN-RF','Precision',precision[1]],['CNN-RF','Recall',recall[1]],['CNN-RF','F1 Score',fscore[1]],['CNN-RF','Accuracy',accuracy[1]],
                       ['CNN-SVM','Precision',precision[2]],['CNN-SVM','Recall',recall[2]],['CNN-SVM','F1 Score',fscore[2]],['CNN-SVM','Accuracy',accuracy[2]],
                       ['RF','Precision',precision[3]],['RF','Recall',recall[3]],['RF','F1 Score',fscore[3]],['RF','Accuracy',accuracy[3]],
                       ['SVM','Precision',precision[3]],['SVM','Recall',recall[3]],['SVM','F1 Score',fscore[3]],['SVM','Accuracy',accuracy[3]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Electricity Theft Detection in Power Grids with Deep Learning and Random Forests', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Electricity Theft Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Generate CNN Model", command=runCNN)
cnnButton.place(x=200,y=150)
cnnButton.config(font=font1) 

cnnrfButton = Button(main, text="CNN with Random Forest", command=runCNNRF)
cnnrfButton.place(x=500,y=150)
cnnrfButton.config(font=font1)

cnnsvmButton = Button(main, text="CNN with SVM", command=runCNNSVM)
cnnsvmButton.place(x=200,y=200)
cnnsvmButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=500,y=200)
rfButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=200,y=250)
svmButton.config(font=font1)

predictButton = Button(main, text="Predict Electricity Theft", command=predict)
predictButton.place(x=500,y=250)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=800,y=250)
graphButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
