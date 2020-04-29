#Splitting downloaded data into training,validation and testing data

import csv 
import os
import shutil

num_train=0
num_test=0
num_validation=0

#declaring path of destination and source folders
train_path='C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/training_data'
validation_path='C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/validation_data'
test_path='C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/testing_data'
source="C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/maestro-v2.0.0/"

#make folders if they do not exist
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(validation_path):
    os.mkdir(validation_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)


with open('C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/refined_data.csv',newline='') as csvfile:
    readerfile=csv.reader(csvfile, delimiter=',')
    for row in readerfile:

        #the data is trainingdata 
        if row[0]=="train":
            num_train+=1
            sourcePath=source+str(row[1])
            print(sourcePath)
            destination_path=train_path+str(row[1][4:])
            shutil.move(sourcePath,destination_path)   
            

        #this data is testing data
        elif row[0]=="test":
            num_test+=1
            sourcePath=source+str(row[1])
            destination_path=test_path+str(row[1][4:])
            shutil.move(sourcePath,destination_path)

        #this data is validation data
        elif row[0]=="validation":
            num_validation+=1
            sourcePath=source+str(row[1])
            destination_path=validation_path+str(row[1][4:])
            shutil.move(sourcePath,destination_path)
            
    #print(num_train)
    #print(num_validation)
    #print(num_test)
    print("done")