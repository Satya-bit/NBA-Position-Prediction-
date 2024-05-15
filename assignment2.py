#Name- SATYA SHAH
#UTA ID-1002161494
#Data Mining Assignment-2

# Run code with the following command-  python .\assignment2.py nba_stats.csv,dummy_test.csv
#Note- STEPS FOR MY APPROACH
#I have keep the random state 0 as given in the instruction.
#I have used pearson correlation coeffecient. The threshold value is 0.85 i,e, I have removed the features highly correlated features whose threshold coeeficient is more than 0.85. I have also removed some redundant features.
#I have also done some feature engineering for this by creating some new features.
#For outliers I have done winsorization.
#Also done standardisation
#I have used Multi Layer Perceptron Classifier. I have tuned the hyperparameters and also used GridSearchCV for getting optimal hyperparameters
#K FOLD startified cross validation with value of K=10.
#Also made confusion matrix in the end.

#RESULTS
# I got 62.65% training accuracy and 58.86% on Test data(Random state=0)
# Average of K FOLD cross validation accuracy is 59%(Random state=0)
# Accuracy on dummy_test is 69.3%(This is high because sir told that the data of dummy_test is a sample from the nba_stats.csv)
# My model is not overfitting as all accuracies are almost close to eachother. Nor it is underfitted as the base class is 20%(5 positions to predict)

#Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import sys 
from sklearn.metrics import confusion_matrix

#Removing warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Reading the train and test data-File Handling(python .\assignment2.py nba_stats.csv,dummy_test.csv)
def file_check(file_name):
    try:
        data = pd.read_csv(file_name)
        if data.isnull().any().any():
            print("There are null values in the data.")
        return data
    except FileNotFoundError:
        print("File not found:", file_name)
        exit(0)

file_list=sys.argv[1].split(',')
if(len(file_list)!=2):
    print("Invalid number of arguments!!")
else:
    data = file_check(file_list[0])
    test_data = file_check(file_list[1])


# Handling blank spaces in columns by replacing them with 0(Though there are no null values in training data)
data.fillna(0, inplace=True)



# #Heatmap of pearson correlation
# ## Data preprocessing- Pearson correlation
# # Encode the 'POS' column for pearson correlation matrix
# label_encoder = LabelEncoder()
# data['POS_encoded'] = label_encoder.fit_transform(data['Pos'])
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Compute the correlation matrix
# corr = data.corr()

# # Set up the matplotlib figure
# plt.figure(figsize=(25, 15))

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# # Add title
# plt.title('Correlation Heatmap')

# # Show plot
# plt.show()





# Created a new binary feature based on '3P%' column, This means most of the PointForward and Centre position players donot take 3pointer shots as they are inside the ring. So I created new binary column whether the players are taking 3P shots or not  
data['3P%_binary'] = data['3P%'].apply(lambda x: 0 if x == 0 else 1)


# Created BLK_custom feature which will make 0 if there is 0 in BLK and the position of the player is PG,SF and SG else it will be 1. This is because majority of the blocks are done by C and PF players as they stand nearby the ring(basket).
data['BLK_custom'] = np.where((data['Pos'].isin(['PG','SF','SG'])) & (data['BLK'] == 0), 0, 1)

# Created ORB_custom feature which will make 0 if there is 0 in ORB and the position of the player is PG,SF and SG else it will be 1. This is because majority of the Offensive Rebounds are done by C and PF players as they stand nearby the ring(basket).
data['ORB_custom'] = np.where((data['Pos'].isin(['PG','SF','SG'])) & (data['ORB'] == 0), 0, 1)

# Created STL_custom feature which will make 0 if there is 0 in STL and the position of the player is C and PF else it will be 1. This is because majority of the steals are done by PG,SF,SG players as they stand nearby the ring(basket).
data['STL_custom'] = np.where((data['Pos'].isin(['C','PF'])) & (data['STL'] == 0), 0, 1)
data = data[data['MP'] >= 5]#Filtering data of players who played less than 5 minutes 
data = data[data['G'] >= 5]#Filtering data of players who played less than 5 games

# Dropping irrelevant columns

#eFG% and DRB can be removed because it is highly coorelated with FG% and TRB respectively.
#Also 'G','GS','MP','Age' can be removed because it has less correlation with y(position) variable and cannot be used for predicting position of players.
#FG-FGA,2P-2PA,3P-3PA,FT-FTA are also removed as in the dataset FG%,2P%,3P% and FT% is already there and they are calculated from it.

columns_to_drop = [	'Age','G','GS','MP','FG','FGA','3P','3PA','2P','2PA','eFG%','FT','FTA','DRB','PF','TOV']
data.drop(columns=columns_to_drop, inplace=True)



# Defining the features (X) and the target (y)
X = data.drop(columns=['Pos'])
y = data['Pos']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data into training and testing sets(80-20)
X_train, X_validation, y_train, y_validation = train_test_split(X, y_encoded, test_size=0.2, random_state=0)



# Winsorizing the feature variables to handle outliers
X_train_winsorized = winsorize(X_train.to_numpy(), limits=[0.05, 0.05])
X_validation_winsorized = winsorize(X_validation.to_numpy(), limits=[0.05, 0.05])


# Standardizing the winsorized feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_winsorized)
X_validation_scaled = scaler.transform(X_validation_winsorized)



# Initializing the MLPClassifier
mlp_classifier = MLPClassifier(max_iter=70,hidden_layer_sizes=(20,50,), activation='relu', solver='adam', random_state=0,alpha=0.001,learning_rate_init=0.001)

# Training the classifier
mlp_classifier.fit(X_train_scaled, y_train)


# Making predictions on the training data
y_pred_train = mlp_classifier.predict(X_train_scaled)


# Making predictions on the test data
y_pred_validation = mlp_classifier.predict(X_validation_scaled)


# Evaluating the predictions on training data
accuracy_train = accuracy_score(y_train, y_pred_train)
print("MLP Classifier - Training Accuracy:", accuracy_train)

# Evaluating the predictions
accuracy_validation = accuracy_score(y_validation, y_pred_validation)
print("MLP Classifier - Validation Accuracy:", accuracy_validation)
###################################################################################################


#Confusion matrix for train and validation############################## 
decoded_labels = list(label_encoder.inverse_transform(range(len(label_encoder.classes_))))

# Converting y_train and y_pred_train to arrays of integers
y_train_int = np.array(y_train, dtype=int)
y_pred_train_int = np.array(y_pred_train, dtype=int)

# Computing the confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train_int, y_pred_train_int, labels=range(len(decoded_labels)))

# Printing the confusion matrix for training data with labels
print("Confusion Matrix - Training Data:")
print(pd.DataFrame(conf_matrix_train, index=decoded_labels, columns=decoded_labels))

# Converting y_test and y_pred_test to arrays of integers
y_test_int = np.array(y_validation, dtype=int)
y_pred_validation_int = np.array(y_pred_validation, dtype=int)

# Computing the confusion matrix for test data
conf_matrix_validation = confusion_matrix(y_test_int, y_pred_validation_int, labels=range(len(decoded_labels)))

# Printing the confusion matrix for test data with labels
print("\nConfusion Matrix - Validation Data:")
print(pd.DataFrame(conf_matrix_validation, index=decoded_labels, columns=decoded_labels))
####################################################################################################





####KFOLD Stratified Cross validation######

X_combined_scaled = np.vstack((X_train_scaled, X_validation_scaled))###Combining the train and test data and applying stratified cross validation
y_combined = np.concatenate((y_train, y_validation))
skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)

scores = cross_val_score(mlp_classifier, X_combined_scaled, y_combined, cv=skf)
print("Startified Cross-validation scores of each Fold: {}".format(scores))
print("Average stratified cross-validation score: {:.2f}".format(scores.mean()))
############################################################################################







######Test Data#########
# Reading the test data from the CSV file
# test_data = pd.read_csv('C:/Users/satya/Documents/utasem2/DM/P2/dummy_test.csv')


# Handling blank spaces in columns by replacing them with 0
test_data.fillna(0, inplace=True)

# Checking if the "Predicted Pos" column exists in the test data(Because this column was there in the dummy_test)
if 'Predicted Pos' in test_data.columns:
    # If it exists, drop the column
    test_data.drop(columns=['Predicted Pos'], inplace=True)
#Making the dimension of test data same for prediction 
test_data['3P%_binary'] = test_data['3P%'].apply(lambda x: 0 if x == 0 else 1)
test_data['BLK_custom'] = np.where((test_data['Pos'].isin(['PG','SF','SG'])) & (test_data['BLK'] == 0), 0, 1)
test_data['ORB_custom'] = np.where((test_data['Pos'].isin(['PG','SF','SG'])) & (test_data['ORB'] == 0), 0, 1)
test_data['STL_custom'] = np.where((test_data['Pos'].isin(['C','PF'])) & (test_data['STL'] == 0), 0, 1)

#Dropping columns
test_data.drop(columns=columns_to_drop, inplace=True)

X1 = test_data.drop(columns=['Pos'])
y1 = test_data['Pos']

label_encoder = LabelEncoder()
y1_encoded = label_encoder.fit_transform(y1)


# Preprocessing the test data
# Winsorize the feature variables to handle outliers
test_data_winsorized = winsorize(X1.to_numpy(), limits=[0.05, 0.05])

# Standardize the winsorized feature variables
test_data_scaled = scaler.transform(test_data_winsorized)

# Makin predictions on the test data
y_pred_test = mlp_classifier.predict(test_data_scaled)

# print(y_pred_test)
#Evaluating the predictions
accuracy_test = accuracy_score(y1_encoded, y_pred_test)
print("MLPC - Test Accuracy on Dummy test:", accuracy_test)
####################################################################################3


###Confusion Matrix of Test data###

## From the below confusion matrix it can be seen that [SF and PF], [C and PF] are missclassified because they have same stats. This means the model is not too bad for classifying positions.   
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Inversing transform the encoded labels to get the original class names
decoded_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

y1_encoded_int=np.array(y1_encoded, dtype=int)
y_pred_test_int=np.array(y_pred_test, dtype=int)

# Computing the confusion matrix for training data
conf_matrix_test = confusion_matrix(y1_encoded_int, y_pred_test_int, labels=range(len(decoded_labels)))

# Printing the confusion matrix for training data with labels
print("Confusion Matrix - Test Data:")
print(pd.DataFrame(conf_matrix_test, index=decoded_labels, columns=decoded_labels))


# Visualize the confusion matrix with decoded labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
plt.xticks(ticks=np.arange(len(decoded_labels)) + 0.5, labels=decoded_labels, rotation=45)
plt.yticks(ticks=np.arange(len(decoded_labels)) + 0.5, labels=decoded_labels, rotation=0)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix with Decoded Labels on dummy test file")
plt.show()