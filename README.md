# NBA-Position-Prediction using Neural Networks

![image](https://github.com/Satya-bit/NBA-Position-Prediction-using-Neural-Networks/assets/70309925/35762c0a-8f12-4031-b377-ca8123a22280)

The following code depicts the position of NBA players which is divided into 5 categories i.e. Point Guard, Shooting Guard, Small Forward, Power Forward, Center.

![image](https://github.com/Satya-bit/NBA-Position-Prediction-using-Neural-Networks/assets/70309925/aedbe55a-4e54-45bc-a7d9-67c3d32cbea7)

# MY APPROACH

![image](https://github.com/Satya-bit/NBA-Position-Prediction-using-Neural-Networks/assets/70309925/14b70059-aaab-41c9-935c-33603cc3c1e4)

->I have used pearson correlation coeffecient. The threshold value is 0.85 i,e, I have removed the features highly correlated features whose threshold coeeficient is more than 0.85. I have also removed some redundant features.

->I have also done some feature engineering for this by creating some new features.

->For outliers I have done winsorization.

->Also performed standardisation

->I have used Multi Layer Perceptron Classifier. I have tuned the hyperparameters and also used GridSearchCV for getting optimal hyperparameters

->K FOLD startified cross validation with value of K=10.

->Also made confusion matrix in the end.

# RESULTS

->I got 62.65% training accuracy and 59% on Test data(Random state=0)

->Average of K FOLD cross validation accuracy is 59%(Random state=0)

->Accuracy on dummy_test is 69.3%

->My model is not overfitting as all accuracies are almost close to eachother. Nor it is underfitted as the base class is 20%(5 positions to predict)
