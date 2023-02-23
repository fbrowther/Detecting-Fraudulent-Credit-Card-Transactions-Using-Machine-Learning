# Detecting Fraudulent Credit Card Transactions

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Credit-Card-Fraud-Detection.jpg)
## Brief Introduction -
Credit card fraud is becoming very common nowadays. It is a fraud committed using a credit card to obtain goods or services or to make payment to another account, which is controlled by a criminal. 

Detecting these types of fraud is a challenging task as it requires detecting fraudulent transactions out of millions of daily transactions. Due to the enormous amount of data, it is nearly impossible for human specialists to identify meaningful patterns from transaction data. However, employing machine learning models these types of fraudulent transactions can be detected with some success. 

## Aims and Objective -
To build  Machine Learning models that can detect fraudulent transactions in the real-world dataset, obtained from european credit card users.

## Dataset - 
    (1) The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
    (2) The data available for this project was anonymized and collected over two days.
    (3) Data contained 492 fraudulent transactions out of 284,807. 
    (4) The dataset is severly imbalanced posing additional challenges for the ML models. 
    (5) Necessary ML steps will be taken to improve the model performance in such an imbalanced dataset. 
    (6) Imbalanced dataset in real life scenarios where the anomaly needs to be detected are very common.
    (7) These scenarios include (but not limited to)
            •	Claim Prediction
            •	Default Prediction.
            •	Churn Prediction.
            •	Spam Detection.
            •	Anomaly Detection.
            •	Outlier Detection.
            •	Intrusion Detection
            •	Conversion Prediction.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Class%20Distribution%20in%20original%20Dataset.png)

Class 0 are non-fradulent while Class 1 are fradulent transactions

## ML Steps executed -

(1) Data Preparation, Cleaning and Scaling -

(2) Building ML models (in original dataset)

            (A) Supervised ML Models 
                (1) Logistic Regression
                (2) K Nearest Neighbors
                (3) SVM (with and without class weight assignment)
                (4) Random Forest Classifier
                (5) Extra Tree Classifier
                (6) Adaboost Classifier
                (7) SelectFromModel and Feature_Importance
            (B) Unsupervised ML Models
                (1) Dimensionality Reduction using PCA and t-SNE
                (2) Cluster Analysis using KMeans Clustering
            (C) Artificial Neural Network - ANN
                
(3) Balancing the dataset -
            (A) Random over-sampling (RandomUnderSampler), 
            (B) Random under-sampling (RandomUnderSampler) &
            (C) Random over & under-sampling (SMOTETomek)

(4) Building ML models in balanced dataset
            (1) Random Forest Classifier

(5) Making recommendations to end-users (credit card companies) for deployement 
             
      
## Specific Libraries and modules employed -
      
      (1) Scikit-Learn ML and Library-
            (a) Preprocessing - StandardScaler, Normalize, OneHotEncoder
            (b) model_selection - train_test_split, SelectFromModel
            (c) Decomposition - PCA  
            (d) Manifold - TSNE
            (e) Cluster - KMean, AgglomerativeClustering
            (f) Linear_model - LogisticRegression
            (g) Neighbors - KNeighborsClassifier
            (h) SVM - SVC
            (i) Metrics - Classification_report, roc_curve, auc
            (j) Ensemble - RandomForestClassifier  
       (2) Imblearn -  
            (a) over_sampling - RandomOverSampler
            (b) under_sampling - RandomUnderSampler
            (c) combine - SMOTETomek
       (3) TensorFlow - Neural Network
       (4) Keras-tuner - To automate the Neural Network to choose the best model and the best hyperparameters
       (5) Pandas
       (6) Matplotlib
       (7) Numpy
       (8) Seaborn
       (9) TensorFlow - Neural Network
       (10) Keras-tuner - To automate the Neural Network to choose the best model and the best hyperparameters
   
## Model Performance -    

## Supervised ML Models -

Supervised ML Models rely on the data labels to train algorithms that will then classify data or predict the outcomes accurately. 

### (1) Logistic Regression
This model usually employed in making categorical predictions, like yes/no, true/false or class-0/class-1. In this case, this algorithm was used to predict whether the transaction was fradulent or not. This model gave a precision of 82% and a recall score of 52%. 

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Log.png)

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/KNN.png)

![LR]()

![LR]()





The first supervised model we will talk about is logistic regression. We chose this model because this is what is usually used in making categorical predictions, like yes/no, binary or true/false. In our case we are traying to predict which transactions are non-fraudulent (0) or fraudulent (1). When we run the model, we can see that it gives a precision of 82%, a recall of 52% and an f1-score of 64%, which is not the greatest result.

Therefore, we used another prediction classifier, k nearest neighbors. As you can see from the graph, when we separate the data into test (yellow) and training(blue) accuracy, it stablilizes at around 3 neighbors. We can also observe a slight improvement from logistic regression,in the accuracy scores, with a precision of 88%, recall of 74% and f1-score of 81%.

We then tried to do an ROC probability corve of plottin TP rate vs FP rate. We can observe that the highest TP rate is approx. 0.85 and it starts at approx. 0.84, which is not a bad yield.

Because the dataset is imbalanced, we used a prediction classifier that is commonly used in imbalanced datasets, Support Vector Machine (SVM). We used both with and without classweight to measure as accurate as possible. Although the result difference was negligible, the model performed slightly better without class weight, with a precision of 81%, recall of 78% and f1-score of 80%.  


### (2) K Nearest Neighbors -

### (3) SVM (with and without class weight assignment)
### (4) Random Forest Classifier
### (5) Extra Tree Classifier
### (6) Adaboost Classifier
### (7) SelectFromModel and Feature_Importance

      
      
      
      
      
      
The four possible outcomes are

TP: True positives. These are the instances of class 1 (fraudulent transactions), that the classifier correctly predicts as fraudulent.
TN: True negatives. These are the instances of class 0 (genuine transactions), that the classifier correctly predicts as genuine.
FP: False positives. These are the instances of class 0 (genuine transactions), that the classifier incorrectly predicts as fraudulent.
FN: False negatives. These are the instances of class 1 (fraudulent transactions), that the classifier incorrectly predicts as genuine.
      
    

