# Detecting Fraudulent Credit Card Transactions

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Credit-Card-Fraud-Detection.jpg)
## Brief Introduction -
Credit card fraud is becoming very common nowadays. It is a fraud committed using a credit card to obtain goods or services or to make payment to another account, which is controlled by a criminal. 

Detecting these types of fraud is a challenging task as it requires detecting fraudulent transactions out of millions of daily transactions. Due to the enormous amount of data, it is nearly impossible for human specialists to identify meaningful patterns from transaction data. However, employing machine learning models these types of fraudulent transactions can be detected with some success. 

## Aims and Objective -
To build  Machine Learning models that can detect fraudulent transactions in the (anonymized) real-world dataset, obtained from european credit card users.

## Dataset - 
    (1) The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
    (2) The data available for this project was collected over two days.
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

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Class%20Distribution%20in%20original%20Dataset.png" width="400" height="400">
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

## Supervised ML Models

Supervised ML Models rely on the data labels to train algorithms that will then classify data or predict the outcomes accurately. 

### (1) Logistic Regression
This model is usually employed in making categorical predictions, like yes/no, true/false or class-0/class-1. In current project, this algorithm was used to predict whether the transaction was fradulent or not. This model gave a precision of 82% and a recall score of 52%. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Log1.png" width="1000" height="350">

### (2) K Nearest Neighbors (KNN)
KNN uses proximity to make classifications or predictions about a group of data points. We observed a slight improvement in precision (88%) and recall scores (74%) with this model. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/KNN.png" width="1000" height="350">

### (3) SVM (with and without class weight assignment)
Support Vector Machine (SVM) is a prediction classifier that is commonly used in imbalanced datasets. We either assigned an equal or double the weight to class 1 compared to class 0 before training the models. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/SVM.png" width="450" height="500">

### (4) Random Forest Classifier
A random forest classifier is an estimator algorithm that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting and then outputs the most optimal result. 

This was the best performing algorithm which gave us the best precision and recall values of 0.93 and 0.80 for fraudulent transactions. So, we used this as a baseline for all the other subsequential models.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/RFC.png)

### (5) Extra Tree Classifier
ExtraTreesClassifier implements an estimator algorithm, that fits a number of randomized decision trees (extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. This is similar to Random Forest classifier. 

This model gave us precision and recall values of 0.91 and 0.79 for fraudulent transactions.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Extra%20Tree%20Classifier.png)

### (6) Adaboost Classifier
An AdaBoost classifier is an estimator algorithm that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. This algorithm can be used to boost the performance of any machine learning algorithm. It is best used with weak learners. 

For this dataset, we got precision and recall values of 0.78 and 0.70 for fraudulent transactions.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Adaboost%20Classifier.png)

### (7) SelectFromModel and Feature_Importance
Feature Selection is the method of reducing the input variable to your model by using only relevant data and getting rid of noise in data. It is the process of automatically choosing relevant features for your machine learning model based on the type of problem you are trying to solve.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/FeatureSelection.png)

We employed SelectFromModel to train the dataset however this model was developed using only the selected features as obtained from above. 

This gave us precision and recall values 0.90 and 0.74 for fraudulent transactions.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/SelectFromModel-FeatureSelection.png)

### Summary
Out of all the supervised machine learning algorithms we employed, Random Forest Classifier was the best model so far.


## Unsupervised ML Models
Unsupervised learning method allows us to cluster data in order to find hidden or unknown patterns. 

Since the incoming new transaction data represents data with no prior label, we wanted to employ these models to find clusters based on their similarity and differences to identify whether they were either fradulant or non-fraudulent.

### (1) PCA
PCA is a statistical technique used to speed up machine learning algorithms and works by reducing the number of input features or dimensions. As you can see from the diagram the PCA reduced the number of features from 29 to 13. 

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/PCA.png)

### (2) TSNE
The TSNE algorithm also known as Stochastic Neighbourhood Embedding is a clustering and visualization method is different from PCA and is able to account for non-linear relationships. This algorithm models the probability distribution of neighbours around each point. 

In the transaction dataset, TSNE algorithm was unable to distinguish the classes.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/TSNE.png)

### (3) K-means Clustering (in conjunction with Elbow Curve)
K represents the number of clusters. The K-means algorithm groups the data into clusters where each piece of data is assigned to a cluster based on similiarity or the distance measured to a centroid. A centroid represents a data point that is the arithmetic mean position of all the points in a cluster. This process is repeated until the data is separated into distinct groups.

The elbow curve represents the number for k and the number of clusters as it is the inflection point where the slope takes a sharp turn and flattens out. 

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Elbow%20Curve.png)

Employing elbow curve we determined the number of cluster that is recognizable in this dataset to be 4. However, on retraining the dataset for Kmeans clustering using K=4, we obtained the inertia value of 11267 and as we read it off the y-axis it can be seen to have 2 clusters as expected. 

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Inertia.jpg)

### Summary
Unsupervised ML models were unable to identify two distinct clusters of fradulent and non-fradulent transactions within our dataset.

## Deep Learning

Neural networks are a subset of machine learning algorithms which mimic the working of a human brain with regards how the information is processed and understood.  

![LR]()

Neural networks rely on training data to learn and improve their accuracy. However, once these learning algorithms are fine-tuned for accuracy, 
they are powerful tools in machine learning to classify and cluster data at a very high velocity. In this case to determine whether a transaction was fraudulent or not.

Since the dataset we are dealing with is severely imbalanced, it is very important to note that determining loss or accuracy alone will not give an indication of how well the neural networks are working in predicting the minority class.

These metrics were fed into the network to be computed during the training process.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Metrics-NN.jpg" width="1000" height="350">


## Artificial neural networks (ANN) performance

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/ANN%20comparison.png)


![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Final%20Table.png)
 








      
      
      
      
      
      
The four possible outcomes are

TP: True positives. These are the instances of class 1 (fraudulent transactions), that the classifier correctly predicts as fraudulent.
TN: True negatives. These are the instances of class 0 (genuine transactions), that the classifier correctly predicts as genuine.
FP: False positives. These are the instances of class 0 (genuine transactions), that the classifier incorrectly predicts as fraudulent.
FN: False negatives. These are the instances of class 1 (fraudulent transactions), that the classifier incorrectly predicts as genuine.
      
    

