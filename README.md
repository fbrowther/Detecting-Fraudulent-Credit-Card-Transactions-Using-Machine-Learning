# Detecting Fraudulent Credit Card Transactions

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Credit-Card-Fraud-Detection.jpg)
## Brief Introduction -
Credit card fraud is becoming very common nowadays. Detecting these types of fraud is a challenging task as it requires detecting fraudulent transactions out of millions of daily transactions. Due to the enormous amount of data, it is nearly impossible for human specialists to identify meaningful patterns from transaction data. However, employing machine learning models these types of fraudulent transactions can be detected. 

## Aims -
To build machine learning models that can detect fraudulent transactions in the real-world dataset (anonymized dataset obtained from european credit card users)

## Dataset - 
    (1) The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
    (2) The data was collected over two days.
    (3) It contained 492 fraudulent ones out of 284,807 transactions i.e., 1:577. 
    (4) The dataset is severly imbalanced posing additional challenges for the ML models. 
    (5) Necessary steps will be taken to improve the model performance. 
    (6) Imbalanced dataset in real life scenarios are very common.
        These scenarios include (but not limited to) -
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

## ML Steps to be executed -

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

(4) Building ML models in the newly balanced dataset using Random Forest Classifier

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
   
# Model Performance -    

# Supervised ML Models

Supervised ML Models rely on the data labels to train algorithms that will then classify data or predict the outcome accurately. 

### (1) Logistic Regression
This model is (usually) employed in making categorical predictions, like yes/no, true/false or class-0/class-1. In this current project, this algorithm was used to predict whether the transaction was fradulent or not. This model gave a precision of 82% and a recall score of 52%. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Log1.png" width="750" height="250">

### (2) K Nearest Neighbors (KNN)
KNN uses proximity to make classifications or predictions about a group of data points. We observed a slight improvement in precision (88%) and recall scores (74%) compared to the logistic regression. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/KNN.png" width="750" height="250">

### (3) SVM (with and without class weight assignment)
Support Vector Machine (SVM) is a prediction classifier that is commonly used in imbalanced datasets. We assigned either an equal or double the weight to class 1 (compared to class 0) before training the models. This model showed more balanced score for both precision and recall.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/SVM.png" width="350" height="375">

### (4) Random Forest Classifier
A random forest classifier is an estimator algorithm that fits a number of decision tree on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. This then outputs the most optimal result. 

So far, this was the best performing algorithm that gave us the precision and recall scores of 0.93 and 0.80 for fraudulent transactions. So, we used this as a baseline for the subsequent models.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/RFC.png" width="700" height="250">

### (5) Extra Tree Classifier
ExtraTreesClassifier implements an estimator algorithm, that fits a number of randomized decision trees (extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and controls the over-fitting. This is similar to Random Forest classifier. 

This model gave us precision and recall values of 0.91 and 0.79 for fraudulent transactions.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Extra%20Tree%20Classifier.png" width="350" height="150">


### (6) Adaboost Classifier
An AdaBoost classifier is an estimator algorithm that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. This algorithm can be used to boost the performance of any machine learning algorithm. It is best used with weak learners. 

For this dataset, we got precision and recall values of 0.78 and 0.70 for fraudulent transactions using this model.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Adaboost%20Classifier.png" width="350" height="150">

### (7) SelectFromModel and Feature_Importance
Feature Selection is the method of reducing the input variable to the model by using only relevant data and getting rid of noise in data. It is the process of automatically choosing relevant features for your machine learning model based on the type of problem one is trying to solve.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/FeatureSelection.png" width="350" height="250">

We employed SelectFromModel to train the dataset; however, this model was developed using only the selected features as obtained from above. 

This gave us precision and recall values 0.90 and 0.74 for fraudulent transactions.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/SelectFromModel-FeatureSelection.png" width="350" height="150">

### Summary of Supervised ML models
Out of all the supervised ML algorithms we employed, Random Forest Classifier was the best model so far.


# Unsupervised ML Models
Unsupervised learning method allows us to cluster data to find hidden or unknown patterns. 

Since the incoming new transaction data represents data with no prior label, we wanted to employ these models to find clusters based on their similarity and differences to identify whether they were either fradulant or non-fraudulent.

### (1) PCA
PCA is a statistical technique used to speed up machine learning algorithms and works by reducing the number of input features or dimensions. As you can see from the diagram the PCA reduced the number of features from 29 to 13. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/PCA.png" width="350" height="250">

### (2) TSNE
The TSNE algorithm also known as Stochastic Neighbourhood Embedding is a clustering and visualization method. It is different from PCA and is able to account for non-linear relationships. This algorithm models the probability distribution of neighbours around each point. 

In the transaction dataset, TSNE algorithm was unable to distinguish the classes.

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/TSNE.png" width="350" height="250">

### (3) K-means Clustering (in conjunction with Elbow Curve)
The K-means algorithm groups the data into clusters where each piece of data is assigned to a cluster based on similiarity or the distance measured to a centroid. A centroid represents a data point that is the arithmetic mean position of all the points in a cluster. This process is repeated until the data is separated into distinct groups.

The elbow curve represents the number for k / clusters as it is the inflection point where the slope takes a sharp turn and flattens out. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Elbow%20Curve.png" width="350" height="250">

Employing elbow curve we determined the number of cluster that is recognizable in the current dataset to be 4. However, on retraining the dataset for Kmeans clustering using K=4, we obtained the inertia value of 11267 and as we read it off the y-axis it can be seen to have 2 clusters as expected. 

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Inertia.jpg" width="150" height="75">

### Summary of Unsupervised ML models
Unsupervised ML models were unable to identify two distinct clusters representing fradulent and non-fradulent transactions and therefore was not included for comparison with other models.

# Deep Learning

Neural networks are a subset of machine learning algorithms which mimic the working of a human brain with regards to how the information is processed and understood.  

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/NN.png" width="800" height="300">

Neural networks rely on training data to learn and improve their accuracy. However, once these learning algorithms are fine-tuned for accuracy, 
they are powerful tools in machine learning to classify and cluster data at a very high velocity. 

In this use case, this model would identify whether a transaction is fraudulent or not.

Since the dataset we are dealing with is severely imbalanced, it is very important to note that determining loss or accuracy alone will not give an indication of how well the neural networks are working in predicting the minority class.

Therefore, these metrics were fed into the network for computing at every epoch during the training process. 

The testing scores for these metrics are presented as follows-

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Metrics-NN.jpg" width="500" height="272">


## Performance of Artificial neural networks (ANN)

<img src="https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/ANN%20comparison.png" width="725" height="200">

The four possible outcomes of the Neural Network models are -
       TP: These are the instances of class 1 (fraudulent transactions), that the ANN correctly predicts as fraudulent.
       TN: These are the instances of class 0 (genuine transactions), that the ANN correctly predicts as genuine.
       FP: These are the instances of class 0 (genuine transactions), that the ANN incorrectly predicts as fraudulent.
       FN: These are the instances of class 1 (fraudulent transactions), that the ANN incorrectly predicts as genuine.
      
Based on the best scores for FN and FP, the highlighted models with binary accuracy of 0.6 was chosen for deployment. In the current use case for detecting fradulent transactions, FN values plays a crucial role.

## Balanced Dataset & RFC Model

Balancing of the current dataset was attempted employing RandomOverSampler, RandomUnderSampler, and SMOTETomek approaches. These methods yielded near to perfect scores for all the metrics analysed for Random Forest Classifier. 

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Balanced%20Dataset.png)

## Final Comparison of the Models

When all the relevant models were compared we obtained two best models in the original imbalanced dataset to proceed for deployment (highlighted in yellow).

However, if the dataset were to be balanced further, these models became perfect enough to predict the classes (fraudulent or not) with 100% accuracy (highlighed in red)

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Images%20for%20Readme/Final%20Table.png)


## Conclusion and Limitations -
         (1) Best_Models - 
             ML models with ‘best possible’ Precision and Recall scores in both imbalanced and balanced dataset were developed
         (2) Imbalanced Dataset - 
             Larger list of metrics had to be computed to assess the performance of the models in the minority class
         (3) Computationally demanding - 
             A number of metrics had to be calculated for this large dataset and therefore modeling time was substantially longer
         (4) Cloud Technologies - To cope with the increased computational demands of this project, cloud Technologies (AWS, Azure, Snowflake etc.,) 
             could be employed in future
