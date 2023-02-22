# Detecting Fraudulent Credit-Card Transactions

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Credit-Card-Fraud-Detection.jpg)
## Brief Introduction -
Credit card fraud is becoming very common nowadays. It is a fraud committed using a credit card to obtain goods or services or to make payment to another account, which is controlled by a criminal. 

Detecting these types of fraud is a challenging task as it requires detecting fraudulent transactions out of millions of daily transactions. Due to the enormous amount of data, it is now nearly impossible for human specialists to identify meaningful patterns from transaction data. However, employing machine learning models these types of fraudulent transactions can be detected with some success. 

## Aims and Objective -
To build  Machine Learning models that can to detect the fraudulent transactions in the (anonymized) real-world dataset obtained from european credit card users.

## Dataset - 
    (1) The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
    (2) The data available for this project was collected over two days.
    (3) Data contained 492 fraudulent transactions out of 284,807. 
    (4) The dataset is severly imbalanced posing additional challenges for the ML models. 
    (5) Necessary ML steps will be taken to improve the model performance in such an imbalanced dataset. 
    (6) Imbalanced dataset in real life scenarios where the anomaly needs to be detected are very common.
    (7) These senarios include (but not limited to)
            •	Claim Prediction
            •	Default Prediction.
            •	Churn Prediction.
            •	Spam Detection.
            •	Anomaly Detection.
            •	Outlier Detection.
            •	Intrusion Detection
            •	Conversion Prediction.

![LR](https://github.com/fbrowther/Anomaly-Detection-to-identify-Fraudulent-Credit-Card-Transactions/blob/main/Images/Class%20Distribution%20in%20original%20Dataset.png)
Class 0 are non-fradulent transactions and Class 1 are fradulent

## Aims and Objective -
To build Unsupervised Machine Learning that can categorize Myopic children (value =1) from those who do not have Myopia (value=0) in the age group of 5-9 years old.

## ML Steps executed -

(1) Data Preparation and Scaling -

(2) Building ML models (in original dataset)

            (A) Supervised ML Models -
            (B) Unsupervised ML Models -
                (1) Dimensionality Reduction using PCA and t-SNE,
                (2) Cluster Analysis using KMeans Clustering,
            (C) Artificial Neural Network -
                
(3) Balancing the dataset 
            (A) random over-sampling (RandomUnderSampler), 
            (B) random under-sampling (RandomUnderSampler) &
            (C) random over & under-sampling (SMOTETomek)

(4) Building ML models (in balanced dataset) 

(5) Making recommendations to end-users (credit card companies) for deployement 
             

      
## Specific Libraries and modules employed -
      
      Scikit-Learn ML and Library-
      
            (a) Preprocessing - StandardScaler, normalize
  
            (b) Decomposition - PCA, 
  
            (c) Manifold - TSNE
  
            (d) Cluster - KMeans
  
      Scipy.cluster.hierarchy-
      
            (a) Dendrogram, Linkage
      
    

