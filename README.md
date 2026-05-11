# Fraud Detection Portfolio Projects

I am currently applying to a few fraud detection data science roles, so I decided to put together some sample projects.

I worked on some fraud detection ML validation projects for Synchrony Bank early in my professional career, and I was aware of the prevalence of XGBoost for transaction-level fraud scoring. XGBoost is popular in finance, and I've also seen it being used for credit scoring (i.e. VantageScore) and for anti-money laundering transaction monitoring applications. 

Building a good fraud detection ML model is challenging for multiple reasons, including the fact that successful cases of fraud are generally not observable in the data, and the sparsity of true-label examples presents a class imabalance problem when training an ML model. Selecting the right evaluation metrics for hyperparameter tuning. 

Another area that I wanted to challenge myself with in this project was Graph Neural Networks (GNN). I took a course offered by Stanford Online (XCS224W: Machine Learning with Graphs) taught by Prof. Jure Leskovec two years ago, but never had the chance to properly digest the material. A few months ago, I attended a webinar hosted by Kumo.ai and Prof. Leskovec on applications of GNNs for identifying fraud rings, and my interest in the subject matter was piqued again -- so this project seemed like a great opportunity for me to recap all of those materials from his course once again. 

# Description of Files

* `fraud_detection.ipynb`
    * **Info**: Demo of an XGBoost classifier method to flag fraudulent transactions
    * **Files**: 
* `gnn.ipynb`
    * **Info**: Demo of a Graph Neural Network based fraud detction model, to identify fraud rings 
