# Class-Imbalance-Classification-Performance-Analysis
This repository contains the code, project report, and datasets for a comprehensive exploration of machine learning techniques to address class imbalance. The project investigates the impact of various methods, including ADASYN, KMeansSMOTE, and Deep Learning Generator, on classification performance using three real-world datasets. Implemented in Python and utilizing Jupyter Notebooks for experimentation and analysis, this repository offers detailed documentation and insights, along with the necessary datasets for reproduction of the results.

## Class Imbalance Solution Techniques Selected:
• ADASYN (Adaptive Synthetic Sampling): Generates synthetic samples for the minority class, focusing on regions with low imbalance ratios.
• KMeansSMOTE: Combines K-Means clustering with SMOTE to generate synthetic samples within clusters.
• Deep Learning Generator: Utilizes advanced deep learning models (GANs, Transformers, Variational Autoencoders, Autoregressive models) from MostlyAI to create high-quality synthetic data.

## Following Five Classification Algorithms Are Selected For This Project:
• KNN, Logistic Regression, Gaussian Naive Bayes, Linear SVM, & Decision Trees

## Following Evaluation Metrics Are Selected For This Project:
• Precision (Both Classes), Recall (Both Classes), F1-Score (Both Classes), Accuracy, & AUC

## Following Three Datasets Are Selected For This Project:
### Dataset 1
Title: Lending Club Loan Data Analysis

Domain: Finance

Number of Rows: 9578

Number of Columns: 14

Feature Type: Mixed

Class Balance: 84:16 (Percentage)

Source: https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis/data


### Dataset 2
Title: Hotel Reservation Classification Dataset

Domain: Hospitality

Number of Rows: 36275

Number of Columns: 19

Feature Type: Mixed

Class Balance: 67:33 (Percentage)

Source: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset


### Dataset 3
Title: Churn Modelling

Domain: Finance

Number of Rows: 10000

Number of Columns: 14

Feature Type: Mixed

Class Balance: 79:21 (Percentage)

Source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

## Project Objectives:
• Explore and compare the performance of ADASYN, KMeansSMOTE, and a Deep Learning Generator in addressing class imbalance.
• Evaluate the impact of these techniques on selected classification algorithms.
• Provide comprehensive insights and documentation to facilitate the reproduction and understanding of results.
