# Class-Imbalance-Classification-Performance-Analysis
This repository contains the code, documentation, and datasets for a comprehensive exploration of machine learning techniques to address class imbalance. The project investigates the impact of various methods, including ADASYN, KMeansSMOTE, and Deep Learning Generator, on classification performance using three real-world datasets. The code is implemented in Python, utilizing Jupyter Notebooks for experimentation and analysis. Additionally, detailed documentation and insights are provided in a Word file, along with the necessary datasets for reproduction of the results.

### Class Imbalance Solution Techniques Selected:
#### Solution # 1: ADASYN
What is ADASYN?

ADASYN (Adaptive Synthetic Sampling) is an oversampling technique used to address class imbalance in datasets. It generates synthetic samples for the minority class, focusing on regions where the minority class is underrepresented. ADASYN identifies minority class instances in the dataset. It then computes the density of each minority class instance. Density is the imbalance ratio, indicating the representation of minority class instances among its k nearest neighbors. From the nearest neighbors, it randomly selects one neighbor to create synthetic sample. For the minority instance and randomly selected neighbor, it calculates a weighted average of the two for each feature and this gives a new synthetic data point and it then assigns it minority label and continues to do so until class imbalance is treated. ADASYN prioritizes regions with low imbalance ratios, meaning it creates more synthetic samples in the regions with less minority labelled neighbors.

Why ADASYN?

In our datasets at max we got class imbalance of 84:16 so we could've gone with both ADASYN & SMOTE in this case. However, ADASYN is still the preferred choice. But because when visualized (through PCA) we found that for each dataset we're unable to identify distinct simple linear decision boundaries, ADASYN turns out to be a better choice than SMOTE.
Moreover, since our datasets are relatively small in size there's no practical difference between SMOTE & ADASYN in terms of computation and time cost (otherwise for large datasets SMOTE is better and major reason why one should pick SMOTE over ADASYN otherwise ADASYN is a better choice overall)

#### Solution # 2: KMeansSMOTE
What is KMeansSMOTE?

KMeansSMOTE is an extension over SMOTE oversampling technique where first it identifies clusters within the data using K-Means Clustering and then generates synthetic samples for minority instances within a cluster using SMOTE. Intiially, k cluster centers (centroids) are selected and based on distance, instances closer to centroids are assigned to their respective clusters. Centroids are recalculated (mean is taken) and all minority instances are reassigned a cluster again based on distance and this continues until centroids don't change much and that's how we get our clusters. Then similar to ADASYN it generate synthetic samples using the neighbors of each minority instance but this time neighbors are restricted to clusters and unlike ADASYN number of synthetic samples to generate for each minority instance is not based on instances' density.

Why KMeansSMOTE?

When visualized (through PCA) we found that for each dataset we've complex non-linear decision boundaries (low correlation as well) so clustering-based oversampling methods like KMeansSMOTE is particularly beneficial as it captures the local structure of the data more effectively but only generating within clusters, leading to improved model generalization and performance. Sampling generation through this method is highly interpretable and also computationally efficient and scalable to large datasets. This scalability makes it suitable for datasets with a large number of rows and columns, such as ours with more than 36000 rows and about 19 columns.

#### Solution # 3: Deep Learning Generator (Mostly AI)
What is Deep Learning Generator?

It is an online generator which uses a number of Deep Learning model architectures and approaches that have emerged to create synthetic data, including Transformers, GANs, Variational Autoencoders as well as Autoregressive Networks. It has and continue to actively research all of them, and have developed it's own unique combination of techniques to provide the best possible results in terms of accuracy, privacy as well as flexibility.

##### GANs
GANs consists of two neural networks. There is a generator, G(x), and a discriminator, D(x). Both of them engage in a competing game. The generator's goal is to trick the discriminator by providing data that is similar to that of the training set. The discriminator will attempt to distinguish between fake and real data. Both work together to learn and train complicated data such as audio, video, and image files.
The Generator network takes a sample and creates a simulated sample of data. The Generator is trained to improve the Discriminator network's likelihood of making errors.

##### Transformers
Transformers are neural network architectures that convert or modify input sequences into output sequences. They accomplish this by acquiring context and tracking the relationships between sequence components. Consider the following input sequence: "What is the color of the sky?" The transformer model employs an internal mathematical representation to determine the relevance and relationship among the words color, sky, and blue. It uses that knowledge to produce the following output: "The sky is blue." 

##### Variational Autoencoders
A variational autoencoder (VAE) uses probability to describe a latent space observation. Thus, instead of creating an encoder that outputs a single value to describe each latent state characteristic, It will create an encoder that describes a probability distribution for each latent property. It is used in a variety of applications, including data compression and synthetic data synthesis.

Variational autoencoders differ from autoencoders in that they give a statistical method for representing the dataset's samples in latent space. As a result, the variational autoencoder generates a probability distribution in the bottleneck layer rather than a single output value.

##### Autoregressive models
Autoregressive models are a class of machine learning (ML) models that automatically predict the next component in a sequence by taking measurements from previous inputs in the sequence.

Why Deep Learning Generator?

The AI tool leverages a combination of cutting-edge techniques, including Generative Adversarial Networks (GANs), Transformers, Variational Autoencoders, and Autoregressive models. These state-of-the-art methods enable the generation of synthetic data that closely resembles the distribution of the original datasets. By employing these advanced techniques, we can create high-quality synthetic samples that effectively represent the minority class, thereby addressing the imbalance issue.


### Following Five Classification Algorithms Are Selected For This Project:
• KNN, Logistic Regression, Gaussian Naive Bayes, Linear SVM, & Decision Trees

### Following Evaluation Metrics Are Selected For This Project:
• Precision (Both Classes), Recall (Both Classes), F1-Score (Both Classes), Accuracy, & AUC


## Dataset 1
Title: Lending Club Loan Data Analysis

Domain: Finance

Number of Rows: 9578

Number of Columns: 14

Feature Type: Mixed

Class Balance: 84:16 (Percentage)

Source: https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis/data


## Dataset 2
Title: Hotel Reservation Classification Dataset

Domain: Hospitality

Number of Rows: 36275

Number of Columns: 19

Feature Type: Mixed

Class Balance: 67:33 (Percentage)

Source: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset


## Dataset 3
Title: Churn Modelling

Domain: Finance

Number of Rows: 10000

Number of Columns: 14

Feature Type: Mixed

Class Balance: 79:21 (Percentage)

Source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
