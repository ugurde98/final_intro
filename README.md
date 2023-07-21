# DNA Microbe Classifier

## Introduction

This project aims to implement a classifier algorithm using DNA samples of microbes in human blood to predict four different cancer types: Colon cancer, breast cancer, lung cancer, and prostate cancer. The dataset contains blood sample data from 355 individuals, and each sample has 1836 different microorganism features represented as counts of DNA fragments.

The goal of the project is to achieve the highest possible correct classification scores for the cancer types using artificial neural networks.

## Data

The dataset is provided in two files:
- `data.csv`: Contains the DNA fragment counts for each microorganism for each sample.
- `labels.csv`: Contains the disease type (cancer type) for each sample.

Each data component is given as counts, and for better performance, data normalization was performed by dividing the count of each microorganism by the summation of all 1836 counts for each sample.

## Classification Algorithm

For this project, a multi-layer perceptron (MLP) classifier was implemented using the PyTorch framework. The model architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with a softmax activation for multi-class classification.

## Performance Measures

The model's performance was evaluated using the following metrics:
- Precision: The ratio of correctly predicted positive cases to the total number of samples predicted as positive.
- Recall: The ratio of correctly predicted positive cases to the total number of actual positive cases.
- F2 score: A combined measure of Precision and Recall, giving more weight to minimizing false negative predictions.

## How to Use

1. Download the dataset from the following link: [Dataset](https://drive.google.com/file/d/15evTOZTYuopoBnolYWOPy2P_VF6wnlFm/view?usp=sharing)

2. Run the `dna_microbe_classifier.py` script to train and test the neural network model.

3. The script will output the Precision, Recall, and F2 score for the model's performance.

## Requirements

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- numpy

## Conclusion

The implemented neural network classifier demonstrated promising results in predicting cancer types based on DNA samples of microbes. Further improvements and optimizations could be explored, such as experimenting with different architectures, hyperparameters, and data preprocessing techniques. This model can potentially serve as a valuable tool for cancer diagnosis and research.

# Final Report

![Result](/final_intro/Sonuc.jpeg)

In this project, an artificial neural network classifier algorithm was implemented to predict four different cancer types using the DNA samples of microbes in human blood. The dataset used for this project consists of blood sample data from 355 individuals, with four most common cancer types: Colon cancer, breast cancer, lung cancer, and prostate cancer.

The dataset contains 1836 different microorganism features, and each numerical component provided for each sample represents the count of DNA fragments belonging to each microorganism type. Data normalization was performed by dividing the count of each microorganism by the summation of all 1836 counts for each sample.

Performance evaluation of the model was carried out using Precision, Recall, and F2 score as performance metrics. Precision indicates the ratio of correctly predicted positive cases to the total number of samples predicted as positive, while Recall represents the rate of correctly predicted positive cases to the total number of actual positive cases. F2 score is a combined measure of Precision and Recall, giving more weight to minimizing false negative predictions.

Based on the results, the artificial neural network model achieved an F2 score of {{0.611}}, a Precision of {{0.59}}, and a Recall of {{0.6}}. These results indicate that the model is successful in predicting the four different cancer types.

For the implementation of the project, a multi-layer perceptron (MLP) model was created using the PyTorch library. The model utilized a convolutional layer with 1x1 filters to process the data. Accuracy was also used as an evaluation metric during model training and performance assessment.

In conclusion, a successful classification model was built using artificial neural networks to predict cancer types based on microorganism DNA samples. The obtained results suggest the potential use of this model as an assisting tool in cancer diagnosis. However, further refinement and experimentation with different data normalization techniques could enhance the model's performance.
