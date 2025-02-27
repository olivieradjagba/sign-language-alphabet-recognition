# Sign Language Classification using Deep Learning

## Introduction

Sign language is a crucial mode of communication for the deaf and hard-of-hearing community. It enables individuals to express themselves through hand gestures, facial expressions, and body movements. However, there is still a communication gap between sign language users and those unfamiliar with it. Automated sign language recognition using deep learning can help bridge this gap by enabling real-time translation into text or speech.

This project aims to classify sign language alphabets and digits using three different deep learning models: a Convolutional Neural Network (CNN), a Transfer Learning model based on ResNet, and a Vision Transformer (ViT) model built from scratch. By comparing their performance, we aim to identify the most effective approach for sign language classification and also explore the power of these deep learning architecture.

## Models Overview

### 1. CNN Model

A Convolutional Neural Network (CNN) was developed from scratch to extract spatial features from sign language images. CNNs are particularly effective for image classification tasks due to their hierarchical feature extraction capabilities.

### 2. Transfer Learning with ResNet

This model utilizes a pre-trained ResNet architecture, leveraging its feature extraction capabilities to improve classification performance. Transfer learning allows the model to benefit from features learned on large-scale datasets, making training more efficient and effective.

### 3. Vision Transformer (ViT)

A Vision Transformer (ViT) model was implemented from scratch to explore the potential of transformer-based architectures in image classification. Unlike CNNs, ViTs process images as sequences of patches, capturing global dependencies more effectively.

## Model Performance

The table below summarizes the performance of each model, including accuracy, number of epochs required to achieve it, and total training time.

| Model Type                 | Accuracy | Epochs | Training Time |
| -------------------------- | -------- | ------ | ------------- |
| CNN                        | XX%      | XX     | XX minutes    |
| ResNet (Transfer Learning) | XX%      | XX     | XX minutes    |
| Vision Transformer         | XX%      | XX     | XX minutes    |

## Confusion Matrices

Below are the confusion matrices for each model, providing insights into their classification performance.

### CNN Model Confusion Matrix

```
[Insert confusion matrix here]
```

### ResNet Model Confusion Matrix

```
[Insert confusion matrix here]
```

### Vision Transformer Model Confusion Matrix

```
[Insert confusion matrix here]
```

## Conclusion

Based on the evaluation metrics, the best-performing model is **[insert best model]**, which achieved the highest accuracy and efficiency in terms of training time and classification performance. While the CNN model provides a strong baseline, the ResNet model benefits from pre-trained knowledge, and the Vision Transformer explores a novel approach to image recognition. The choice of model depends on the specific requirements, such as computational efficiency or accuracy.

Further improvements could involve hyperparameter tuning, data augmentation, or ensembling multiple models for improved accuracy.

## Running the Code

```
python src/main.py -t --model_type [cnn,tl,vit] -m --mode [train,test] -s --save [best,last] #-e --epochs [num_epochs] --lr [learning_rate] -p --patience [patience] -f --print_freq [print_freq]
```
