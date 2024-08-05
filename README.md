# PrintGuard

Detecting 3D printing faults in real-time using computer vision and machine learning techniques by analyzing print layer consistency and filament deposition to enhance print quality, improve efficiency, and minimize material waste.

## Table of Contents
1. [Introduction](#introduction)
2. [Comparison of approaches](#comparison-of-approaches)
3. [Implementation](#implementation)
4. [Technologies](#technologies)
5. [Dataset](#Dataset) 
6. [Technologies](#technologies)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [References](#references)


## Introduction

Rapid technological progress has revolutionised various industries, including manufacturing and prototyping.  An innovation that has gained considerable attention is 3D printing. 3D printing, also known as additive manufacturing, allows for the production of three-dimensional objects by sequentially depositing materials in layers according to a digital model. Although the use of 3D printers is increasing, 3D printers are prone to various issues and errors. Even a small error during printing can result in a significant loss of time and resources.  

Timely identification of these errors enables users to intervene and rectify the issue during printing and avert the need to restart the printing process, preventing the wastage of printing materials and time. One such common error related to 3D printing is under-extrusion. Under-extrusion in 3D printing is caused by insufficient filament deposition by the printer during printing (in FDM 3D printers). It results in poor print quality and reduced structural strength and surface finish. In recent years, researchers have attempted to detect 3d printer extrusion-related issues of several printing technologies using various machine-learning techniques. The following table contains details about the machine-learning methods used by researchers.

| Citation | Description | Techniques used |
|----------|----------|----------|
| (Zhang, Fidan and Allen, 2020) | This study proposed a method for increasing the efficiency of FDM 3D printing by identifying 3d print failures early on in the printing process. They have used images captured by a camera and a deep learning model to predict various extrusion-related issues of 3d prints. They were able to achieve an overall accuracy of 70%. | Convolutional Neural Network (CNN) |
|(Farhan Khan et al., 2021) | This paper proposed a CNN-based method for identifying various 3d print issues, including under extrusion, using images captured from a camera during printing. They were able to achieve an accuracy of 84%. | CNN |
| (Lut et al., 2023) | This study compares the effectiveness of different YOLOv5 models for detecting under-extortion in FDM 3D printers. They were able to achieve 99% accuracy by using the YOLOv5xl model | CNN (VGG-16) |
| (Gobert et al., 2018) | This paper proposed an SVM-based methodology for detecting print defects in laser-powder bed fusion (L-PBF) printers. They used CT scan data to train the classifier and used images from a DSLR camera for defect detection during printing. They were able to achieve an accuracy of 80%. | Support Vector Machine (SVM) |
|  (Ye et al., 2018) | This paper proposed a method to detect defects in selective laser melting (SLM) prints using acoustic signals during printing. They achieved an accuracy of 95% using a DBN model. | Deep Belief Network (DBN), SVM |
| (Okaro et al., 2019) | This paper proposed a method to identify the quality of the L-PBF printed parts using data gathered from 2 photodiodes during printing.  They used a semisupervised approach and achieved an accuracy of 77%. | Gaussian Mixture Model (GMM) |
| (Khanzadeh et al., 2018) | This study evaluated various supervised machine learning algorithms for the real-time identification of melt pool abnormalities of L-PBF prints by images captured by an infrared camera. KNN model achieved the best accuracy of 98%. | Decision Tree (DT), K-Nearest Neighbor (KNN), SVM, Discriminant Analysis (DA) |
| (Bacha, Sabry and Benhra, 2019) | This paper proposed a methodology for predicting print failures of FDM printers using data from various sensors (temperature, current, voltage of stepper motors and drivers and limit switch data). They achieved an accuracy of 98% using the Bayesian network model. | Naïve Bayes, Bayesian Network |


## Comparison of approaches

From the abovementioned techniques utilised by other researchers, CNN, SVM, and KNN techniques are compared and evaluated in the table below.

| Algorithm | Strengths | Weaknesses | Advantages | Disadvantages | Input data | Output data |
|----------|----------|----------|----------|----------|----------|----------|
| Convolutional Neural Network (CNN) | CNN can learn features automatically from input data, making them excel in image recognition. CNNs inherently capture spatial hierarchies of the input data, allowing them to model complex problems. | CNN requires a large amount of data for training. CNN model training is computationally expensive and time-consuming. | CNNs are efficient in image processing tasks and provide high accuracies. Transfer learning enables the building of new models efficiently using pre-trained models. | Require a large amount of data for training. Optimising hyperparameters can be a challenging task. | CNNs are predominantly utilised for image-related tasks, and the input will be a matrix of pixel values. In 3d printing failure detection, CNNs have been used with image data. | For image classification, the output will be probabilities for each class. |
| Support Vector Machine (SVM) | Exhibit strong performance in feature spaces with high dimensions. Have the ability to deal with non-linear decision boundaries. | SVMs are sensitive to outliers. Selecting an optimal kernel function is challenging. | SVMs can be used in both classification and regression problems. SVMs are effective for high-dimensional data | Selecting an optimal kernel is a challenging task. Training time can be significant with large datasets. | Capable of handling both numerical and categorical data. In 3d print failure detection SVMs have been used with image and acoustic signal data. | Output depends on the task. For classification, the output will be the class label. For regression, the predicted value for the input. |
| K-Nearest Neighbor (KNN) | Simple and easy to implement. Doesn't have a training phase. | Sensitive to outliers. Can cause significant computational costs during the prediction phase. | Easy to implement and understand. | Sensitive to outliers and noisy data. Does not work well with large datasets. | Capable of handling both numerical and categorical data. In 3d printing failure detection, KNN have been used with image data. | Output depends on the task. For classification, the output will be the class label. For regression, the predicted value for the input. |

In the domain of 3D print failure detection ML techniques such as CNN, SVM, KNN, GMM, DBN, and Decision Trees have been explored with several input data types. When failure detection using image data is considered CNN has shown promising results compared to other techniques (Lut et al., 2023) (Farhan Khan et al., 2021). Because of that, the author decided to implement the prototype using CNN.

## Implementation

### High-level diagram


<img src="./system_high_level _diagram.png" alt="system high level diagram"/>


[High-level diagram](https://drive.google.com/file/d/126doQVN5B3Wk8UDxdZOMhX1B1MAOWE5U/view?usp=sharing)

### Technologies

- Python
- Tensorflow
- opencv
- matplotlib 
- seaborn
- CNN

[![Built with](https://skillicons.dev/icons?i=python,tensorflow,opencv,figma)](/)

## Dataset

The data set used in this study was available publicly on Kaggle. It contained 81, 060 images  of size 256 x 256, belonging to 2 classes: normal and underextrution. The images were taken from a camera mounted near the 3D printer nozzle. Because it was difficult to compute that number of images 10,000 images were taken from the dataset for the study. The graph below visualises the data distribution between the 2 classes of the new dataset. 

Source: [Early detection of 3D printing issues](https://www.kaggle.com/datasets/gauravduttakiit/early-detection-of-3d-printing-issues-256256)

## Testing and Evaluation

The accuracy of the implemented model is 99.6%.  A high accuracy shows that the model is performing well in the overall classification of images.

The precision of the model implemented model is 99.2%. A high precision indicates that the model has a low false positive rate. This means when the model predicts that the printer is under-extruding, it is most likely correct.

The recall of the model is 100%. A high recall indicates that the model effectively identifies most of the under-extrusion instances. A high recall value is important for a 3d print failure detection model where classifying a positive instance as negative can cause issues.

The F1 score of the model is 99.6%. A high  F1 score indicates a good balance between precision and recall.

#### Comparison of techniques

The table below compares the accuracy of the models in related literature.

| Citation | Technique | Accuracy |
|----------|----------|----------|
| (Zhang, Fidan and Allen, 2020) | CNN | 70% |
| (Farhan Khan et al., 2021) | CNN | 84% |
| (Lut et al., 2023) | CNN (VGG-16) | 99% |
| (Gobert et al., 2018) | SVM | 80% |
| (Khanzadeh et al., 2018) | KNN | 98% |

The implemented model was able to achieve an accuracy of 99.6%, which is similar to the model implemented by (Lut et al., 2023). And perform better than other models. However, the comparison of accuracies isn’t consistent because the datasets they were evaluated on are different. But in general, it shows that CNN models perform better in 3d print error detection compared to other techniques.

#### Pros and cons of the selected technique

| Pros | Cons | 
|----------|----------|
| Can learn features automatically from input data, making them excel in image recognition | Model training is computationally expensive and time-consuming. |
| Usually yield high accuracy with minimum effort. | Hyperparameter tuning is time-consuming. |
| In the context of 3D printing, images are easy to obtain since most printers contain prebuilt cameras for monitoring the printing progress. | The trained model might be specialized for the particular printer that was used to create the dataset. |


## References

Bacha, A., Sabry, A.H. and Benhra, J. (2019). Fault Diagnosis in the Field of Additive Manufacturing (3D Printing) Using Bayesian Networks. International Journal of Online and Biomedical Engineering (iJOE), 15 (03), 110. Available from https://doi.org/10.3991/ijoe.v15i03.9375.

Build a Deep CNN Image Classifier with ANY Images. (2022). Directed by Nicholas Renotte. Available from https://www.youtube.com/watch?v=jztwpsIzEGc [Accessed 8 January 2024].

Farhan Khan, M. et al. (2021). Real-time defect detection in 3D printing using machine learning. Materials Today: Proceedings, 42, 521–528. Available from https://doi.org/10.1016/j.matpr.2020.10.482.

Gobert, C. et al. (2018). Application of supervised machine learning for defect detection during metallic powder bed fusion additive manufacturing using high resolution imaging. Additive Manufacturing, 21, 517–528. Available from https://doi.org/10.1016/j.addma.2018.04.005.

Image classification | TensorFlow Core. (no date). TensorFlow. Available from https://www.tensorflow.org/tutorials/images/classification [Accessed 8 January 2024].

Khanzadeh, M. et al. (2018). Porosity prediction: Supervised-learning of thermal history for direct laser deposition. Journal of Manufacturing Systems, 47, 69–82. Available from https://doi.org/10.1016/j.jmsy.2018.04.001.

Lut, M. et al. (2023). YOLOv5 Models Comparison of Under Extrusion Failure Detection in FDM 3D Printing. 2023 IEEE International Conference on Automatic Control and Intelligent Systems (I2CACIS). 17 June 2023. Shah Alam, Malaysia: IEEE, 39–43. Available from https://doi.org/10.1109/I2CACIS57635.2023.10193388 [Accessed 1 January 2024].

Okaro, I.A. et al. (2019). Automatic fault detection for laser powder-bed fusion using semi-supervised machine learning. Additive Manufacturing, 27, 42–53. Available from https://doi.org/10.1016/j.addma.2019.01.006.

Ye, D. et al. (2018). Defect detection in selective laser melting technology by acoustic signals with deep belief networks. The International Journal of Advanced Manufacturing Technology, 96 (5–8), 2791–2801. Available from https://doi.org/10.1007/s00170-018-1728-0.

Zhang, Z., Fidan, I. and Allen, M. (2020). Detection of Material Extrusion In-Process Failures via Deep Learning. Inventions, 5 (3), 25. Available from https://doi.org/10.3390/inventions5030025.

