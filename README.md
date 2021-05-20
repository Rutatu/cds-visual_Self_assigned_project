# cds-visual_Self_assigned_project


***Self-assigned project for visual analytics class at Aarhus University.***

***2021-05-19***



#  Emotion recognition using transfer learning and grid search for fine-tuning 

## About the script


This assignment is a self-assigned project. This project focuses on an emotion recognition task using transfer learning from a pre-trained VGG-Face deep CNN. The task is to classify images of facial expressions into 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. The script outputs the classification report and a performance graph. After training and evaluating the CNN, the script, using grid search method, returns the best parameters based on highest achieved accuracy.


## Methods

The problem of the task relates to classifying emotions. To address this problem, I have used transfer learning from a pre-trained VGG-Face deep CNN. VGG-Face is based on  VGG-Very-Deep-16 CNN architecture and is evaluated on the Labeled Faces in the Wild (Huang et al., 2007) and the YouTube Faces (Wolf et al., 2011) dataset. The backbone of the VGG-Face CNN is a deep CNN with 18 weighted layers organized into five blocks. Each block contains 2 or 3 convolutional layers, followed by a max-pooling layer.  The primary goal of this project is to learn the basics of transfer learning and CNNs instead of building the best performing model, therefore, to speed up the training process, the classifier layer was simplified by adding on top a new fully-connected classifier relevant to this project´s task. The three fully connected classifier layers of the original architecture: fc-4096, fc-4096 and fc-2622, were replaced by two smaller ones: fc-256 and fc-7 followed by a softmax activation function. The weights were updated during the training. 

Depiction of the modified model´s architecture can be found in folder called ***'output'***.


## Repository contents

| File | Description |
| --- | --- |
| data | Folder containing the data for the project |
| data/Face_emotions | Data folder with training and test sets |
| output | Folder containing files produced by the script |
| output/Emotions_classifier_report.csv | Classification metrics of the model |
| output/Emotions_classifier_performance.png | Model´s performance graph |
| output/VGG-Face_CNN´s_architecture.png | Depiction of CNN model´s architecture used |
| src | Folder containing the script |
| src/emotion_class.py | The script |
| README.md | Description of the assignment and the instructions |
| emotion_venv.sh | bash file for creating a virtual environmment  |
| kill_emotion.sh | bash file for removing a virtual environment |
| requirements.txt| list of python packages required to run the script |



## Data

For this project The Facial Expression Recognition 2013 (FER-2013) dataset was used. The dataset consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. Images fall into seven categories: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The training set consists of 28,709 examples and the test set consists of 7,178 examples.

__Data structure__

Before executing the code make sure that the images are located in the following path: ***'data/Face_emotions'***

***'Face_emotions'*** folder should contain two folders: train and test, each of which contains seven folders labeled by an emotion.
The code should work on any other similar image data structured this way, however the model parameters and preprocessing might require readjustments based on different data.


__Data preprocessing__

The only pre-processing of the data was the subtraction of the mean RGB value, computed on the training set, from each pixel. The same pre-processing step was applied to the ImageNet images for the VGG16 model, on which VGG-Face CNN is based on. This was performed using Keras function ```preprocess_input()```.



## Intructions to run the code

The code was tested on an HP computer with Windows 10 operating system. It was executed on Jupyter worker02.

__Code parameters__


| Parameter | Description |
| --- | --- |
| train_data  (trd) | Directory of training data |
| val_data (vald) | Directory of validation data |
| optimizer (optim) | A method to update the weight parameters to minimize the loss function. Choose betweeen SGD and Adam |
| learning_rate (lr) |The amount that the weights are updated during training. Default = 0.001 |
| epochs (ep) |Defines how many times the learning algorithm will work through the entire training dataset. Default = 50 |


__Steps__

Set-up:

```
#1 Open terminal on worker02 or locally
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-visual_Self_assigned_project.git  

#4 Navigate to the newly cloned repo
$ cd cds-visual_Self_assigned_project

#5 Create virtual environment with its dependencies and activate it
$ bash emotion_venv.sh
$ source ./emotion_venv/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the script
$ cd src

#7 Run the code with default parameters
$ python emotion_class.py -trd ../data/Face_emotions/train -vald ../data/Face_emotions/test -optim Adam

#8 Run the code with self-chosen parameters
$ python emotion_class.py -trd ../data/Face_emotions/train -vald ../data/Face_emotions/test -optim SGD -lr 0.003 -ep 100

#9 To remove the newly created virtual environment
$ bash kill_emotion.sh

#10 To find out possible optional arguments for the script
$ python emotion_class.py --help


 ```

I hope it worked!


## Results

Facial expressions are very difficult to classify for the computer due to complicated muscle movements, therefore, deep learning techniques need to be employed in order to achieve at least mediocre results. This project showed how transfer learning can be used for an emotion classification problem based on extracted facial expression features. The pre-trained VGG-Face deep CNN (optimizer = Adam, learning rate = 0.001)  achieved a weighted average accuracy of 41% for correctly classifying faces according to their emotional expression. Such results are not satisfactory, thus, having more data or/and fine-tuning hyperparameters of the model might increase the accuracy.



## References

Brownlee, J. (2016, August 9). How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras. Machine Learning Mastery  https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

Huang, G. B., Ramesh,M., Berg, T., Learned-Miller, E. (2007)  Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst

Parkhi, O. M.,  Vedaldi, A., and Zisserman, A. (2015). Deep Face Recognition (poster).  
University of Oxford. [https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/poster.pdf]

Serengil, S. (2018, August 6). Deep Face Recognition with Keras. Sefik Ilkin Serengil
https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

Wolf, L., Hassner, T., Maoz, I. (2011). Face Recognition in Unconstrained Videos with Matched Background Similarity. Computer Vision and Pattern Recognition (CVPR)




