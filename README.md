
# Notes Separator




# Impact

This Project will help students to automatically remove images of notes from their gallery automatically without the user looking for any particular Images.

## Aim of Project

The aim of this Project to autonomously detect whether a given image is taken from some notes or not. To achieve this we will use Convolutional Neural Network(CNN) as our tool. We will be using  different CNN architectures and choose the one with maximum accuracy parameters. 



## DataSets
Data Set was prepared with the help of the Student Body that had notes and non-notes images, separated category wise. They tried best to get un-filtered image, as filters brighten up the images that would’ve misled the program.

Data Set is used to train the model can be found at https://drive.google.com/file/d/1y-YYS6_XE_ClmshJPhKNIiixedT6Rnkn/view

The above dataset contains total of 1500 images(approx..)

850 notes images(approx..)

650	n-notes images(approx..)

Size of each images is --> 30px can be reduced to--> 15px


## Libraries/ Dependencies

-->TensorFlow

-->Keras

-->OS

-->Matplotlib.py


## Model Inputs and Features

From tweaking the model again and again, the following facts were illustrated.

1.The model showed optimum accuracy when the learning rate was reduced to 0.001

2.The training time was highly reduced without compromising with the accuracy when the first three layers had 16, 32  and 64 neurons in the first three layers.

3.The accuracy of the model highly depends on the non-notes datasets, since the non-note part has a huge diversity to offer since everything that is not a note is a non-note data.



## Accuracy
Test Accuracy- 75 %

Training Accuracy – 88.68%


## Authors and Contribution

Author- Dev Prakash Sharma

### Licence

[copy] Licence Under the MIT Licence for Open Source Software Documentation


```python

```
