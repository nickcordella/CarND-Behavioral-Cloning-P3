# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is largely based on NVIDIA's architecture from their paper [here](https://arxiv.org/pdf/1604.07316v1.pdf). It starts with normalizing all data around zero, with a distribution between -1 and 1. It also crops everything between the horizon and the top of the car.

After that, I employed 3 convolutional layers with a 5x5 kernel and 2x2 strides, followed by 2 convolutional layers with 3x3 kernels and 1x1 strides. `tanh` activation functions were used to introduce nonlinearities after all the convolutional layers. The end of the network consists of 4 fully connected layers, giving a final prediction on a single value, the steering angle. All the interesting parts are in lines 68-85 of `model.py`

#### 2. Attempts to reduce overfitting in the model

There are dropout layers after the 1st, 3rd, and 5th convolutional layers, as well as a dropout layer before the second-to-last fully connected layer, in order to prevent overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training and validation data sets were randomly selected using `sklearn.model_selection.train_test_split` with an 80/20 ratio. respectively. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88). I also selected a batch size of 128 because it seems like a number of our exercises have used that batch size successfully, and it seems to allow for efficient computation.

#### 4. Appropriate training data

I combined the sample data provided by the project with some of my own training data. I tried to carefully record samples of turning away from lane boundaries as well as driving in reverse around the course to avoid bias.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple to make sure everything worked and iterate from there

My first step was to use actually use a simple, fully connected single layer network to make sure the code actually generated predictions at all. It resulted in fairly meaningless wobbling around but indicated that the basic data pipeline was set up. I then looked up the NVIDIA model in their paper and did my best to reproduce it.

It wasn't working great so I then cropped the images to only focus on the relevant part of the road that should influence the steering angle. Then I also noticed that my validation losses were increasing slightly even as my training losses decreased, so I somewhat randomly inserted dropout layers to mitigate overfitting.

The final adjustment I made was to incorporate data from the left and right cameras. This followed the approach laid out in the preceding lesson, with a rough correction factor being added to each angle associated with the snapshot. After trying different factors, I decided 0.3 gave the best results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-85) consisted of a convolution neural network with the following layers and layer sizes:
Cropped Input: 75x320x3
2D convolution: 35x158x24
2D convolution: 15x77x36
2D Convolution: 5x36x48
2D Convolution: 3x34x64
2D Convolution: 1x32x64
Flattened: 2048
Fully Connected: 100
Fully Connected: 10
Fully Connected: 1 final prediction!

#### 3. Creation of the Training Set & Training Process

I tried using sample data provided by the project, as well as training data created on my own. In the end, combining both sets of data gave the best results. My strategy for creating training data was to circle carefully around the track a couple times while staying in the middle of the lane and handling curves smoothly. Then I went backwards around the track to counteract biases that might limit right-turning ability. Finally, I went around the track once intentionally veering toward the boundary while not recording, then recording a short turn away from the boundary. I also used all the camera angles from both the example data and my own data.

After the collection process, I had about 50,000 data points. Each of these were preprocessed by normalizing to a distribution between -1 and 1, and cropping all points above the hood and below the horizon

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Training for 7 epochs seemed to show sufficiently loss deterioration without overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
