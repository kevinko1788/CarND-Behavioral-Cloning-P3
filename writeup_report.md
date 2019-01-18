# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 

## Solution to the Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project ##
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report (You are reading it)


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 3 convolution layers of filter size 5x5 and follow by two 3x3 filter convolution layers. In the beginning of the model, I used Kera Lambda function to normalize (code line 88) and cropped the image's unnecessary parts using Cropping2D (code line 91).
After 5 conv layers, I flattend to fit fully-connected layer. And the 4 fully-connected layers generates 100-> 50 -> 10 -> 1 outputs. I used dropout of 0.25 after the first fully-connected layer to avoid overfitting.

ELU activation function was chosen after I read [this](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/), it seems like ELU is a greate alternative and the benefit of ELU activation function is it genrate negative outputs as well. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting.The model contains dropout layers in order to reduce overfitting (model.py line 121) at the rate of 25%. In addition to that, I splitted the samples to train_samples and validation smaples by ratio of 0.2

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137). by lr = 1e-4.

#### 4. Appropriate training data

I used data provided from Udacity, the images have three camera angles. I was able to pull all three and used it as training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As the instructions from Udacity suggest, I thought that nVidia is model is pretty powerful and seemed like it is a good starting point instead of grinding from scratch. The model explaination can be found on the web. [nVidia Autonomous Car Group](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). on the beginning of the model, I normalized the image and cropped top and the bottom of the images to get rid of pixels that we don't necessary. The diagram below is a capture of nVidia model architecture.
![alt text][image1]

The hardest part for me was to create the generator. My initial generator had for loop inside of for loop, therefore it iterates so many times per batch request from the model.fit_generator(). I spent hours of compling and I could not finished the training. 
I knew it is not the training model because it is proven model from nVidia.
After hours of debugging.  I found that my generator has loop issue. After the fix, it only took few minutes to complete an epoch.

After an epoch training, I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 1) right after the bridge 2) sharp corners.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The Final architecture summary as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

Since the control for the simulator is pretty hard even though I am a huge gamer!
I couldn't capture relatively good driving behavior. I decided to use provided data from Udacity. I initially, total number of the data set is around 24,000, this number includes cetner, left and right cameras.

To utilize provided data, I decided to use images from all three angles. I +- tune angle value of 0.2 when I read the data from the .csv. Also, flipped the images to augment the data set, this will cover both left and right swerve. It helps vehicle going back to middle lane whenever it swevers. If the data wasn't enought to recover to center lane, I would have to capture more recovering to center lane data. But, it was not necessary. Data size is doubled just by flipping the images. 

 Random brightness may be an additional filter option to me, but I don't think the brightness will change too much since the simulator environment seems like there is not much of change in brightness unlike real road environment (at least for the first track).

The data is also randomly shuffled before pre_proceed the images. One quick improve I can make to this model is flip first then do pre_proceed the data. That way, I can eliminate bias per each batch because a batch has images and flips of the images in current model. This may create bias.

I finally randomly shuffled the data set and put 0.2 of the data into a validation set. (code line 74)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as I tried to train the model 20 times. After 3 epochs, val_loss does not get improve. 
