# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[input]: ./assets/input.png "Input"
[hist_unbalanced]: ./assets/hist_unbalanced.png "Unbalanced"
[hist_balanced]: ./assets/hist_balanced.png "Balanced"
[cropped_resized]: ./assets/cropped_resized.png "Cropped and resized"
[yuv]: ./assets/yuv.png "YUV"
[loss]: ./assets/loss0_00854.png "Loss"

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
* video.mp4 video recording of my vehicle driving autonomously

The result video is als available here:
https://www.youtube.com/watch?v=1l8DhjI7Uk8

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

For saving the best fitting model I've used Keras ModelCheckpoint callback (model.py, line 116).

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with filter sizes 3x3 and 5x5 and depths between 24 and 64 (model.py, lines 66-70) 

The model includes ELU layers to introduce nonlinearity (model.py, lines 66-75), and the data is normalized in the model using a Keras lambda layer (model.py, line 65).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py, line 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 118-125). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py, line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the recorded driving on first track and the data provided in lecture.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initial approach was to use LeNet architecture with modified output layer - 1 neuron with no activation, since the problem class is a regression.

Because of the poor performance and the fact that LeNet was designed with different rationale:
1. As a classification algorithm
2. Processing images of different scale and complexity,

I decided to implement the architecture proposed by NVIDIA's in their DAVE-2 autonomous driving solution: https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

The main problem was that during evaluation (autonomous driving) the car was unstable on the track and had a tendency to slowly go to the side. After balancing the dataset (described in paragraph 3.) those symptoms disappeared.

The record of training and validation loss presented in the last paragraph show, that implemented architecture performs well - the validation loss is low and there is no overfitting.

#### 2. Final Model Architecture

The final model architecture (model.py, lines 60-76) consisted of a convolution neural network with the following layers and layer sizes:

Type | Size
--- | ---
Input | 3@66x200
Convolutional | 24@31x98
Convolutional | 36@14x47
Convolutional | 48@5x22
Convolutional | 64@3x20
Convolutional | 64@1x18
Fully connected | 100
Fully connected | 50
Fully connected | 10
Output | 1

Images are resized to 200x66 resolution and converted to YUV colorspace, to comply with NVIDIA approach (model.py, lines 63-64).

#### 3. Creation of the Training Set & Training Process

For creating the training set I combined recording of three laps on first track and the sample data provided in the lecture (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

Images from left and right cameras have (rspectively) +0.2 and -0.2 correction applied to the steering angle (helpers.py, lines 61-64).

Below are examples of the training set (steering angles displayed at the top):

![alt text][input]

To comply with chosen network architecture the images have to be resized and converted to YUV colorspace. To keep the ratio there is a cropping applied.

Cropping and resizing results:
![alt text][cropped_resized]

YUV conversion:
![alt text][yuv]

To augment the data set, I applied flippling and blurring on images (helpers.py, lines 94-102). Recorded dataset is multiplied (helpers.py, lines 66-76) and only the additional samples are processed.

Because of the nature of driving down the road, most of the dataset was representing steering angle around 0.0 (or values correlated with performing vehicle recovering), as shown on the histogram below. That leads to unbalanced training dataset:

![alt text][hist_unbalanced]

My approach to get it balanced was following:
1. Partitioning the dataset into ranges, corresponding to the histogram bins.
2. Calculating the mean value of the frequencies. Mean turned out to be the best choice - standard devation was too big, so the clipping would be too small (data would still be unbalanced), and median value was to small (becauce of those huge peaks) - clipping would remove to many samples
3. Clipping size of each parition up to the value of mean.

Results are shown below, the dataset got more balanced:
![alt text][hist_balanced]

20% of the dataset is used as the validation set, the number of samples are following:

Training set: 15243
Validation set: 3811

The training process progress is illustrated below:

![alt text][loss]

Validation loss at which the model was captured is 0.0085.