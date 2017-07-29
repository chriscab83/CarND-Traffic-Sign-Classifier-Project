# **Traffic Sign Recognition** 

## Writeup Template

### ou can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/visualization.jpg "Visualization"
[image2]: ./writeup_imgs/all_classes.jpg "All Classes"
[image3]: ./writeup_imgs/all_classes_preprocessed.jpg "Normalized Dataset"
[image4]: ./writeup_imgs/my_images.jpg "My Images""
[image5]: ./writeup_imgs/my_imgs_top5.jpg "Prediction Top 5"
[image14]: ./writeup_imgs/translate.jpg "Shifted"
[image15]: ./writeup_imgs/rotate.jpg "Rotated"
[image16]: ./writeup_imgs/scale.jpg "Scaled"
[image17]: ./writeup_imgs/warp.jpg "Warped"
[image18]: ./writeup_imgs/augment.jpg "Augmented"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/chriscab83/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_2.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.


Below is an image representation of each class in the the datasets.

![alt text][image2]

Here is an exploratory visualization of the data set. It is a bar chart showing the class representations of data in the test set. It is clear that some classes are under-represneted which could cause some biases in the training of the neural net.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My first step was pad the dataset by generating images for the under-represented classes.  I generated the images by performing various augmentations including: 
1. shifting an image on the x and y axis between [-2,2] pixels

![alt text][image14]

2.  rotating an image between [-15,15] degrees

![alt text][image15]

3.  scaling an image between [-2, 2] pixels

![alt text][image16]

4.  and warping the image with opencv's warp affine transform. 

![alt text][image17]


The final augmented image looked as follows:

![alt text][image18]

I generated these images to help prevent the neural net from improperly over training on certain images that were highly represented and also undertraining on images under represented, which could cause the neural net to biasely choose the images it has learned more of.  I ran the augmentation methods over each class until each class had a minimum of 1000 images.

Next, I normalized the entire dataset by first converting the images to the YCrCb color scale to access the intensity channel of the image. I then called opencv's equalize histogram method on that channel to normalize the image across the entirety of the channel. Afterwards, I converted the image back to RGB. Lastely, I divided the entire dataset by 255 to scale the pixel values down to be between [0, 1] rather than [0, 255].

Below is an example of the augmented dataset.

![alt text][image3]

Performing this normalization helped to bring out features in the images by increasing the contrast in the image. Also, this normalization bring the mean from 78.985 to 0.503 and the standard deviation in the dataset down from 66.89 to 0.354 which should help the neural net perform better. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was based on the LeNet design covered in the lessons. I tested many changes to it, in the end the one that seemed to help the neural net the most was adding an extra convolution prior to the fully connected layers. After adding this layer, my accuracy increased slightly and my net performed better on the dataset of images I put together. My model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x3 Normalized RGB image | 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 5x5x16 |
| Convolution 2x2 | 1x1 stride, valid padding, outputs 4x4x32 |
| RELU | |
| Fully connected | inputs 512, outputs 120 |
| Dropout RELU | |
| Fully Connected | inputs 120, outputs 84 |
| Dropout RELU | |
| Fully Connected | inputs 84, outputs 43 |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model using the Adam Optimizer to minimize my reduce mean cross entropy loss operation. I had a learning rate of 0.0009 and I trained over 100 epochs with an early exit if the model had not improved its accuracy in the previous 10 epochs.  In the case of an early exit of the training, the model would role back the weights to its highest trained epoch by loading them from the saved session. I also included a dropout during training between the fully connected layers with a keep probability of 0.5 to prevent overfitting. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.8% 
* test set accuracy of 92.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  

I first chose to start with the LeNet architecture.  It is a well known architecture used in image recognition and we had already spent quite a bit of time on it during our lessons so I felt comfortable iterating from there.

* What were some problems with the initial architecture?

There were no major problems with the implmentation and it performed relatively well right out of the box.  I was very quickly able to get above the 93% mark once I normalized my dataset with the equalize histogram method. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I tested various changes to the architecture to improve performance by adding and removing layers, tuning hyperparameters, shifting dropout keep probability, etc.  In the end, I found that most of my changes did little to the overall accuracy of the neural net.  The one change that stayed in the final projet was the addition of a smaller convolution before the fully connected layer.  Prior to adding this layer, I was unable to get above 70% on the images I put together from the internet.  After adding this layer, I was able to get that test up to a 90% accuracy. 

I did not find that I was having a problem with overfitting or see very large differences in my validation accuracy when compared to my test accuracy. I am farely certain this is do to the addition of the dropout on the fully connected layers.

* Which parameters were tuned? How were they adjusted and why?

Although I did make changes to my mu and sigma parameters during while iterating, I found that they were not making large changes to my accuracy.  I did however end up shifting my learning rate as it was set a bit high and I found the loss value was jumping rather than smoothly reducing. Reducing the learning rate helped my model better settle at its highest accuracy levels.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Using confolution layers in for this project makes since because they work on images to pull out features that a traditional feed forward network may miss. It does this by not only placing importance on specific pixels, but also placing importance in how those pixels relate to the pixels around it. This works well for image  classification as the objects in the image are made up of many features such as lines, curves, and colors that the convolutional layers pick up and activate on giving them insight to the underlying object a traditional neural network may not have. 
  
 We use dropout layers to prevent overfitting of the model which would cause the model to perform very well on the data it is training on because it would over learn features unique to the data it has rather than generalizing the data into features shared by other images in the class. This overfitting would cause a high training accuracy but a poor test accuracy. The dropout layers prevent this by shutting off various input nodes in each run so the neural net has a chance to learn key features on more than one node.  
 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:
![alt text][image4]

The neural net should not have a hard time recognizing most of the images.  However, the extra signs under the slippery road image and general caution image may increase the difficulty on getting a correct prediction.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way | Right-of-way | 
| Roundabout mandatory | Roundabout mandatory |
| Slippery Road | Slippery Road |
| Road work | Road work |
| Children crossing | Children crossing |
| General caution | No passing		(X) |
| Turn right ahead | Turn right ahead |
| Stop | Stop |
| No passing | No passing |
| Speed limit (50km/h) | Speed limit (50km/h) |


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 92.9%. As predicted, the one sign it could not predict was the General Caution sign.  I believe this may have something to do with the extra text sign under the general caution as well as its flattened appearance due to the angle the picture was taken from.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

My predictions for the ten images are as follows:

![alt text][image5]

As can be seen in the above diagrams, most images the net was very confident in its predictions. 

The first exception would be the first image in which the model was only about 55% sure of its correct prediction, confusing the right of way sign with the beware of ice/snow sign.

As for the general caution sign, our incorrect prediction, it surprised me to see that the correct prediction wasn't even in the model's top 5 guesses. 

The last exception would be the speed limit sign in which it was again only just at about 55% confident in its correct prediction, however the rest of the confidence was placed in another speed limit sign of a different speed.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


I did perform step 4 of the project, curious to see how the activations on my convolutional layers looked during prediction. I was especially interested in reviewing the general caution sign that was so incorrectly predicted in the previous step.  In reviewing the data, I see that, as I expected, the model seemed to strongly key in on the square sign under the general caution sign rather than the general caution sign itself.  This could explain why it was unable to properly predict that sign in the test. The output from the test can be found in the last cell of the jupyter notebook.

