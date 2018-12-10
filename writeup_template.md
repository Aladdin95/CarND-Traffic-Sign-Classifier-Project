# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/vis.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/colored.png "Random Noise"
[image4]: ./test/ahead.jpg "Traffic Sign 1"
[image5]: ./test/70.jpg "Traffic Sign 2"
[image6]: ./test/stop.jpg "Traffic Sign 3"
[image7]: ./test/up_right.jpg "Traffic Sign 4"
[image8]: ./test/60.jpg "Traffic Sign 5"
[image9]: ./examples/gray.png "Random Noisea"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README


You're reading it! and here is a link to my [project code](https://github.com/Aladdin95/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of examples per class...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because color doesnt affect the shape of the sign, grayscale images would make the model smaller and more efficient

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

![alt text][image9]

As a last step, I normalized the image data because normalizing image would make the loss function easier to minimize because normalizing helping to avoid narrow elipse


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5x6   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5x16    | 1x1 stride, valid padding, outputs 10x10x16	|
| Fully connected		| from 400 to 120								|
| Fully connected		| from 120 to 84								|
| Fully connected		| from 84 to 43									|
| Softmax				|         										|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size = 128, 100 epochs and learning rate of 0.001
i also used dropout regulaization with keep_prob = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.9% 
* test set accuracy of 94.3%

an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
first i tried to enlarge the lenet architecture because it's designed for 10 classes and we have 43 classes

* What were some problems with the initial architecture?
the problem was that the model getting stuck in a local minima and the validation accuaracy never pass 5%

* How was the architecture adjusted and why was it adjusted? 
i tried using sigmoid instead of relu that worked but it was too slow in learning parameters so i rolled back to relu and tried another method of initializing the weights and that worked very well but the model where too big to be trained in 100 epochs so i rolled back the model of lenet5 and without enlargement and it worked really good.

* Which parameters were tuned? How were they adjusted and why?
learning rate: changing the number passed to the constructor of the optimizer
i tried to make it 0.01 instead of 0.001 to accelerate the learning process but it affect the training accuarcy so i rolled back to 0.001
conv layer size: by changing the dimention of weights and filter
dropout: i tried values of 1 and 0.5 for it 
batch size: 128 and 256
epochs: 80, 1000, 100

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
convolution layer looks for some features in the whole image and this works well in classifying models
drop out is a very good technique to reduce overfitting, the problem when i have a high accuarcy in the training set but a lower accuaracy in the validation set

If a well known architecture was chosen:
* What architecture was chosen? 
lenet5
* Why did you believe it would be relevant to the traffic sign application? 
because i studied it the course :D and it is a basic architecture that is been used in classifying models
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
very high accuarcy in all of them provided evidence that the model is working well
and a low difference from the training and validation accuracy
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the sign is warped
The second image might be difficult to classify because it is containing a number
The third image might be difficult to classify because it is noisy
The fourth image might be difficult to classify because the background color is close to sign color
The fifth image might be difficult to classify because it has less training data than other number signs

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| General caution 							| 
| Speed limit (70km/h) 	| Speed limit (70km/h)							|
| Stop					| Stop											|
| Go straight or right	| Go straight or right			 				|
| Speed limit (60km/h)	| Speed limit (60km/h)     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 44th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a General caution sign (probability of 0.922), and the image doesn't contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .92         			| General caution       						| 
| .049     				| Right-of-way at the next intersection 		|
| .029					| Speed limit (20km/h)	    					|
| .0 			    	| Pedestrians					 				|
| .0				    | Wild animals crossing 						|


For the second image, the model is sure that this is a Speed limit (70km/h) sign (probability of 1), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (70km/h)   						| 
| 0     				| Speed limit (20km/h)  						|
| 0 					| Speed limit (80km/h)							|
| 0     				| Speed limit (120km/h)			 				|
| 0		    		    | End of all speed and passing limits			|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


