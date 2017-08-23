##Writeup: Project Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/bbboxes_grid.png
[image4]: ./output_images/heatmap_and_false_positive_corr.png
[image5]: ./output_images/pipeline.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The features should be extracted for the following images

![alt text][image1]

The training of the SVM and the extraction of features is done in the `train_classifier.ipynb` file, which calls the `feature_extraction.py` file.
The `feature_extraction.py` file contains a small main() method which was used to try different parameters for the Histgram of Oriented Gradiants Algorithm.

These are my final paramters for feature extraction:

```python
cspace = `YUV`
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel='ALL'
```

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I created Excel Tables with pandas and tried to find out where the accurency of predicting the test data is highest. Then i used the above described paramters.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The following code describes a training procedure as seen in my ipython notebook `train_classifier.ipynb` in cell 5 and 6:

```python
rand_state = 815

X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features)))) 
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
```

The test accurency is ``0.99042792792792789`` which should be sufficent for our case.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slinding window meachanism is located in the notebook ``detection.ipynb`` and file ``hog_subsample.ipynb``

These are the bounding boxes i used:
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My pipeline used the above windows and the classifier to detect vehicles. As you can see below:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a lot of time fine-tuning the classifier to find the optimial parameters.
