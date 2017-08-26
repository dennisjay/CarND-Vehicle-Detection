import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from lesson_functions import get_hog_features, color_hist, bin_spatial

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace, orient,
                        pix_per_cell, cell_per_block, hog_channel, nb_bins, spatial_size):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        #Calculate color hist
        color_features = color_hist(image, nb_bins)
        spatial_features = bin_spatial(image, size=spatial_size)

        # Append the new feature vector to the features list
        features.append(np.hstack([hog_features]))
    # Return list of feature vectors
    return features


def train(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, nb_bins, spatial_size ) :
    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, nb_bins=nb_bins, spatial_size=spatial_size)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, nb_bins=nb_bins, spatial_size=spatial_size)
    t2 = time.time()
    hog_time = round(t2-t, 2)
    print(hog_time, 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    train_time = round(t2-t, 2)
    print(train_time, 'Seconds to train SVC...')
    # Check the score of the SVC
    accurency = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', accurency )
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    predict_time =round(t2-t, 5)
    print(predict_time, 'Seconds to predict', n_predict,'labels with SVC')

    return accurency, hog_time, train_time, predict_time

if __name__ == '__main__':
    # Divide up into cars and notcars
    images = glob.glob('train_data/**/**/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 5000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    parameters = []

    for colorspace in ['YCrCb']:
        for orient in [9]:
            for pix_per_cell in [8]:
                for cell_per_block in [2]:
                    for hog_channel in ['ALL']:
                        for nb_bins in [32]:
                            for spatial_size in [(32, 32)]:
                                accurency, hog_time, train_time, predict_time = train(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, nb_bins, spatial_size)
                                params = {
                                    'colorspace': colorspace,
                                    'orient': orient,
                                    'pix_per_cell': pix_per_cell,
                                    'nb_bins': nb_bins,
                                    'cell_per_block': cell_per_block,
                                    'hog_channel': hog_channel,
                                    'accurency': accurency,
                                    'hog_time': hog_time,
                                    'train_time': train_time,
                                    'predict_time': predict_time,
                                    'spatial_size': spatial_size
                                }
                                parameters.append(params)

    frame = pd.DataFrame(parameters)
    print(frame.sort_values(['accurency', 'predict_time'], ascending=False))