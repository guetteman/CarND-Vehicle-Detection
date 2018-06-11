# **Vehicle Detection and Tracking**

![alt text][image1]

---

[//]: # (Image References)

[image1]: ./output_images/main_image.png "Advanced Lane Finding"
[image2]: ./output_images/dataset.png "Dataset"
[image3]: ./output_images/original_hog.png "Original vs. HOG"
[image4]: ./output_images/find_cars.png "Finding Cars in Image"
[image5]: ./output_images/find_cars_more_windows.png "Finding Cars in Image with More Sliding Windows"
[image6]: ./output_images/heatmap.png "Original vs heatmap"
[image7]: ./output_images/final_boxes.png "Final Image"
[image8]: ./output_images/hls_channels.png "HLS Image Splitted in three channels"
[image9]: ./output_images/binary_image.png "Binary Image"
[image10]: ./output_images/histogram.png "Histogram"
[image11]: ./output_images/sliding_window.png "Sliding Window and Fit a Polynomial"
[image12]: ./output_images/skip_sliding_window.png "Skip Sliding Window and Fit a Polynomial"
[image13]: ./output_images/new_image.png "Image with Lane Lines Found"
[image14]: ./output_images/new_image_with_data.png "Final Image"

### This is the fifth project of self-driving cars engineer nanodegree. In this project we will detect and make tracking of cars.

---

**Vehicle Detection and Tracking project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### My project includes the following files:
* **project.ipynb** containing all the code.
* **project_video_out.mp4** which shows a working example for project video.
* **test_video_out.mp4** which shows a working example for test video.
* **writeup.md/README.md** summarizing the results.

This project has 2 parts:
* Define step by step the image processing.
* Define a complete pipeline to process videos.

I splitted the first part of the project in 10 steps:
1.  Loading Data
2.  Data Visualization
3.  Data Augmentation (not used at the end)
4.  Image to Histogram of Oriented Gradients Conversion
5.  HOG Visualization
6.  Combine Features and Training Classifier
7.  Find Vehicle in Image
8.  Testing with Different Sliding Window Ranges
9.  Applying Headmap
10. Drawing Labeled Bounding Boxes

# Pipeline for a Image

## 1. Loading Data

In this step I loaded the vehicle and non vehicle data (file arrays) with the next code:

```python
import glob

vehicle_images = glob.glob('./dataset/vehicles/**/*.png')
vehicles_count = len(vehicle_images)
print(vehicles_count)
non_vehicle_images = glob.glob('./dataset/non-vehicles/**/*.png')
non_vehicles_count = len(non_vehicle_images)
print(non_vehicles_count)
```

## 2. Data Visualization

Next I visualized random data from the dataset. First I defined a method to show grid of images, and then showed 10 vehicles and 10 non-vehicle images.

```python
import matplotlib.pyplot as plt
%matplotlib inline

def plot_images(nrows, ncols, figsize, images, axis=False, axs_titles=[], title_fontsize=30, height_limit=None, width_limit=None, cmap=None):
    if height_limit != None:
        height = height_limit
    else:
        height = img.shape[0]
        
    if width_limit != None:
        width = width_limit
    else:
        width = img.shape[1]
    
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(hspace = .2, wspace=.05)
    axs = axs.ravel()
    
    for i, image in enumerate(images):
        if not axis:
            axs[i].axis('off')
        
        if len(axs_titles) > 0:
            axs[i].set_title(axs_titles[i], fontsize=title_fontsize)
        
        axs[i].set_xlim([0,width])
        axs[i].set_ylim([height,0])
        
        axs[i].imshow(image, cmap=cmap)
    return axs
```

```python
import numpy as np
import cv2

images = []
titles = []

for i in range(10):
    img = cv2.imread(vehicle_images[np.random.randint(0, vehicles_count)])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    images.append(img)
    titles.append('Vehicle ' + str(i+1))

for i in range(10):    
    img = cv2.imread(non_vehicle_images[np.random.randint(0, non_vehicles_count)])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    images.append(img)
    titles.append('Non-Vehicle ' + str(i+1))

axs = plot_images(4, 5, (16, 16), images, axs_titles=titles, title_fontsize=12)
```
which results:

![alt text][image2]

## 3. Data Augmentation (not used at the end)

I tried to use data augmentation by flipping, rotating, scaling and changing brightness. But As I was testing on my computer I got memory issues when I used augmentation. So, at the end I didn't used it. But here is the code:

```python
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img
```

```python
new_vehicle_images = []
new_non_vehicle_images = []

for file in vehicle_images:
    image = mpimg.imread(file)
    new_vehicle_images.append(transform_image(image,10,5,5,brightness=1))
    
for file in non_vehicle_images:
    image = mpimg.imread(file)
    new_non_vehicle_images.append(transform_image(image,10,5,5,brightness=1))
```

## 4. Image to Histogram of Oriented Gradients Conversion

In this step I defined a method to get HOG features:

```python
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

## 5. HOG Visualization

Next, I picked random vehicle and non-vehicle images and then applied `get_hog_features` method, then I showed the original image and HOG image:

```python
import matplotlib.image as mpimg

vehicle_img = mpimg.imread(vehicle_images[np.random.randint(0, vehicles_count)])
gray_vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2GRAY)
vehicle_features, vehicle_hog = get_hog_features(gray_vehicle_img, 9, 8, 8, vis=True, feature_vec=True)

non_vehicle_img = mpimg.imread(non_vehicle_images[np.random.randint(0, non_vehicles_count)])
gray_non_vehicle_img = cv2.cvtColor(non_vehicle_img, cv2.COLOR_RGB2GRAY)
non_vehicle_features, non_vehicle_hog = get_hog_features(gray_non_vehicle_img, 9, 8, 8, vis=True, feature_vec=True)

images = [vehicle_img, vehicle_hog, non_vehicle_img, non_vehicle_hog]
titles = ['Vehicle Image', 'Vehicle HOG', 'Non-Vehicle Image', 'Non-Vehicle HOG']

axs = plot_images(2, 2, (10, 10), images, axs_titles=titles, title_fontsize=12, cmap='gray')
```

which results:

![alt text][image3]

## 6. Combine Features and Training Classifier

I this step, I extracted features from HOG Images, then I trained the classifier:

```python
def convert_to_cspace(img, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: out_img = np.copy(img)
        
    return out_img        

def extract_features(imgs, augmented_imgs=False, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        if(not augmented_imgs):
            image = mpimg.imread(file)
            image = (image * 255).astype(np.uint8)
        else:
            image = file
            image = (image * 255).astype(np.uint8)
        # apply color conversion if other than 'RGB'
        feature_image = convert_to_cspace(image, cspace=cspace)      

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
  
        features.append(hog_features)
    # Return list of feature vectors
    return features
```

```python
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

t=time.time()
vehicle_features = extract_features(vehicle_images, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
non_vehicle_features = extract_features(non_vehicle_images, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
#augmented_vehicle_features = extract_features(new_vehicle_images, augmented_imgs=True, cspace=colorspace, orient=orient, 
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel)
#augmented_non_vehicle_features = extract_features(new_non_vehicle_images, augmented_imgs=True, cspace=colorspace, orient=orient, 
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel)

#vehicle_features += augmented_vehicle_features
#non_vehicle_features += augmented_non_vehicle_features

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
print(len(vehicle_features), len(non_vehicle_features))
# Define the labels vector
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
#svc = GridSearchCV(svr, parameters)
t=time.time()
svc.fit(X_train, y_train)

# Use a linear SVC 
#svc = LinearSVC()

# Check the training time for the SVC
#t=time.time()
#svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```
which result:

```python
78.97 Seconds to extract HOG features...
8792 8968
Using: 11 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1188
39.74 Seconds to train SVC...
Test Accuracy of SVC =  0.9854
My SVC predicts:  [ 0.  0.  1.  1.  1.  1.  1.  1.  0.  0.]
For these 10 labels:  [ 0.  0.  1.  1.  1.  1.  1.  1.  0.  0.]
0.02741 Seconds to predict 10 labels with SVC
```

I tested with various color spaces and I realized that YUV was the best choice. But, in `hog` method I used `transform_sqrt=True` I had to scale image like this `image = (image * 255).astype(np.uint8)`. Another thing is that I tested with a `LinearSVC` but I get a max accuracy of 96% so I decided to use the standard `SVC` with `C=1.0, kernel='rbf'` which have an accuracy of 98.54% as you can see.

## 7. Find Vehicle in Image

In this step, I defined a method (almost duplicated from course) to find cars in image:

```python
def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, cspace='RGB'):
    coordinates = []
    
    img_tosearch = img[ystart:ystop,:,:]
    
    ctrans_tosearch = convert_to_cspace(img_tosearch, cspace=cspace)  
    ctrans_tosearch = (ctrans_tosearch * 255).astype(np.uint8)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=(32, 32))
            #hist_features = color_hist(subimg, nbins=32)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1))   
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                coordinates.append((xleft,ytop)) 
            
    return coordinates, scale, ystart

def draw_rectangles(img, coordinates, scale, ystart):
    window = 64
    rectangles = []
    
    for coordinate in coordinates:
        xleft = coordinate[0]
        ytop = coordinate[1]
    
        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        
        rectangle = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
        rectangles.append(rectangle)
        cv2.rectangle(img,rectangle[0], rectangle[1],(0,0,255),6)
    
    return img, rectangles
```

Then, I tested it with test images provided by udacity:

```python
test_img = mpimg.imread('./test_images/test5.jpg')

ystart = 400
ystop = 500
scale = 1.5
    
coordinates, scale, ystart = find_cars(test_img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, colorspace)

draw_img = np.copy(test_img)
draw_img, rectangles = draw_rectangles(draw_img, coordinates, scale, ystart)

plt.imshow(draw_img)
```

which returns

![alt text][image4]

## 8. Testing with Different Sliding Window Ranges

Here I defined a method to detect cars in different windows, because, as we know, as closer is the car, the bigger it is. also, this will welp to assure if it is a car.

```python
def find_cars_more_windows(params, img, svc, orient, pix_per_cell, cell_per_block, colorspace):
    draw_img = np.copy(img)
    rectangles = []
    
    for param in params:
        ystart = param[0]
        ystop = param[1]
        scale = param[2]
        coordinates, scale, ystart = find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, colorspace)
        draw_img, partial_rectangles = draw_rectangles(draw_img, coordinates, scale, ystart)
        rectangles += partial_rectangles
    
    return draw_img, rectangles
```
testing with a image,

```python
test_img = mpimg.imread('./test_images/test4.jpg')

colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2

# (ystart, ystop, scale)
params = [
    (400, 464, 1),
    (400, 500, 1.5),
]



draw_img, rectangles = find_cars_more_windows(params, test_img, svc, orient, pix_per_cell, cell_per_block, colorspace)
    
plt.imshow(draw_img)
```
which results:

![alt text][image5]

As we can see, this improve the cars finding. I only defined two windows:

```python
# (ystart, ystop, scale)
params = [
    (400, 464, 1),
    (400, 500, 1.5),
]
```

When I added more windows, it started to found false positives.

## 9. Applying Headmap

Once the rectangles are found, we can apply the headmap:

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_heatmap(draw_img, rectangles):
    heat = np.zeros_like(draw_img[:,:,0])

    # Add heat to each box in box list
    heat = add_heat(heat, rectangles)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)
    
    return heatmap, labels
```

```python
from scipy.ndimage.measurements import label

heatmap, labels = apply_heatmap(draw_img, rectangles)

fig = plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
```

which results

![alt text][image6]

As we can see, the image shows with a strong with yellow/white spots which tell us where the cars are.

## 10. Drawing Labeled Bounding Boxes

Finally, We draw boxes for the strongest points in the image.

```python
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

```python
# Draw bounding boxes on a copy of the image
draw_img = draw_labeled_bboxes(np.copy(test_img), labels)
# Display the image
plt.imshow(draw_img)
```

wchich results:

![alt text][image7]


# Pipeline for videos

Now I defined the pipeline for the videos:

```python
class Detections():
    def __init__(self):
        self.rectangles = [] 
        
    def add_new_rectangles(self, new_rectangles):
        self.rectangles += new_rectangles
        
        if len(self.rectangles) > 20:
            self.rectangles = self.rectangles[len(self.rectangles)-20:]

def detect_vehicles(img):
    colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2

    # (ystart, ystop, scale)
    params = [
        (400, 500, 1.5),
        (400, 530, 2),
    ]
    
    draw_img, rectangles = find_cars_more_windows(params, img, svc, orient, pix_per_cell, cell_per_block, colorspace)
    
    if len(rectangles) > 0:
        detections.add_new_rectangles(rectangles)
    
    heatmap, labels = apply_heatmap(draw_img, detections.rectangles)
    
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img
```

then, I tested for test video and project video:

```python
from moviepy.editor import VideoFileClip

detections = Detections()

test_out_file = 'project_video_out.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(detect_vehicles)
%time clip_test_out.write_videofile(test_out_file, audio=False)
```

<a href="https://youtu.be/usGqyDuXK68" target="_blank">Here's a link to my video result</a>
---

# Conclusion

When I tested my code with the challenges videos I saw the next issues:

* At the start of the project video, the code "finds" a car in left mountains, but then it improve tracking.
* As I saw, the best color space for this project is YUV with `transform_sqrt=True` to help to improve robustness against shadows. Other color spaces has a lot o noise.
* Also, I've found that using standard SVC with `kernel='rbf' and C=1` worked a lot better than LinearSVC.
* I would like to try with more data, like the udacity dataset, but my computer has memory limitations, I think that the problem at the start of the video is because I need more data.
* Also it would be nice to try a different algorithm.
* I just focused on HOG features because when I added bin_spatial and color history didn't work as expected.

As always, I enjoyed it a lot and I already want to see what is comming next.

## See you in the next project!  

