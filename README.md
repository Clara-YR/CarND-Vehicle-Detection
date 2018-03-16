# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/resize.png
[image2]: ./output_images/raw_features.png
[image3]: ./output_images/color_histogram.png
[image4]: ./output_images/car&notcar.png
[image5]: ./output_images/hog_visualize.png
[image6]: ./output_images/scale_features.png
[image7]: ./output_images/test_images.png
[image8]: ./output_images/slide_win.png
[image9]: ./output_images/draw_windows.png
[image10]: ./output_images/search_area.png
[image11]: ./output_images/sub_sampling.png
[image12]: ./output_images/heat_map.png
[image13]: ./output_images/labels.png
[image14]: ./output_images/bboxes.png
[image15]: ./output_images/pipeline.png

[video1]: ./output_videos/test_video.mp4
[video2]: ./output_videos/project_video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Files

README.md - writeup
Vehicle Detection.ipynb
output_images - a folder saving all the images used in README
output_files - a folder saving my final feature vectors and classifier
output_videos - a folder saving outputs of test and project videos


## Catalog 

* <span style='color:red'>__Part 1. Dara Preparation__</span>
	* __1.1 Load Balanced Dataset__
	* __1.2 Data Shuffel__
	* __1.3 Some Basic Fuctions__
* <span style='color:red'>__Part 2. Choose Features__</span>
	* __2.1 Raw Pixel Features__
		* 2.1.1 Extract raw pixel features from a image
		* 2.1.2 Extract raw features from images both of car and notcar
		* 2.1.3 Parameter tuning for raw features extraction
	* __2.2 Color Histogram Features__
		* 2.2.1 Extract color histogram features from a single image
		* 2.2.2 Extract color histogram features from images both of car and notcar
		* Parameter tuning for color histogram features extraction
	* __2.3 HOG Features__
		* 2.3.1 Extract HOG features from a image
		* 2.3.2 Extract HOG features from images both of car and notcar
		* 2.3.3 Parameter tuning for HOG features extraction
	* __2.4 Synthetical Features__
* <span style='color:red'>__Part 3. Train a Classifier__</span>
	* __3.1 Scale Feature Vectors__
	* __3.2 Train Classifier Via Scaled Feature Dataset__
* <span style='color:red'>__Part 4. Sliding Window__</span>
	* __4.1 Slide window__
	* __4.2 Search and Classify__
		* 4.2.1 Search in a single image window
		* 4.2.2 Search all windows in a image
	* __4.3 Hog Sub-sampling Window Search__
	* __4.4 Multiple Detections & False Positives__
		* 4.4.1 Take heat map
		* 4.4.2 Draw bounding boxes
* <span style='color:red'>__Part 5. Detection Pipeline Output__</span>
	* __5.1 Define Function to Process Video Frame__
	* __5.2 Video Detection Output__

* <span style='color:red'>__Part 6. Discussion__</span>
	* __6.1 My Consideration of Problems/Issues__
		* 6.1.1 Color Space for HOG Features
		* 6.1.2 Multiple-scale Windows
		* 6.1.3 HOG Sub-sampling Window Search
	* __6.2 My Questions__

---
# <span style='color:red'>Part 1. Dara Preparation</span>
 

## 1.1 Load Balanced Dataset 

`data_look(car_list, notcar_list)` - Define a function to return some characteristics of the dataset

> Your function returned a count of 8792  cars and 8968  non-cars
of size:  (64, 64, 3)  and data type: float32
Now we have a balanced dataset (have almost as many positive as negative examples)

## 1.2 Data Shuffle

## 1.3 Some Basic Fuctions

`cspace_convert(image, color_space)` - The function converts rgb img into specified color space as a feature image.

    
`update_progress(job_title, progress)` - The function is an auxiliary tool to display a progress bar of other function.


---
# <span style='color:red'>Part 2. Choose Features</span>

## 2.1 Raw Pixel Features

### 2.1.1 Extract raw pixel features from a image

Here is an example of 
![alt text][image1]

* `bin_spatial(img, spatial_size=(32,32))` - The function computes binned color histogram features in both y and x directions.

![alt text][image2]

### 2.1.2 Extract raw features from images both of car and notcar

* `extract_raw_features(imgs_list, color_space='RGB', spatial_size=(32, 32), progress_bar=True)` - The function extracts features from a list of images. It calls `bin_spatial()`


### 2.1.3 Parameter tuning for raw features extraction

I tried various combinations of parameters and finally selected:

```
# Set parameters
p_ss = (16,16) # spatial space
p_cs_raw = 'RGB' # color space for raw features
```



## 2.2 Color Histogram Features

### 2.2.1 Extract color histogram features from a single image
![alt text][image3]

### 2.2.2 Extract color histogram features from images both of car and notcar

### 2.2.3 Parameter tuning for color histogram features extraction

I tried various combinations of parameters and finally selected:

```
# Set parameters 
p_hb = 16  # hist_bins
p_br = (0, 256)  # bins_range
p_cs_color = 'LUV'  # color space for color histogram: 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'
p_pb = True  # whether to show a progress bar
```

## 2.3 HOG Features

### 2.3.1 Extract HOG features from a image

* `get_hog_features(channel_img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True)` - The function accepts params and returns HOG features (optional flattened) and an optional matrix for visualization. Features will always be the first return (flattened is feature_vector=True). A visualization matrix will be the second return if vis=True.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image5]

### 2.3.2 Extract HOG features from images both of car and notcar

* `extract_hog_features(imgs_list, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, progress_bar=True)` - Define a function to extract hog features from a list of images. It calls `get_hog_features()`



### 2.3.3 Parameter tuning for HOG features extraction

I tried various combinations of parameters and finally selected:

```
# Set parameters 
p_cs_hog = 'HLS' # color space for hog: 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'
p_o = 9 # orient for hog
p_ppc = 8 # pix_per_cell
p_cpb = 2 # cell_per_block
p_hc = 1 # hog_channel: 0, 1, 2 or 'All'
```

## 2.4 Synthetical Features

* `extract_features(imgs_list, color_space=['RGB', 'YCrCb', 'LUV'], spatial_size=(32,32) ,hist_bins=32, bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, progress_bar=False)` - The function extracts both color and HOG features from a list of images. It calls `extract_raw_features`, `extract_color_features()` and `extract_hog_features()`

Parameter selections above are reused here to optimize my classifier:


---
# <span style='color:red'>Part 3. Train a Classifier</span>
 

## 3.1 Scale Feature Vectors

![alt text][image6]

## 3.2 Train Classifier Via Scaled Feature Dataset

I trained a linear SVM using raw, color histogram and HOG features. 

* `train_classifier(car_features, notcar_features)` - The function take both car and notcar features, scales them and train a classifier for that features.

**Previous parameters setting**

|Parameters|Values|
|:---:|:---:|
|show progress bar|True|
|color_space|['RGB', 'LUV', 'HLS']|
|spatial_size |  (24, 24) |
|hist_bins |  32 |
|bins_range |  (0, 256)|
|orientation |  12|
|pix_per_cell |  8|
|cell_per_block |  2|
|hog_channel |  1|


---
# <span style='color:red'>Part 4. Slinding Window</span>

![alt text][image7]

## 4.1 Slide window

* `slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5))` - The function takes an image, start and stop position in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y) as inputs.

* `draw_boxes(img, bboxes, color=(0, 0, 255), thick=6)` - The function takes an image, a list of bounding boxes, and optional color tuple and line thickness as inputs then draw boxes in that color on the output.

![alt text][image8]


## 4.2 Search and Classify

So far `get_hog_features()`, `bin_spatial()`, `color_hist()`, `extract_features()`, `slide_window()` and `draw_boxes()`.

### 4.2.1 Search in a single image window

* `single_img_features(img, color_space=['RGB', 'YCrCb','LUV'], spatial_size=(32,32), hist_bins=32, bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True)` -  Define a function to extract features from a single image window. This function is very similar to `extract_features()` just for a single image rather than list of images

### 4.2.2 Search all windows in a image

* `search_windows(img, windows, svc, scaler, color_space=['RGB', 'YCrCb', 'HLS'], spatial_size=(32, 32), hist_bins=32, bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, progress_bar=False)` - The function passes an image and the list of windows to be searched (output of `slide_windows()`).
![alt text][image9]

## 4.3 Hog Sub-sampling Window Search

* `find_cars(img, ystart, ystop, scale, svc, X_scaler, color_space=['RGB', 'YCrCb', 'HLS'], spatial_size=(32,32), hist_bins=32, bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0)` - The function extracts features using hog sub-sampling and makes predictions. 

__NOTE__ My `find_cars()` function can extracts raw, color histogram and HOG features on different color space, which is not achievable in the [tutorial example](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/05b03f20-8c4b-453c-b41e-9fac590278c7/concepts/c3e815c7-1794-4854-8842-5d7b96276642#). The modifies I made is discussed in Part 6.

**Addvantage of Sub-sampling**

A more efficient method for doing the sliding window approach is to extract the whole Hog features array once and extract and ravel the sub-array for the window to be searched as its HOG feature vectors.

The code below defines a single function find_cars that's able to both extract features and make predictions.

The find_cars only has to extract hog features once, for each of a small set of predetermined window sizes (defined by a scale argument), and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor that impacts the window size. The scale factor can be set on different regions of the image (e.g. small near the horizon, larger in the center).

**Only Detect Right Bottom Area**

It seems that my classifier is susceptible to the noise of shadows and trees. Besides the vehicles concentrate on the right lane side. Thus I only apply my `find_cars()` function to the area below:

![alt text][image10]

Here is the detection outcome of my sub-sampling window search:

![alt text][image11]

## 4.4 Multiple Detections & False Positives

### 4.4.1 Take heat map

* `add_heat(heatmap, bbox_list)` - The function adds _heat_ to a map for a list of bounding boxes.

* `apply_threshold(heatmap, threshold)` - The function zeros out pixels below the threshold from the heatmap.

I set __heatmap threshold = 4__.
Here are the six heatmaps correponding to the six test frames:
![alt text][image12]

### 4.4.2 Draw bounding boxes

* `draw_labeled_bboxes(img, labels)` - 
The function takes the `labels` image and put bounding boxes around the labeled regions.

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image13]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image14]


---
# <span style='color:red'>Part 5. Detection Pipeline</span>

## 5.1 Define Function to Process Video Frame

* `pipeline(frame)` 

```
The function steps:
1. Image Preprocessing
    1.1 pick out the image to be searched in a video frame
    1.2 resize to be the same size as that of train dataset
    1.3 convert to appropriate color space
2. Extract the HOG Features of the Entire Image
3. Search and Classify Features Window by Window
    3.1 extract sub-array of HOG features 
    3.2 extract other features of the sub-image
    3.3 features prediction
    3.4 draw rectangle if window prediction is positive
4. Return the drawn image and the list of positive windows
```


__Heatmap threshold = 7__ in the `pipeline` function. Here is the output of the six test frames:
![alt text][image15] 

## 5.2 Video Detection Output

The output is saved in './output_videos'

---

# <span style='color:red'>Part 6. Discussion</span>

## 6.1 My Consideration of Problems/Issues

### 6.1.1 Color Space for HOG Features

Having tried several color space combination to extract HOG features, I found out that outstanding differences between car and nor-car exist in the 'L' channel of both 'LUV' and 'HLS' color space, as well as the 'Y' channel of both 'YUV' and 'YCrCb' color space.

Although to use all three channels of a color space could improve the accuracy of the HOG features SVM more or less, I decide to use a single channel in order to save 2/3 calculation cost.

As a result, I used the 'L' channel of 'HLS' color space to extract HOG features.

### 6.1.2 Multiple-scale Windows

In general, I don't know what size my object of interest will be in the image I'm searching. So it makes sense to search in multiple scales. First of all I'll establish a minimum and a maximum scale at which I expect the object to appear, and then reasonabel number of intermediate scales to scan as well.

In this case I choose four scales:(32,32), (64,46), (96,96) and (128,128). Here are the parameters setting of my multiple-scale windows in __4.1 Scaling Windows__.

```
# Set parameters
p_xss = [[None,None], [None,None], [None,None], [None,None]] # x_start_stop
p_yss = [[400,470], [380,550], [400,600],[480,680]] # y_start_stop
p_xyw = [(32,32), (64,64), (96,96), (128,128)] # xy_window
p_xyo = [(0.5,0.5), (0.5,0.5), (0.5,0.5), (0.5,0.5)] # xy_overlap
```

### 6.1.3 HOG Sub-sampling Window Search

I made several modification of the function `find_cars` in __4.3 Hog Sub-sampling Window Search__, considering the reasons below:

**Multiple Color Space**

Tutorial example

```
ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
```

My code

```
# Color space conversion on image area to be searched
raw_feature_img = cspace_convert(img_tosearch, color_space[0])
color_feature_img = cspace_convert(img_tosearch, color_space[1])
hog_feature_img = cspace_convert(img_tosearch, color_space[2])
```

**Multiple Channel Selection for HOG**

Discussion for each channel selection was added twice in my code. Once for extraction of the entire HOG features array of the image area to be searched and another in the `for` loop when extracting sub-array of HOG feautures each time.

Tutorial example

```
# Compute individual channel HOG features for the entire image
hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```

My code

line 21~24 

```
# Call get_hog_features() with vis=False, feature_vec=True
if hog_channel=='All':
    HOG_features = []
    for channel in range(hog_feature_img.shape[2]):
        channel_HOG_feature = get_hog_features(hog_feature_img[:,:,channel],
                                               orient, pix_per_cell, cell_per_block,
                                               vis=False, feature_vec=False)
        HOG_features.append(channel_HOG_feature)
else:
    HOG_features = get_hog_features(hog_feature_img[:,:,hog_channel],
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=False)

```
line 57~65

```
# Extract HOG for this patch
if hog_channel=='All':
    hog_features = []
    for channel_HOG_feature in HOG_features:
        channel_hog_feature = channel_HOG_feature[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
        hog_features.append(hog_feat_i)
    hog_features = np.ravel(hog_features)
else:
    hog_features = HOG_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
``` 

## 6.2 My Questions

### 6.2.1 Rotate raw feature image 90 degrees

I wonder whether rotate the feature image 90 degrees will get a totally different group of raw features.

Actually I've tried the code below:

```
def bin_spatial(img, spatial_size=(24,24)):
    resize_img = cv2.resize(img, spatial_size)
    features_y = resize_img.ravel()
    features_x = np.rot90(resize_img).ravel()
    features = np.concatenate((features_y, features_x))
    
    return features
```
However it didn't improve the accuracy of car detection in test images comparing with the original code:

```
def bin_spatial(img, spatial_size=(24,24)):
    resize_img = cv2.resize(img, spatial_size)
    features = resize_img.ravel()
    
    return features
```

### 6.2.2 How to apply while loop in pipeline?

I wanted to take use of the bounding boxes of last frame to improve the accuracy of the current frame. But I forgot how apply a `while true` loop in the function to add an updatable variable according to the outcome of last function calculation.
              