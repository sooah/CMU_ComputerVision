# CMU_ComputerVision

This is a summary of the task.

### HW1 
#### Image Filtering and Hough Transform
- Hough Transform Line Parametrization
- Convolution
- Edge detection
- The Hough transform
- Finding lines
- Fitting line segments for visualization
- Use MATLAB
---------------------------------------------

### HW2
#### Scne recognition with bag of words
##### Part1. Build Visual Words Dictionary
- Extract Filter Responses
- Collect sample of points from image
- Compute Dictionary of Visual Words

##### Part2. Build Visual Scne Recognition System
- Convert image to word map
- Get Image Features
- Build Recognition System - Nearest Neighbors

##### Part3. Evaluate Visual Scne Recognition System
- Image Feature Distance
- Evaluate Recognition System - NN and kNN

- Using Python, OpenCV
---------------------------------------------

### HW3
#### Neural Networks for Recognition
- Network Initialization
- Forward Propagation
- Backwards Propagation
- Training Loop
- Numerical Gradient Checker
- Training Models
---------------------------------------------

### HW4
#### Augmented Reality with Planar Homographies
##### 1. Homographies
- Planar Homographies as a Warp
	- Homography
- The Direct Linear Transform
	- Correspondences
- Using Matrix Decompositions to calculate the homography
- Eigenvalue Decomposition
- Singular Value Decomposition
- Theory
	- Homography under rotation
	- Understanding homographies under rotation
	- Limitations of the planar homography
	- Behavior of lines under perspective projections

##### 2. Computing Planar Homographies
- Feature Detection and Matching
	- FAST Detector
	- BRIEF Descriptor
	- Matching Methods
	- Feature Matching
	- BRIEF and Rotations
- Homography Computation
	- Computing the Homography
- Homography Normalization
	- Homography with normalization
- RANSAC
	- Implement RANSAC for computing a homography
- Automated Homography Estimation and Warping
	- Puttin it together

##### 3. Creating your Augmented Reality application
- Incorporating video
---------------------------------------------

### HW5
#### 3D Reconstruction
##### 1. Sparse Reconstruction
- Implement the eight point algorithm
- Find epipolar correspondences
- Write a function to compute the essential matrix
- Implement triangulation
- Write a test script that uses data/temple_coords.npz

##### 2. Dense Reconsturction
- Image Rectification
- Dense window matching to find per pixel disparity
- Depth map
---------------------------------------------

### HW6
#### Video Tracking
##### 1. Lucas-Kanade Tracker
- Lucas-Kanade Forward Addictive Alignment with Translation
- Lucas-Kanade Forward Addictive Alignment with Affine Transformation
- Inverse Compositional Alignment with Affine Transformation
- Test Algorithm
