+++
title = 'Autonomous Driving Review'
date = 2023-12-23T14:12:05+08:00
draft = false
tags = ["tech"]
categories = ["tutorial"]
+++
# Introduction
Automated Driving Systems (ADSs) aims to prevent traffic accidents and mitigate congestion [^1]. ADS is empowered by recent development of Deep Learning and sensor modalities (such as lidar).DARPA Grand Challenge is the first major competition in this field where human interation is prohibited. However the environment is relatively simple.  

Society of Automotive Engineers (SAE) defined five levels of driving automation from L0 to L5. L1 include simple tasks such as adaptive cruise control. L2 is partial automation such as collision avoidance. L3 is conditional automation where the driver is require quick respons to any emergencey. L4 automation does not need any human interation but only possible in certain environment or require high precision map. L5 is capable in all environemnt.

ADS is challenged by the balance between more automation and safety. Also, ethical dilemmas are hard for machine to solve. Like other transpotation, ADS needs a rigorous safety standard but related standards are yet to be established. Sharing thousands info between vehicles is hard so we need to convert raw sensors data into a common framework.
# Architecture
ADS can be classified into **ego only** or **connected system**.
## Ego Only
Perform operations in the single vehicle. Most popular method.
## Connected System
No operational connected ADS in use yet but might be the future. Vehicular Ad hoc NETwork (**VANETs**) distribute task into differne agents. **V2X** is a term that stands for 'vehicle to everything'. Vehicles are able to access information from peers and traffic signals. It solve some drawbacks of ego-only system that such as limited computation or sensing abilities.
## Modular System
Core functions of a modular ADS can be summarized as: localization and mapping, perception, assessment, planning and decision making, vehicle control, and human-machine interface. Breakdown the hard problem into pieces. Good: Easy to add functions. Bad: error propaganda and complexity.
## End-to-End Driving
Generate ego-motion(direct control such as turning) from only sensor inputs. It consists of direct supervised deep learning,  neuroevolution and the deep reinforcement learning. 
# Sensors and Hardare
## Mono Camera
A kind of **passive sensor**. Simple and direct. Influenced by illumination and no depth perception. 
## Omnidirectional Camera
Panoramic view
## Event Camera
Each pixel record change of brightness individually. High response, dynamic range low power.
## Radar
Active sensors, might interfere with other active sensors. Less accurate, cheap.
## Lidar
Light wave other than radio wave. High accuracy but influenced by weather. large size. Outperform human in terms of sensing in dark environment.
## Proprioceptive Sensors
Proprioceptive sensors measure status of the car itself (speed,acceleration, yaw). Include wheel encoder, IMU.

# Localization and Mapping
Where am I right now?
## GPS-IMU Fusion
GPS-IMU Fusion localizing the vehicle with dead reckoning. IMU errors accumulate with time and they often lead to failure in long-term operations. With the integration of GPS readings, the accumulated errors of the IMU can be corrected. Cannot be used in ADS directly with low accuracy.
## Simultaneous localization and mapping (SLAM)
Online map making and localizing the vehicle in it at the same time. No need a pre-built map but have high computational requirements and more suitable indoor.
## Priori Map-based Localization
Compare between reading and pre-built map, fnding the location of the best possible match. 
### Landmark search
In an urban environment, poles, curbs, signs and road markers can be used as landmarks. Perform road marking detection and compare with 3D map to determine location. Require abundant amount of landmarks.
### Point Cloud Matching 
Online-scanned point cloud which covers a smaller area, is translated and rotated around its center iteratively to be compared against the larger a priori point cloud map. The position and orientation that gives the best match between the two point clouds give the localized position of the sensor relative to the map. Map-making and maintaining is time and resource consuming.
### 2D To 3D Matching
Matching online 2D readings to a 3D a priori map. Only camera no need Lidar.
# Perception
Sense the environment and get information.
## Detection
### Image-based object detection
Determine size and location. Divided into:
* Single stage detection. 
High computation power, difficult to train. 
* Region proposal detection.
Fast inference, low memory cost (YOLO). 
#### Omnidirectional and Event based perception
Panoramic vision is achieved by camera arrays + extrinsic & intrinsic calibration. Widely used in 3D reconstruction + SLAM.

Event camera output asynchronous events, used for dynamic object detection.
#### Changing appearance and Illumination
Camera might fail when the appearance of scenes have changed. Can be solved by sensor fusion (lidar, radar, infrared)
### Semantic Segmentation
Precise define objects rather than bounding boxes. Usually use transposed convolution to obtain pixel resolution labels. Relatively slow, but can learn universal feature representations, can be generalized for many tasks.
### 3D Object Detection
Camera-based method is influenced by illumination and scale of scene is lacked. Depth information is necessary and can be estimated by single or stereo camera. 3D lidar collect discrete 3D points representing the surfaces of the scene. Long range and can use clustering to group points into objects. RGB-D method is limited by range and not used in ADS yet.

3D object can be represented with 3D occupancy grid (voxel), triangle mesh, multi-view representation(in 2D) and point clouds [^2].

**multi-view representation** use only 2D perception. multi-view CNN (2015) feed images from different views, combine and output prediction. But 2D view is limited and cannot represent 3D shape, also data needs to be generated by ground truth 3D model + virtual cameras. 

**Voxel grids** use quantization and each grid is represented by a probability. It has relatively low resolution and high memory consumption. VoxNet takes input of voxel and use 3D convolution + 3D pooling + FC to generate prediction. To makes it  rotation-invariant, during training they use different rotation of same object. When testing, pool the output among different rotations. 

**Point clouds** Directly operate on raw point clouds PointNet(2016). Point cloud is : scattered, unordered, invariance of transformation. Therefore, we define a **symmetric functions** to produce same output under different permutations (such as max,sum). Also, a small **T-Net** is used in each stage for invariant input transformations property. 

**Depth image**
2D projection of 3D point clouds, can be used for depth estimation. 

**Bird's eye view (BEV)**
Top-view image of point clouds, occlusion in vertical axis.
## Object Tracking
Need to obtain range information to get heading and velocity + location. Usually use sensor fusion for tracking. Object tracked in 3D space use nearest neighbor methods to track among different frames. Image-based methods require some appearance model such as gradient. After association is built, we use filtering (Bayes,Kalman). Deep Learning is also widely used mainly in image domain.
## Road and Lane Detection
Determine drivable road, not suitable to use bounding box or semantic segmentation. Need to understand lanes and their connections.

Firstly we need to decide the drivable area. Then divide them into lanes include the host lane. This enables lane keeping function. A more challenging task is to determine other lane & directions. Full automatic system need semantic understand of road structures and the ability to detect several lanes at long ranges. 

Road detection usually start with correction and filtering and find the road location. Then we separate roads & lanes by lane markings. Model fitting is performed next with geometric shapes so a continued lane is defined. Temporal integration is used next with info from vehicle dynamics.
# Assessment
## Risk Assessment
Bayesian method is used to quantify and measure uncertainties of deep neural networks, get risk of the whole system. Or we can assess the overall risk level of the driving scene separately
# References
[^1]: A Survey of Autonomous Driving: Common Practices and Emerging Technologies
[^2]: https://thegradient.pub/beyond-the-pixel-plane-sensing-and-learning-in-3d/