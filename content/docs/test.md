+++
title = 'Autonomous Driving'
date = 2023-12-23T14:12:05+08:00
draft = false
tags = ["tech"]
categories = ["tutorial"]
+++


# Autonomous Driving Review
## Introduction
Automated Driving Systems (ADSs) aims to prevent traffic accidents and mitigate congestion [^1]. ADS is empowered by recent development of Deep Learning and sensor modalities (such as lidar).DARPA Grand Challenge is the first major competition in this field where human interation is prohibited. However the environment is relatively simple.  

Society of Automotive Engineers (SAE) defined five levels of driving automation from L0 to L5. L1 include simple tasks such as adaptive cruise control. L2 is partial automation such as collision avoidance. L3 is conditional automation where the driver is require quick respons to any emergencey. L4 automation does not need any human interation but only possible in certain environment or require high precision map. L5 is capable in all environemnt.

ADS is challenged by the balance between more automation and safety. Also, ethical dilemmas are hard for machine to solve. Like other transpotation, ADS needs a rigorous safety standard but related standards are yet to be established. Sharing thousands info between vehicles is hard so we need to convert raw sensors data into a common framework.
## Architecture
ADS can be classified into **ego only** or **connected system**.
### Ego Only
Perform operations in the single vehicle. Most popular method.
### Connected System
No operational connected ADS in use yet but might be the future. Vehicular Ad hoc NETwork (**VANETs**) distribute task into differne agents. **V2X** is a term that stands for 'vehicle to everything'. Vehicles are able to access information from peers and traffic signals. It solve some drawbacks of ego-only system that such as limited computation or sensing abilities.
### Modular System
Core functions of a modular ADS can be summarized as: localization and mapping, perception, assessment, planning and decision making, vehicle control, and human-machine interface. Breakdown the hard problem into pieces. Good: Easy to add functions. Bad: error propaganda and complexity.
### End-to-End Driving
Generate ego-motion(direct control such as turning) from only sensor inputs. It consists of direct supervised deep learning,  neuroevolution and the deep reinforcement learning. 
## Sensors and Hardare
### Mono Camera
A kind of **passive sensor**. Simple and direct. Influenced by illumination and no depth perception. 
### Omnidirectional Camera
Panoramic view
### Event Camera
Each pixel record change of brightness individually. High response, dynamic range low power.
### Radar
Active sensors, might interfere with other active sensors. Less accurate
### Lidar
Light wave other than radio wave. High accuracy but influenced by weather. large size. Outperform human in terms of sensing in dark environment.
### Proprioceptive Sensors
Proprioceptive sensors measure status of the car itself (speed,acceleration, yaw). Include wheel encoder, IMU.
## Localization and Mapping
## References
[^1]: A Survey of Autonomous Driving: Common Practices and Emerging Technologies