# Autonomus_driving_with_voice_recognition

In this project, you will learn how jetbots can drive autonomously and interact with a human voice for navigation.  JetBot can autonomously drive based on the trained model (refer to X’s project), If it encounters an unexpected situation (e.g., intersection, traffic light), it couldn’t appropriately make navigational decisions. In this case, it requires human intervention; i.e., voice-based commands for manual navigation.

In this project, a [camera](https://www.raspberrypi.org/products/camera-module-v2/) and [speaker](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/) sensors were used. You can use a USB camera instead of a Raspberry PI camera. Also you can use any Bluetooth speaker

We used Convolutional Neural Network (CNN) to train the model. If you don't have a lot of knowledge about deep learning, we provide in-depth tutorials about transfer learning, and thus, we believe you can easily follow our tutorial. 


### Demo video
<a href="https://youtu.be/t8r-ahh4DBk
" target="_blank"><img src="https://ifh.cc/g/DQR3pv.jpg)" 
alt="IMAGE ALT TEXT HERE" width="400"  border="10" /></a>



### Objective
The objectives of this project are given as follow
1. deliver voice commands to the JetBot through voice recognition using [```snowboy```](https://snowboy.kitt.ai/)
2. collect image data and train jetbots to create autonomous driving model
3. create a jetbot hat processes voice commands during autonomous driving by combining 1 and 2

#### JetBot
 <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/embedded-systems/embedded-jetbot-ai-kits-seeed-2c50-D.jpg">
JetBot is an open-source robot based on NVIDIA Jetson Nano. Through JetBot, we learned to sense the surrounding environment (image, sound) and actuate (motion, i.e., forward, stop, left, right) based on sensing.

> To get started, read the [JetBot Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki) 

#### Voice recognition
<img src='https://snowboy.kitt.ai/3ee1353fe05ea13250318e7aa14f4a31.png' width='500'>
We used Snowboy for voice recognition. Snowboy is a highly customizable hotword detection engine that is embedded real-time and is always listening (even when off-line) compatible with Raspberry Pi, (Ubuntu) Linux, and Mac OS X. Here, a hotword (also known as wake word or trigger word) is a keyword or phrase that the computer constantly listens for as a signal to trigger other actions.

> For more information see the [Snowboy Wiki](https://github.com/kitt-ai/snowboy)

#### Deep learning
We used deep learning (CNN) to process the real-time image coming from the camera and determine the current context. 
We tried both PyTorch and TensorFlow, but we used PyTorch due to memory issues. If you have enough memory, you can use TensorFlow. 
We provided step-by-step from pre-processing to model evaluation so that you could easily understand it even if you were new to deep learning. Here, we used Convolutional Neural Network (CNN) and transfer learning.

##### Convolution Neural Network (CNN)
A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

<img src='https://ifh.cc/g/44uMg3.png'>
CNN consis of two parts: (1) convolutional layers, (2) fully connected layers
* Convolutional layers take the large number of pixels of an image and convert them into a much smaller representation (feature extractors)
* Fully connected layers convert features into probailities (classifiers)

##### Transfer learning


### Overview
This project is a modified [```Collision avoidance```](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance) example from NVIDIA JetBot Wiki and [```Finding-path-in-maze-of-traffic-cones```](https://github.com/dvillevald/Finding-path-in-maze-of-traffic-cones/tree/master/traffic_cones_driving) from dvillevald. It consists of four major steps, each described in a separate Jupyter notebook:

#### Step 0. Build and setup Jetbot
* The first thing to do is assemble the JetBot. You must install a speaker sensor and camera.
* For hardware setup, please refere the following [link](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/hardware-setup)
* For sotfware setup, please refere the following [link](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/software-setup)
* You can write and run code with the Jupyter notebook via http://<jetbot_ip_address>:8888.


#### Step 1. Install snowboy and make personal hotword model
* Access Microphone: Install PortAudio as a cross-platform support for audio in/out. We also use sox as a quick utility to check whether the microphone setup is correctly.
* Install Snowboy
* Run a demo
* Make personal hardword model
* **Teleoperate jetbot via voice commands**

#### Step 2. Collect dataset
We need to collect data to help the Jetbot find the way. The Jetbot encount 4 situations (i.e, free, blocked, left, and right). You will collect images corresponding to each situation.
* Collect data through the user interface provided by ipywidgets (100-200 images per class)
* Compress the collected dataset into a zip file

#### Step 3. Training dataset through GPU provided by Colab
You can train the dataset using the Jetson nano's GPU, but we have found that the training was slow due to limited performance. Therefore, we recommend using the free GPU provided by Google to quickly training the dataset.

* Upload zip file to Google Drive
* Change runtime to GPU
* Train the dataset and save the model
* Download the model to your JetBot

#### Step 4. Live demo on JetBot
Now everything is ready. Run your jetbot smartly using the trained model and snowboy
* Load the trained model
* Create logic to control the robot

### Lessons learned
* We not only let JetBot drive automatically through image processing, but also allow us to control it during autonomous driving through voice commands even in sudden situations (traffic lights or crossroads).
* We used various speech recognition library (e.g., [Pocketsphinx](https://pypi.org/project/pocketsphinx/), [google speech recognition](https://pypi.org/project/SpeechRecognition/2.1.3/)) before using Snowboy. However, in the case of Snowboy, the trained voice model was available, so Snowboy's voice recognition performance was the best. 
* Not surprisingly, the quality of the collected data was the most important factor in the performance of autonomous driving than the model used. For best performance, it is important to provide as much data as possible, in different situations (e.g., light, floor color, distance and angle).
* We did the same task using Keras, but we observed a Jetbot shutdown due to the memory limitations of the Jetson Nano. Various attempts seem to be necessary to overcome memory limitations. (Swap memory, Tensorflow Lite)

### Future Work
In our practice, in a sudden situation (traffic light, crossroads), the user first intervened and spoken by voice, but this is a very cumbersome and difficult task. It would be interesting to be able to apply the idea provided by [SelectiveNet](https://deepai.org/publication/selectivenet-a-deep-neural-network-with-an-integrated-reject-option) to the robot to recognize the sudden situation first and apply a query to the user. (Selective offers the 'I don't know' option if the risk is high when predicting a class)

<img src='https://images.deepai.org/converted-papers/1901.09192/arch_ne.png' width='500'>
