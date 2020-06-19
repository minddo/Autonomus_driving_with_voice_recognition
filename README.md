# Autonomus_driving_with_voice_recognition

In this project, you will learn how jetbots can drive autonomously and interact with a human voice for navigation.  JetBot can autonomously drive based on the trained model inspired by the ```Finding-path-in-maze-of-traffic-cones``` project., If it encounters an unexpected situation (e.g., intersection, traffic light), it couldn’t appropriately make navigational decisions. In this case, it requires human intervention; i.e., voice-based commands for manual navigation.

In this project, a [camera](https://www.raspberrypi.org/products/camera-module-v2/) and [speaker](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/) sensors were used. You can use a USB camera instead of a Raspberry PI camera. Also you can use any Bluetooth speaker

We used Convolutional Neural Network (CNN) to train the model. If you don't have a lot of knowledge about deep learning, we provide in-depth tutorials about transfer learning, and thus, we believe you can easily follow our tutorial. 


### Demo video
<a href="https://youtu.be/t8r-ahh4DBk
" target="_blank"><img src="https://ifh.cc/g/DQR3pv.jpg)" 
alt="IMAGE ALT TEXT HERE" width="400"  border="10" /></a>



### Objective
The objectives of this project are given as follow
1. deliver voice commands to the JetBot through voice recognition using [```Snowboy```](https://snowboy.kitt.ai/)
2. collect image data and train Jetbot to create autonomous driving model
3. create a Jetbot that processes voice commands during autonomous driving by combining 1 and 2

#### JetBot
 <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/embedded-systems/embedded-jetbot-ai-kits-seeed-2c50-D.jpg">
 
 [image source](https://www.nvidia.com/ko-kr/autonomous-machines/embedded-systems/jetbot-ai-robot-kit/)
 
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
In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.

If we want to transfer knowledge from one model to another, we want to reuse more of the generic layers (closer to the input) and fewer of the task-specific layers (closer to the output). In other words, we want to remove the last few layers (typically the fully connected layers) so that we can utilize the more generic ones, and add layers that are geared toward out specific classification task. One training begins, the generic layers (which form the majority of our new model) are kept frozen (i.e., they are unmodifiable). In contrast, the newly added task-specific layers are allowed to be modified. It is how transfer learning helps quickly train new models. 

<img src='https://ifh.cc/g/Pg2Vjf.png'>

### Overview
This project is a modified [```Collision avoidance```](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance) example from NVIDIA JetBot Wiki and [```Finding-path-in-maze-of-traffic-cones```](https://github.com/dvillevald/Finding-path-in-maze-of-traffic-cones/tree/master/traffic_cones_driving) from dvillevald. Unlike the previous project, we enabled Jetbot to intervene with human voice in urgent situations while autonomous driving. Also we made a path using **paper cups** instead of traffic cones or wall, and you can use any obstacles to make the path. It consists of four major steps, each described in a separate Jupyter notebook:

#### Step 0. Build and setup Jetbot
1. The first thing to do is assemble the JetBot. You must install a speaker sensor and camera.
2. For hardware setup, please refer the following [link](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/hardware-setup)
3. For software setup, please refer the following [link](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/software-setup)
4. You can write and run code with the Jupyter notebook via http://<jetbot_ip_address>:8888.


#### Step 1. Install Snowboy and make personal hotword model
To run the demo you will likely need the following, depending on which demo you use and what platform you are working with:
* SoX (audio conversion)
* PortAudio or PyAudio (audio capturing)
* SWIG 3.0.10 or above (compiling Snowboy for different languages/platforms)
* ATLAS or OpenBLAS (matrix computation)

1. Access Microphone: install PortAudio as a cross-platform support for audio in/out. We also use sox as a quick utility to check whether the microphone setup is correct.
2. Install SWIG and ATLAS: they are for compiling the Snowboy
3. Install Snowboy: Download libraries and demos provided by Snowboy.
4. Run a demo: test the basic example "Snowboy" using the universal model to make sure Snowboy is installed properly
5. Make personal hardword model: create personal hotword models to improve the performance of speech recognition. This is provided on the official Snowboy website.
6. **Remote operate jetbot via voice commands**: remote operate jetbot via voice commands using the personal hotword model created in 5.

#### Step 2. Collect a dataset for model training
We need to collect data to help the Jetbot find the way. You need to properly collect the images coming from the Jetbot's camera. The Jetbot encount 4 situations (i.e, free, blocked, left, and right). You will collect images corresponding to each situation. 
> We provide a [pre-trained model](https://www.dropbox.com/s/pt59zqrp1mjt39b/best_model_cones.pth?dl=0) so you can skip to step 4 if desired. (This model trained 200 images per class). This model was trained on a limited dataset using the Raspberry Pi V2 Camera.


1. Initialize the camera
2. Define a class for organizing a data set. It includes Directory where images are created, pre-processing methods, etc.
3. Collect data through the user interface provided by ipywidgets (100-200 images per class)
4. Compress the collected dataset into a zip file

#### Step 3. Training a deep learning through GPU provided by Colab
You can train a machine learning model with the collected dataset using the Jetson nano's GPU or other device's GPU. As a result of the test, when using the micro USB connector at 5V / 2A (10W), the robot was shut down during training because only 2 of the 4 CPU cores were used. On the other hand, Google Colab provides 12 GB of RAM, 68 GB of disk, and 11.4 GB of GPU, and it makes possible to build and train models stably. Therefore, we recommend using the free GPU provided by Google to quickly train the dataset. 

1. Upload zip file to Google Drive
2. Change runtime to GPU
3. Train the dataset and save the model
4. Download the model to your JetBot

#### Step 4. Live demo on JetBot
Now everything is ready. Run your jetbot smartly using the trained model and Snowboy. Our scenario is as follows.

<img src='https://ifh.cc/g/fmN9ru.gif' width='800'>

* First, autonomous driving is started through a voice command (wake up!).
* Until your voice command is given, robot will find the way automatically.
* If an unexpected situation (crossroads in our example) occurs, the autonomous driving is stopped through a voice command, then move the jetbot with voice according to the situation. After that, the autonomous driving is started again.
* When your robot arrives at its destination, your robot ends autonomous driving.




To achieve that, we prepared the following:

1. Load the model trained in step 3
2. Configure logic for autonomous driving: The logic refers to the demo code of [Finding-path-in-maze-of-traffic-cones](https://github.com/dvillevald/Finding-path-in-maze-of-traffic-cones/blob/master/traffic_cones_driving/live_demo_cones.ipynb) and consists of the following steps.
>    (1) Pre-process the camera image

>    (2) Execute the neural network

>    (3) If the output of the neural network is "free", the robot moves forward. The jetbot changes direction by comparing the probability of "left" and "right". If the Jetbot doesn't move forward for a long time, explore (turn right).
<img src='https://ifh.cc/g/onOTpZ.png' width='500'>

3. Prepare a personalized hotword model for voice commands (i.e., wakeup, stop, left, right)
4. Control robot via voice commands during autonomous driving through the conditional statement

<img src='https://ifh.cc/g/WkyNF3.png' width='500'>

### Lessons learned
* We not only let JetBot drive automatically through image processing, but also allow us to control it during autonomous driving through voice commands even in sudden situations (traffic lights or crossroads).
* We used various speech recognition libraries (e.g., [Pocketsphinx](https://pypi.org/project/pocketsphinx/), [google speech recognition](https://pypi.org/project/SpeechRecognition/2.1.3/)) before using Snowboy. However, in the case of Snowboy, the trained voice model was available, so Snowboy's voice recognition performance was the best. 
* Not surprisingly, the quality of the collected data was the most important factor in the performance of autonomous driving than the model used. For best performance, it is important to provide as much data as possible, in different situations (e.g., light, floor color, distance and angle).
* We did the same task using Keras, but we observed a Jetbot shutdown due to the memory limitations of the Jetson Nano. Various attempts seem to be necessary to overcome memory limitations. (Swap memory, Tensorflow Lite)

### Future Work
In our practice, in a sudden situation (traffic light, crossroads), the user first intervened and spoken by voice, but this is a very cumbersome and difficult task. It would be interesting to be able to apply the idea provided by [SelectiveNet](https://deepai.org/publication/selectivenet-a-deep-neural-network-with-an-integrated-reject-option) to the robot to recognize the sudden situation first and apply a query to the user. (Selective offers the 'I don't know' option if the risk is high when predicting a class)

