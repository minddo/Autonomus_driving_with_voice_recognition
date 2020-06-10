# Autonomus_driving_with_voice_recognition

This project enables JetBot to listen to and perform human voice commands while autonomous driving. The project used cameras and speaker sensors. If you don't have a lot of knowledge about deep learning, we believe you can easily follow


### Demo video

### Objective
The objectives of this project are as follow
1. deliver voice commands to the JetBot through voice recognition using ```snowboy```
2. collect and learn images through the JetBot to understand the context
3. execute the JetBot commands based on current context.

#### JetBot
 <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/embedded-systems/embedded-jetbot-ai-kits-seeed-2c50-D.jpg">
JetBot is an open-source robot based on NVIDIA Jetson Nano. Through JetBot, we learned to sense the surrounding environment (image, sound) and actuate (motion, i.e., forward, stop, left, right) based on sensing.

> To get started, read the [JetBot Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki) 

#### Voice recognition
<img src='https://snowboy.kitt.ai/3ee1353fe05ea13250318e7aa14f4a31.png' width='500'>
We used Snowboy for voice recognition. Snowboy is an highly customizable hotword detection engine that is embedded real-time and is always listening (even when off-line) compatible with Raspberry Pi, (Ubuntu) Linux, and Mac OS X. Here, A hotword (also known as wake word or trigger word) is a keyword or phrase that the computer constantly listens for as a signal to trigger other actions.

> For more information see the [Snowboy Wiki](https://github.com/kitt-ai/snowboy)

#### Deep learning
We used deep learning (CNN) to process the real-time image coming from the camera and determine the current context. 
We tried both PyTorch and TensorFlow, but we used PyTorch due to memory issues. If you have enough memory, you can use TensorFlow. 
We provided step-by-step from pre-processing to model evaluation so that you could easily understand it even if you were new to deep learning.
<img src='https://drive.google.com/uc?id=1s_gq1sL458tjgX0huKbCjGALfjM1Z9Bp' width='800'>

### Overview
This project is a modified ```Collision avoidance``` example from NVIDIA JetBot Wiki and ```Finding-path-in-maze-of-traffic-cones``` from dvillevald. It consists of four major steps, each described in a separate Jupyter notebook:

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
* Teleoperate jetbot via voice commands

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
* We used various speech recognition library(e.g., [Pocketsphinx](https://pypi.org/project/pocketsphinx/), [google speech recognition](https://pypi.org/project/SpeechRecognition/2.1.3/)) before using Snowboy. However, in the case of Snowboy, the trained voice model was available, so Snowboy's voice recognition performance was the best. 
* Not surprisingly, the quality of the collected data was the most important factor in the performance of autonomous driving than the model used. For best performance, it is important to provide as much data as possible, in different situations (e.g., light, floor color, distance and angle).
* We did the same task using Keras, but we observed a Jetbot shutdown due to the memory limitations of the Jetson Nano. Various attempts seem to be necessary. It seems to need a variety of attempts. (Swap memory, Tensorflow Lite)

### Future work
In our practice, in a sudden situation (traffic light, crossroads), the user first intervened and spoken by voice, but this is a very cumbersome and difficult task. It would be interesting to be able to apply the idea provided by [SelectiveNet](https://deepai.org/publication/selectivenet-a-deep-neural-network-with-an-integrated-reject-option) to the robot to recognize the sudden situation first and apply a query to the user. (Selective offers the 'I don't know' option if the risk is high when predicting a class)
<img src='https://images.deepai.org/converted-papers/1901.09192/arch_ne.png' width='500'>
