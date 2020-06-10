# Autonomus_driving_with_voice_recognition

This project enables JetBot to listen to and perform human voice commands while autonomous driving. The project used cameras and speaker sensors. If you don't have a lot of knowledge about deep learning, we believe you can easily follow


### Demo video

### Objective
The objectives of this project are as follow
1. deliver voice commands to the JetBot through voice recognition using ```snowboy```
2. collect and learn images through the JetBot to understand the context
3. execute the JetBot commands based on current context.

#### JetBot
 <img src='https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/embedded-systems/embedded-jetbot-ai-kits-seeed-2c50-D.jpg'>
JetBot is an open-source robot based on NVIDIA Jetson Nano. Through JetBot, we learned to sense the surrounding environment (image, sound) and actuate (motion, i.e., forward, stop, left, right) based on sensing.

To get started, read the [JetBot Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki) 

#### Voice recognition
<img src='https://snowboy.kitt.ai/3ee1353fe05ea13250318e7aa14f4a31.png' width='500'>
We used **Snowboy** for voice recognition. **Snowboy** is an highly customizable **hotword** detection engine that is embedded real-time and is always listening (even when off-line) compatible with Raspberry Pi, (Ubuntu) Linux, and Mac OS X. Here, A **hotword** (also known as wake word or trigger word) is a keyword or phrase that the computer constantly listens for as a signal to trigger other actions.

For more information see the [Snowboy Wiki](https://github.com/kitt-ai/snowboy)

#### Deep learning
We used deep learning (CNN) to process the real-time image coming from the camera and determine the current context. 
We tried both PyTorch and TensorFlow, but we used PyTorch due to memory issues. If you have enough memory, you can use TensorFlow. 
We provided step-by-step from pre-processing to model evaluation so that you could easily understand it even if you were new to deep learning.

### Overview

### Lessons learned
