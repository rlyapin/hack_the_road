# hack_the_road
Hackathon project to predict the intention of car drivers using the video feed the windshield camera

The implemented proof-of-concept focuses on predicting whether the car is in parking mode and involves:

1) The autoencoder model on video frame level to compress the initial video feed
2) The optical flow calculation to get an estimate of car's velocity
3) LSTM model on top of 5-10s videos t predict whether the car is parking
