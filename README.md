### LSTM-to-Classify-Sim-Paricles

Using AI ( now with the model of LSTM ) to separate simulated Trident particle events into two different types.

- This is my first AI model with Python. It is used to separate different particles passing through Trident detectors into two types, in which type 1 is what we need, muons.
- This is our [website](https://trident.sjtu.edu.cn/en). You can know more about Trident and us in this website.

- In brief, we can detect the arrival time and the intensity of different particles by detectors (PMTs) in strings. The current model can only predict infomations from one single string. Apart from muons, any other types of paricles are separated into type 0, which is not what we need.
