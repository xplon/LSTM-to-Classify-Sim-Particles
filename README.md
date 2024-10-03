### LSTM-to-Classify-Sim-Particles

Using AI ( now with the model of LSTM ) to separate simulated Trident particle events into two different types.

- This is my first AI model with Python. It is used to separate different particles passing through Trident detectors into two types, in which type 1 is what we need, muons.
- This is our [website](https://trident.sjtu.edu.cn/en). You can know more about Trident and us in this website.

- In brief, we can detect the arrival time and the intensity of different particles by detectors (PMTs) in strings. The current model can only predict infomations from one single string. Apart from muons, any other types of paricles are separated into type 0, which is not what we need.

Here is how I get these two pictures. I firstly using [data](https://github.com/wlhwl/siMu_atm/blob/topic_1string_reco_BDT/detector_sim/ana_string/reco_one_string/data/mc_event.parquet) to change them into events on one single string, and caculate the amount of signals happened in one Dom ( there are 21 Doms in one string ), and then record the time ( relative ) when the first signal arrived at the Dom. Then I cut all strange events ( Type 1 ) to avoid training the ai model with some weird data. Next I separate the data into training set and testing set ( firstly my training set uses all Type 0 events which are much more than Type 1 events, then it get predictions in [pre1](https://github.com/xplon/LSTM-to-Classify-Sim-Particles/blob/main/pic/pre1.png). So I cut the amount of Type 0 events in my training set to the length of Type 1 events, which finally get [pre2](https://github.com/xplon/LSTM-to-Classify-Sim-Particles/blob/main/pic/pre1.png) ). Finally I changed data into array format, adding 0 to the columns of amounts about those Dom which didn't detect any amount of signals and NaN to the time of these Dom. 
Here I get all the data prepared. I now use LSTM to use these data to predict the type of all the particles, with the [post-processing](https://github.com/xplon/LSTM-to-Classify-Sim-Particles/blob/main/analyze_type1.py), I get the pictures here.
