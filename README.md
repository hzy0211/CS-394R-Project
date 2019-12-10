Adaptive Batching Using Advantage Actor-Critic
----------------------------------------------------

Python Requirements
===================
Requires python v 3.x, `numpy`, `torch`, and `seaborn` to be able to run everything.


Running
===================
Running `scheduling_a2c.py` will give an output of the plot comparing the performence of GAE Advatege Actor-Critic with REINFORCE and our two baselines (always batching and no batching) in a static environment, which is the Figure 2(b) in the writeup. Running `scheduling_a2c_v2.py` will give an output of the plot comparing GAE Advatege Actor-Critic with the two baselines in streaming environment with increasing number of users. 


Calling within Python
=====================
To call the model within python, just call `from scheduling_a2c import ac_gae`, where the `ac_gae` fucntion solves the scheduling problem with GAE Advatege Actor-Critic taking layer running time and new request sequence as input.
