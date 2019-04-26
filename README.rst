DQN Implementation of Multi-level Jenga Game With Internal Physical Env Simulation
====================

code written by Chen Liang.

To run the code, first install the requirements by running:

::

    pip3 install -r requirements.txt


To train and validate model and save the model in specified save directory

::

    python3 train.py --save-dir=[PATH/TO/SAVEDIR] --use-heuristics=[True/False] --use-dir-info=[True/False] --init-height=[5/10]


Reference
====================

[1] https://github.com/cliang1453/Tetris_RL_player