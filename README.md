# PIRO

Proximal Inverse Reward Optimization

*Note*: If the environment is running for the first time (i.e., no expert data is present in the Folder _expert_data_), please uncomment **Line 84** and **Line 85** in *main.py*. This is for training and saving the expert policy model, and sampling and saving demonstrated trajectories.

PRIO:
>python main.py --env_name = Task Name

f-IRL
>python firl.py --env_name = Task Name

See *argument.py* for more adjustable parameters.