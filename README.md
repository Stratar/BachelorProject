<<<<<<< HEAD
Code for the implementation of Bachelor Project: Testing for generality of a Phasic Policy Gradient (PPG) for learning human-like locomotion in advanced environments.

The code is largely inspired by the 7th-placed solution to the 2018 NIPS AI For Prosthetics competition: http://osim-rl.stanford.edu/docs/nips2018/. OpenAI's Baselines implementation of PPO serves as the basis for the learning algorithm. To speed up training, state trajectories at different walking speeds are included in the osim-rl/osim/data folder. The PPG algorithm is an extension on the previous PPO implementation, extending OpenAI's spinningup version to add the auxiliary update characteristic of PPG

Installing the environment:
1. Ensure you have Anaconda installed on your Linux machine. 
1. Clone this repository to a location of your choice.
1. Create a conda evirontment by running the opensim.yml[1] file with conda: `conda env create -f opensim.yml`
1. Activate your environemt: `conda activate opensim-ppg`
1. Install the dependecies from here: `pip install -e .`
1. Change location to osim-rl folder.
1. Install the dependecies from here: `pip install -e .` 
1. Return back to the root folder.
1. Run the code with: `python3 run_command_file.py`[2] (select the model you want to train

If you get errors such as "import from baseline is missing" or "import from osim is missing" then you should repeat steps 5 to 8. It is also safe to ignore the error regarding MuJoCo when of step 6, this simualtionm does not make use of that.

[1]note; if you want to name your environemt something other than "opensim-rl", then open opensim.yml with any text editor and change to very first line to your desired name.

[2]note; Please check if you are running Python verion 3 or above. Check with `python --version` or `python3 --version`

This simulation is based on two models: a healthy model, and a transfemoral model. Depending on the model you are trying, please ensure that the function at line 570 of osim-rl/osim/env/osim.py is returning the correct value.
=======
# legTests
>>>>>>> 9b1a47323431e1b89b2401b3180af78d5633f16c
