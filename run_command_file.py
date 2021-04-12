import time
import os
import subprocess

# Every 3 hours a new process is started
#max_seconds = 10800
max_seconds = 300
tstart = time.time()
last_timestep = 0
max_timesteps = 12000000

num_proc = 1

print("Please pick a model from:")
i = 0
models = [f for f in os.listdir("./osim-rl/osim/models/") if (os.path.isfile(os.path.join("./osim-rl/osim/models/", f)) and f[-5:] == ".osim")]
for model in models:
    print(f"[{i}] : {model}")
    i += 1

print("")
choice = int(input("Please choose your model: "))
if choice is 14:
    exp_model = 'Pendulum-v0'
else:
    exp_model = models[choice]

print(f"You have chosen:\n[{choice}] : {exp_model}")

try:
    os.mkdir(f"./{exp_model[:-5]}")
    os.mkdir(f"./{exp_model[:-5]}/models")
except FileExistsError as e:
    print("Folder to store data already exists, will overwrite contents of folder.\n")
print(exp_model[:-5])
#Change this if calling the altMain.py and adjust args accordingly
#args = ["mpirun", "-np", "4", "python", "main.py", "0", "0", f"../models/{exp_model}"]
#args = ["python", "main.py", "1", "1", f"../models/{exp_model}"]
#args = ["mpirun", "-np", "4", "python", "altMain.py", "0", "0", f"../models/{exp_model}"]
#No parallelisation
#args = ["python", "spinup/algos/pytorch/ppo/ppg_v0.0.py", "0", "0", f"../models/{exp_model}"]
#args = ["python", "algos/pytorch/ppo/ppg_v0.0.py", "0", "0", f"../models/{exp_model}"]
args = ["python", "spinup/algos/pytorch/ppo/ppg.py", "0", "0", f"../models/{exp_model}", f"{num_proc}"]
#args = ["mpirun", "-np", f"{num_proc}", "python", "algos/pytorch/ppo/ppg_v0.1.py", "0", "0", f"../models/{exp_model}", f"{num_proc}"]
#if choice is 14:
    #args = ["mpirun", "-np", f"{num_proc}", "python", "algos/pytorch/ppo/ppg_v0.1.py", "0", "0", f"{exp_model}", f"{num_proc}"]

proc = subprocess.Popen(args)

# Runs for max_timesteps, instead of running for set number of 'games'
while last_timestep < max_timesteps:

    time.sleep(10)
    if time.time() - tstart > max_seconds:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TIME TO RESTART!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #os.system("ps -aux |grep python |grep -v 'run_command_file.py' |awk '{print $2}' |xargs kill")
        subprocess.Popen.kill(proc)
        # Get iterations and timesteps from file
        with open(exp_model[:-5] + '/iterations.txt', 'r') as f:
        #with open(exp_model + '/iterations.txt', 'r') as f:
            lines = f.read().splitlines()
            # Get the last line as the last stored iteration
            last_iter = int(lines[-1])
        with open(exp_model[:-5] + '/timesteps.txt', 'r') as g:
        #with open(exp_model + '/timesteps.txt', 'r') as g:
            lines = g.read().splitlines()
            # Get the last line as the last stored time step
            last_timestep = int(lines[-1])

        tstart = time.time()
        #args = ["mpirun", "-np", "2", "python", "altMain.py", "1", "1", f"../models/{exp_model}"]
        #args = ["python", "main.py", "1", "1", f"../models/{exp_model}"]

        #args = ["mpirun", "-np", f"{num_proc}", "python", "algos/pytorch/ppo/ppg_v0.1", "1", "1", f"../models/{exp_model}", f"{num_proc}"]
        #args = ["python", "spinup/algos/pytorch/ppo/ppg_v0.0.py", "1", "1", f"../models/{exp_model}"]
        #args = ["python", "algos/pytorch/ppo/ppg_v0.0.py", "1", "1", f"../models/{exp_model}"]
        #args = ["python", "algos/pytorch/ddpg/ddpg.py", "1", "1"]
        args = ["python", "spinup/algos/pytorch/ppo/ppg.py", "1", "1", f"../models/{exp_model}", f"{num_proc}"]
        
        proc = subprocess.Popen(args)

subprocess.Popen.kill(proc)
