import time
import os
import subprocess

# Every 3 hours a new process is started
max_seconds = 10800
#max_seconds = 350
tstart = time.time()
last_timestep = 0
#Added a ton of timesteps
max_timesteps = 1200000000

num_proc = 1

'''d = {"one": [1, 2, 3, 4, 5], "two": [6, 7, 8, 9, 0]}
print(d["one"])
del d["one"][:3]
print(d["one"])
d["one"]+=d["two"]
print(d["one"])
exit()'''

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
    os.mkdir("../Results")
    print("Results folder NOT found! Creating one outside of this directory...\n")
except FileExistsError as e:
    print("Results folder found!\n")
try:
    os.mkdir(f"../Results/{exp_model[:-5]}")
    os.mkdir(f"../Results/{exp_model[:-5]}/models")
    os.mkdir(f"../Results/{exp_model[:-5]}/buffers")
except FileExistsError as e:
    print("Current model folder found! Will overwrite all data...\n")

load_dir = f"../Results/{exp_model[:-5]}"
args = ["mpirun", "-np", f"{num_proc}", "python", "main.py", "1", f"../models/{exp_model}", "0"]
if num_proc == 1:
    args = ["python", "main.py", "1", f"../models/{exp_model}", "1"]

proc = subprocess.Popen(args)

# Runs for max_timesteps, instead of running for set number of 'games'
while last_timestep < max_timesteps:

    time.sleep(10)
    if time.time() - tstart > max_seconds:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TIME TO RESTART!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        subprocess.Popen.kill(proc)
        # Get iterations and timesteps from file
        with open(load_dir + '/iterations.txt', 'r') as f:
        #with open(exp_model + '/iterations.txt', 'r') as f:
            lines = f.read().splitlines()
            # Get the last line as the last stored iteration
            last_iter = int(lines[-1])
        with open(load_dir + '/timesteps.txt', 'r') as g:
        #with open(exp_model + '/timesteps.txt', 'r') as g:
            lines = g.read().splitlines()
            # Get the last line as the last stored time step
            last_timestep = int(lines[-1])

        tstart = time.time()
        args = ["mpirun", "-np", f"{num_proc}", "python", "main.py", "1", f"../models/{exp_model}"]
        if num_proc == 1:
            args = ["python", "main_original.py", "1", f"../models/{exp_model}"]

        proc = subprocess.Popen(args)

subprocess.Popen.kill(proc)
