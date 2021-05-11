'''
This script will plot some of the assumed results from the folders in the same directory.
The first argument will determine whether the regular rewards (0, or no information given), or true rewards (1).
The second argument can stay empty and it will automatically print the healthy_Leanne folder results. Numbers ranging from 1 - 3 will display
the healthy_Leanne, _second and _third respectively. Adding a "t" as argv[2] expects an extra argv[3], being the number of Trial directory
you want to display.
All of them will be plotted against the healthy_Leanne_original directory, which contains the original data up to 80K episodes, displayed in blue.
'''
import matplotlib.pyplot as plt
import numpy as np
import sys

X, Y = [], []
X1, Y1 = [], []

model_file = ""
results = ""

orgcnt = 0

if len(sys.argv) == 2:
    if sys.argv[1] == "1":
        results = "true"
    model_file = "healthy_Leanne"
elif len(sys.argv) >= 3:
    if sys.argv[1] == "1":
        results = "true"
    if sys.argv[2] == "0":
        model_file = "healthy_Leanne_original"
    elif sys.argv[2] == "1":
        model_file = "healthy_Leanne"
    elif sys.argv[2] == "2":
        model_file = "healthy_Leanne_second"
    elif sys.argv[2] == "3":
        model_file = "healthy_Leanne_third"
    elif sys.argv[2] == "t":
        model_file = "Trial " + sys.argv[3]
    else:
        model_file = "healthy_Leanne"


for line in open(f"{model_file}/training_mean_{results}rewards.txt", 'r'):
#for line in open(f"healthy_Leanne{model_file}/training_mean_lengths.txt", 'r'):
    values = [s for s in line.split()]
    X.append(float(values[1]))
    Y.append(float(values[3]))
    orgcnt += 3

for line in open(f"healthy_Leanne_original/training_mean_{results}rewards.txt", 'r'):
#for line in open(f"Trial 1/training_mean_{results}rewards.txt", 'r'):
    values = [s for s in line.split()]
    X1.append(float(values[1]))
    Y1.append(float(values[3]))

model_file = model_file.replace("_", " ")
plt.title(f"{model_file} {results} reward received per episode")    
plt.xlabel('Episodes')
plt.ylabel(f"{results} reward")

# m is slope and b is intercept. [:904]
m, b = np.polyfit(X, Y, 1)
print("The slope is: ", m)
plt.plot(X1[:orgcnt], Y1[:orgcnt], c='b', label=f"original {results} reward")
plt.plot(X, Y, c='r', label=f"{results} reward")

plt.legend()

plt.show()
