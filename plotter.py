'''
This script is called automatically by the tic tac toe program, after completing
all required runs on a particular algorithm. Keep in mind that it does not know which
algorithm's results are being plotted!!!
To generate a graph with the right label, the algorithmName needs to be substituted for
the name of the algorithm that you are planning on getting results from.
This is designed to print 1000 runs of a single code execution and not the average of many.
'''
import matplotlib.pyplot as plt

X, Y = [], []
x_val = 0
for line in open("healthy_Leanne/training_mean_truerewards.txt", 'r'):
  values = [s for s in line.split()]
  #x_val+=float(values[1])
  X.append(float(values[1]))
  Y.append(float(values[3]))

plt.title("True reward received per episode")    
plt.xlabel('Episodes')
plt.ylabel('True reward')

plt.plot(X, Y, c='r', label='True reward')

plt.legend()

plt.show()
