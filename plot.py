
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(dpi=300)


X = np.array([5,10,20,50,100])
Y = np.array([5,10,20]) 

Z = np.array([[0.9999,0.9997,0.9978,0.9839,0.9524],
              [0.9998,0.9985,0.9940,0.9707,0.8242],
              [0.9998,0.9976,0.9798,0.9641,0.7940],
              ])
plt.plot(X,Z[0],'r*-',label="alphabet_length=5")
plt.plot(X,Z[1],'go-',label="alphabet_length=10")
plt.plot(X,Z[2],'b^-',label="alphabet_length=20")
#plt.plot(X,Z[3],'yx-',label="alphabet_length=100")

#ax.set_xticks([5,10,20,50])
#ax.set_yticks([5,10,20])

plt.xlabel('seq_length')
plt.ylabel('accuracy')
plt.legend(loc = 'upper right',bbox_to_anchor = (1,1),prop={'size': 6})
plt.show()
#plt.savefig("plot.png",dpi=300)

