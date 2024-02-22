import matplotlib.pyplot as plt
import numpy as np
mutable_object = {} 
fig = plt.figure()
def onclick(event):
    print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    mutable_object['click'] = X_coordinate

cid = fig.canvas.mpl_connect('button_press_event', onclick)
lines, = plt.plot([1,2,3])
plt.show()
X_coordinate = mutable_object['click']
print(X_coordinate)
