'''
JUST NEW DATASET
MUSIC 
HOW DOES VALENCE AND ENERGY AFFECT POPULARITY?
'''
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
df=pd.read_csv('full_data.csv')

n=30000 # how many points you want

df1=df.sample(n)

# all the neccessary inputs
torch.manual_seed(1)

# set up the model and the training set 
model = nn.Linear(2,1)
valence=df1['valence']
energy=df1['energy']
x1_list=[]
x2_list=[]
x_list=[]

for i in df1.index:
    x1_list.append(valence[i])
    x2_list.append(energy[i])
    x_list.append([valence[i],energy[i]])

x_train = torch.FloatTensor(x_list)

popularity=df1['popularity']
y_list=[]

for i in popularity:
    y_list.append([i])

y_train = torch.FloatTensor(y_list)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001) 

# actual learning process
nb_epochs = 100000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # finding the cost
    cost = F.mse_loss(prediction, y_train) 

    # reset the gradient
    optimizer.zero_grad()
    cost.backward()
    # update W and b
    optimizer.step()

    # print to show
    if epoch % 10000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        print(list(model.parameters()))

new_var =  torch.FloatTensor([[80,75]]) 
pred_y = model(new_var) 
print("predicted value for __ :", pred_y) 

# data for plotting
y_plot_list=[]
w1_final = float(list(model.parameters())[0].data.numpy()[0][0])
w2_final = float(list(model.parameters())[0].data.numpy()[0][1])
b_final = float(list(model.parameters())[1].data.numpy()[0])

print(w1_final)
print(w2_final)
print(b_final)

for i in x_train:
    y_plot_list.append(model(i))

fig = plt.figure()
ax = plt.axes(projection='3d')

# 3d lines
xline = np.linspace(0,1,100)
yline = np.linspace(0,1,100)
zline = xline*w1_final + yline*w2_final + b_final
ax.plot3D(xline, yline, zline, 'blue')

x_plot=[]
y_plot=[]
z_plot=[]
s_color = []
for i in range(len(x_list)):
    x_plot.append(x_list[i][0])
    y_plot.append(x_list[i][1])
    z_plot.append(y_list[i])
    s_color.append(1)

# 3d points
s=ax.scatter3D(x_plot, y_plot, z_plot, c="r", s=s_color, alpha=0.05)
s.set_edgecolors = s.set_facecolors = lambda *args:None

ax.set_title("Valence, Energy, and Popularity")
ax.set_xlabel("Valence")
ax.set_ylabel("Energy")
ax.set_zlabel("Popularity")
plt.show()