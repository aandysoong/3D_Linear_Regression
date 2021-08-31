# 3D_Linear_Regression
## Introduction
In my previous repository, I explained what linear regression is, and how to create a model in Python. ([make sure you know linear regression before you continue](https://github.com/aandysoong/Stock_Linear_Regression)) Now, we take a step further and into 3 dimensional linear regression, which is a specific type of multivariable linear regression. The model that I created uses a kaggle dataset about music, and I look into how Energy and Valence affects a song's popularity. But I will explain that later on. This project is actually pretty simple though if you know linear regression and my previous linear regression model. So, it is be relatively short.

## Multivariable Linear Regression
Multivariable linear regression is basically like linear regression but with more variables. Instead of using one x_train and one y_train sets to train our model, we use x1_train and x2_train. Then, the computer will find the best possible **w1, w2, and b** values for the equation: **y = w1x1 + w2x2 + b**.  So, we just need to change the "nn module" that we have been using into a 2 input, 1 output function system, make the datasets, and run the program. In this model, we will use "valence" as x1, "energy" as x2, and "popularity" as y. Afterwards, we can also plot the graph and the points, which will help us find out if the model seems correct.

## Actual Code

### 1. Data
The dataset we are going to use in this model is from [here]https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks, however, the page is now gone for some reason. Regardless, we have a dataset labeled "full_data.csv" which we can use. Here are the neccessary inputs:

```python
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
df=pd.read_csv('full_data.csv')
```

But there is a problem: the dataset we are using right now may be a bit too big for some computers. Although my computer was able to process it, it was still extremely lagging and slow. So, we will first create a system that can allow users to select the number of songs they want (I did all 30,000 rows of music but its your choice). The "sample" function allows us to randomly select and pick out how many rows we want from a dataframe.

```python
n=30000
df1=df.sample(n)
```

We first need to organize and sort out data in a way that the nn model can process. That is, put the valence column into x1_train, energy into x2, and popularity into y. Keep in mind that there will be no x1_train or x2_train, however, because the model we are using (nn) takes in one x_train set with two values for each element.

What I mean is:

x1 values = 1, 3, 5

x2 values = 2, 4, 6

x_train = [[1,2],[3.4],[5,6]]

y_train = [[7],[8],[9]]

```python
valence=df1['valence'] # pick out two specific columns
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
```

The nn module we will use should be 2 x's and 1 y. So it will look like:
```python
model = nn.Linear(2,1) # (2,1) = (x,y)
```
Now, all the data and models are set up.

### 2. Training
This is the part of the code where we just use the attained dataset and "feed" it to the computer. It will start the machine learning process (for more description of the concept and the code for it, make sure to look at my linear regression model).

```python
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
```

### Graphing
Now that training is over, we can use the obtained w1, w2, and b values to plot the graph and points. Before that, if you want to get a prediction about a certain value(s) of x, you can do this:

```python
new_var =  torch.FloatTensor([[80,75]]) 
pred_y = model(new_var) 
print("predicted value for [80,75] :", pred_y) 
```

Here, I applied [80,75] as my x1 and x2 but change it to fit your needs.

As for graphing, we obviously need a 3d graphing device since we have 3 variables. I used matplotlib 3d for this.
We will first get the w1, w2, and b values.

```python
w1_final = float(list(model.parameters())[0].data.numpy()[0][0])
w2_final = float(list(model.parameters())[0].data.numpy()[0][1])
b_final = float(list(model.parameters())[1].data.numpy()[0])
```
If you are confused with why we have "list(model.parameters())[0].data.numpy()[0][0]", it is because nn modules provide its parameters or the w and b values in tensor form. So we index to find the final value and change the number into a numpy float.

Let us now create a y plot list where we place all the predicted values through using the model.

```python
y_plot_list=[]
for i in x_train:
    y_plot_list.append(model(i))
```

To finish up graphing the predicted best fit line in 3d, we need to tell the program that this figure will be a 3d image and give it values which it will graph with.

```python
fig = plt.figure()
ax = plt.axes(projection='3d') # tell the computer that it is 3d

# 3d lines
xline = np.linspace(0,1,100)
yline = np.linspace(0,1,100)
zline = xline*w1_final + yline*w2_final + b_final
ax.plot3D(xline, yline, zline, 'blue')
```

Here, linspace is a function which returns you a certain amount of equally divided graphing spaces for a parameter. In our case, xline and yline is between 0 and 1 (valence and energy are measured between 0 and 1), and we will split it into 100 pieces.

In zline, we will use the attained w and b values with the x and y line to get the predicted values which the computer will use for graphing. The last line will make the computer use all three (x, y, and z lines) to plot a 3d graph.

Now that the line graph is done, let's plot all the points as well so that we can compare the points and line.

```python
x_plot=[]
y_plot=[]
z_plot=[]
s_color = []
for i in range(len(x_list)):
    x_plot.append(x_list[i][0])
    y_plot.append(x_list[i][1])
    z_plot.append(y_list[i])
    s_color.append(1)
```
This part looks complicated but it is actually simple. We just use the x and y list that we previously made and append the values to each plot list. s_color will be used later.

We now use plot lists in a scatterplot graphing function.

```python
s=ax.scatter3D(x_plot, y_plot, z_plot, c="r", s=s_color, alpha=0.05)
s.set_edgecolors = s.set_facecolors = lambda *args:None
```
c="r" is where we set the color of the points to Red.

s=s_color will help us adjust the transparency of the points graphed. (matplotlib automatically makes points darker and brighter to give them depth but we will get rid of that using s_color)

alpha adjusts the size of the points (in this case, we have need it smaller).

The next line of code is where we use s_color to get rid of transperancy in the points.

```python
ax.set_title("Valence, Energy, and Popularity")
ax.set_xlabel("Valence")
ax.set_ylabel("Energy")
ax.set_zlabel("Popularity")
plt.show()
```
These codes are for labeling and showing the graph.

### Final results (graphs)
The graph we receive looks like:

![valence,energy,pop](https://user-images.githubusercontent.com/70020467/131492440-16dc5434-e129-43e4-8bde-f3705e9849b4.jpg)

We can see that the program has worked. But since it is hard to analyze or judge it, let's see it from other angles. Here is a comparison between valence and popularity.

![valence_vs_pop](https://user-images.githubusercontent.com/70020467/131492708-7da0b8a2-6f9d-4643-b6fb-1b5112aedaea.jpg)
We see that the points and the line both head in the same direction, which means that our model is working. We also observe that as valence increases, popularity also increases.

Similar results come out with energy and popularity.
![energy_vs_popularity](https://user-images.githubusercontent.com/70020467/131492946-a70794d9-8b1b-4893-baca-037d0261a5c0.jpg)


In conclusion, as valence and energy increases, songs tend to be more popular.

This is the final code. Thank you for reading!

```python
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
print("predicted value for [80,75] :", pred_y) 

# data for plotting
w1_final = float(list(model.parameters())[0].data.numpy()[0][0])
w2_final = float(list(model.parameters())[0].data.numpy()[0][1])
b_final = float(list(model.parameters())[1].data.numpy()[0])

y_plot_list=[]
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
```
