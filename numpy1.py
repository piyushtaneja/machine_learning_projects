import numpy as np  
a=np.array([1,2,3,4])
print(type(a)) 
# here normal array made using numpy

b=np.array([[1,2,3,4],[5,6,7,8]])
print(b.shape)

print(b.T)

c=np.random.randint(60,100,10)
print(c)

print(np.argmax(b))
d=np.array([2,4,6,4,2,3,4,5,6,4,2,7,6,6])
print(np.unique(d,return_counts=True))

e=np.array([[1,2,3,4,9],
           [5,6,7,8,55],
           [44,5,6,7,8],
           [33,2,5,6,44],
           [22,3,44,55,55]]) 
f=e[2:4,1:3]
print(f)
 
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=x**2 +3

print (x,y)

