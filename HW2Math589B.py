#!/usr/bin/env python
# coding: utf-8

# ## Problem 18.1

# In[ ]:


import numpy as np


# In[ ]:


# Function H_2
def h(x):
    first = np.exp(8*x[0][0] - 13*x[1][0] + 21)
    second = np.exp(21*x[1][0] - 13*x[0][0]-34)
    third = 0.0001*np.exp(x[0][0]+x[1][0])
    return first + second + third

# Gradient H_2
def gradh(x):
    g = np.array([[0.], [0.]])
    first = np.exp(8*x[0][0] - 13*x[1][0] + 21)
    second = np.exp(21*x[1][0] - 13*x[0][0]-34)
    third = 0.0001*np.exp(x[0][0]+x[1][0])
    g[0][0] = 8*first - 13*second + third
    g[1][0] = -13*first +21*second + third
    return g

# Hessian H_2
def hessh(x):
    h = np.array([[0., 0.], [0., 0.]])
    first = np.exp(8*x[0][0] - 13*x[1][0] + 21)
    second = np.exp(21*x[1][0] - 13*x[0][0]-34)
    third = 0.0001*np.exp(x[0][0]+x[1][0])
    h[0][0] = 64*first + 169*second + third
    h[1][1] = 169*first + 441*second + third
    h[0][1] = -104*first - 273*second + third
    h[1][0] = -104*first - 273*second + third
    return h
    


# In[ ]:


# Implementing Newton-Raphson method
# initialize using x = (1,2)
x = np.array([[1.], [2.]])
value = h(x)
n = 1
grad = gradh(x)
hess = hessh(x)
dt = 1.

while (np.linalg.norm(grad)> 1.e-15):
    print(x, value, np.linalg.norm(grad), dt, n)
    oldx = np.copy(x)
    oldh = np.copy(value)
    inver = np.linalg.inv(hess)
    x = x - dt*np.matmul(inver, grad)
    grad = gradh(x)
    hess = hessh(x)
    value = h(x)
    dt = 1.1*dt
    if value > oldh:
        dt = 0.5*dt
        x = oldx
        value = oldh
        grad = gradh(x)
        hess = hessh(x)
        
    n = n+1
    if n == 10000:
        print("The loop has ended early after 10000 iterations.")
        break


# In[ ]:


# Implementing Newton-Raphson method
# initialize using x = (1,2)
x = np.array([[1.], [2.]])
value = h(x)
n = 1
grad = gradh(x)
hess = hessh(x)
dt = 1.

while (np.linalg.norm(grad)> 1.e-15):
    print(x, value, np.linalg.norm(grad), dt, n)
    oldx = np.copy(x)
    oldh = np.copy(value)
    dt = 1.
    xnew = x-dt*np.linalg.solve(hess, grad)
    value = h(xnew)
    while value >= oldh:
        dt = dt/2.
        xnew = x - dt*np.linalg.solve(hess,grad)
        value = h(xnew)
    x = xnew
    grad = gradh(x)
    hess = hessh(x)
    value = h(x)
            
    n = n+1
    if n == 10000:
        print("The loop has ended early after 10000 iterations.")
        break


# # Problem 18.2

# In[ ]:


import numpy as np


# In[ ]:


def j(x):
    return 3*x[0][0]*x[1][0] - 2*x[1][0] + 1000*((x[0][0]**2) + (x[1][0]**2) - 1.1)*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)


def gradj(x):
    grad = np.array([[0.], [0.]])
    grad[0][0] = 3*x[1][0] +2000*(10*(x[0][0]**3) + 10*(x[0][0])*(x[1][0]**2) -10*x[0][0])*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    grad[1][0] = 3*x[0][0] - 2 +2000*(10*(x[1][0]**3) + 10*(x[1][0])*(x[0][0]**2) -10*x[1][0])*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    return grad

def hessj(x):
    hess = np.array([[0., 0.], [0., 0.]])
    hess[0][0] = 2000*(10*(x[1][0]**2) - 10 +200*(x[0][0]**4) + 200*(x[0][0]**2)*(x[1][0]**2) - 170*(x[0][0]**2))*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    hess[1][1] = 2000*(10*(x[0][0]**2) - 10 +200*(x[1][0]**4) + 200*(x[1][0]**2)*(x[0][0]**2) - 170*(x[1][0]**2))*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    hess[0][1] = 3 + 2000*(200*(x[0][0]**3)*x[1][0] + 200*(x[1][0]**3)*x[0][0] - 180*x[0][0]*x[1][0])*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    hess[1][0] = 3 + 2000*(200*(x[0][0]**3)*x[1][0] + 200*(x[1][0]**3)*x[0][0] - 180*x[0][0]*x[1][0])*np.exp(10*(x[0][0]**2) + 10*(x[1][0]**2) - 10)
    return hess


# In[ ]:


# Implementing Newton-Raphson method
# initialize using x = (1,2)
x = np.array([[1.], [1.]])
value = j(x)
n = 1
grad = gradj(x)
hess = hessj(x)
dt = 1.

while (np.linalg.norm(grad)> 1.e-6):
    print(x, value, np.linalg.norm(grad), dt, n)
    oldx = x
    oldj = value
    dt = 1.
    grad = gradj(x)
    hess = hessj(x)
    deltax = -np.linalg.solve(hess, grad)
    if (np.dot(np.transpose(deltax), grad) > 0):
        deltax = -deltax
    xnew = x + dt*deltax
    value = j(xnew)
    while value >= oldj:
        #print(dt, value, oldj)
        dt = dt/2.
        xnew = x+dt*deltax
        value = j(xnew)
        grad = gradj(x)
        hess = hessj(x)
    x = xnew
      
    n = n+1
    if n == 10000:
        print("The loop has ended early after 10000 iterations.")
        break

print(x, value, np.linalg.norm(grad), dt, n)


# # Problem 18.3

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def e(x):
    first = ((x[0][0] + 1)**2) + ((1 - x[98][0])**2)
    second = (1 - (x[98][0])**2)**2
    for i in range(0,98):
        first = first + ((x[i+1][0] - x[i][0])**2)
        second = second + ((1 - (x[i][0]**2))**2)
    return (first/2.) + (second/16.)

def grade(x):
    grad = np.copy(x)
    grad[0][0] = (7./4.)*x[0][0] + 1 - x[1][0] + (1./4.)*(x[0][0]**3)
    grad[98][0] = (7./4.)*x[98][0] - x[97][0] - 1 + (1./4.)*(x[98][0]**3)
    for i in range(1,98):
        grad[i][0] = (7./4.)*x[i][0] - x[i-1][0] - x[i+1][0] + (1./4.)*(x[i][0]**3)
    return grad

def hesse(x):
    hess = np.zeros([99, 99])
    hess[0][0] = (7./4.) + (3./4.)*(x[0][0]**2)
    hess[0][1] = -1.
    hess[98][98] = (7./4.) + (3./4.)*(x[98][0]**2)
    hess[98][97] = -1.
    for i in range(1, 98):
        hess[i][i] = (7./4.) + (3./4.)*(x[i][0]**2)
        hess[i][i-1] = -1.
        hess[i][i+1] = -1.
        
    return hess
    


# In[ ]:


# Implementing Newton's method
x = np.zeros([99,1])
for i in range(0, 51):
    x[i][0] = -1.
    
for i in range(51, 99):
    x[i][0] = 1

value = e(x)
n = 1
grad = grade(x)
hess = hesse(x)
dt = 1.

while (np.linalg.norm(grad)> 1.e-5):
    print(value, np.linalg.norm(grad), dt, n)
    oldx = x
    olde = value
    deltax = -np.linalg.solve(hess, grad)
    if (np.dot(np.transpose(deltax), grad) > 0):
        deltax = -deltax
    newx = x + dt*deltax
    value = e(newx)
    while (value >= olde):
        dt = dt/2.
        newx = x + dt*deltax
        value = e(newx)
    x = newx
    grad = grade(x)
    hess = hesse(x)
    
    n = n+1
    if n == 30000:
        print("The loop has ended early after 30000 iterations.")
        break

print(value)


# In[ ]:


x
plt.plot(x) 
plt.xlabel("Component")
plt.ylabel("Value")


# # Problem 18.4

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[10]:


def RHS(z):
    r = np.zeros([4, 1])
    x, y, u, v = z[0], z[1], z[2], z[3]
    r[0][0] = u
    r[1][0] = v
    r[2][0] = -np.sqrt((u**2) + (v**2))*u
    r[3][0] = -1. - np.sqrt((u**2) + (v**2))*v
    return r

def RK4(t, theta):
    T = t
    h = T/1000. # initial step size
    initial = np.array([[0.], [0.], [10*np.cos(theta)], [10*np.sin(theta)]])
    xmatrix = np.zeros([4, 1001])
    xmatrix[0:, 0] = np.transpose(initial)
    for i in range(0, 1000):
        dummy = np.zeros([4,1])
        dummy[0][0], dummy[1][0], dummy[2][0], dummy[3][0] = xmatrix[0][i], xmatrix[1][i], xmatrix[2][i], xmatrix[3][i]
        k1 = h*RHS(dummy)
        k2 = h*RHS(dummy + 0.5*k1)
        k3 = h*RHS(dummy + 0.5*k2)
        k4 = h*RHS(dummy + k3)
        dummy1 = dummy + (1/6.)*(k1 + 2*k2 + 2*k3 + k4)
        xmatrix[0][i+1], xmatrix[1][i+1], xmatrix[2][i+1], xmatrix[3][i+1] = dummy1[0][0], dummy1[1][0], dummy1[2][0], dummy1[3][0]
    
    return xmatrix

# howitzer function
def howitzer(theta):
    t = 30. # preliminary guess
    matrix = RK4(t, theta)
    n = 1
    upperbound = 100.
    lowerbound = 0.
    currentt = t
    #print(n, lowerbound, currentt, upperbound)
    while (abs(matrix[1, -1]) >= 1.e-10):
        if (matrix[1, -1] >= 1.e-10):
            lowerbound = currentt
            t2 = 0.5*(lowerbound + upperbound)
            matrix = RK4(t2, theta)
            currentt = t2
            if (matrix[1, -1] >= 0.):
                lowerbound = currentt
            else:
                upperbound = currentt
        elif (matrix[1, -1] <= -1.e-10):
            upperbound = currentt
            t2 = 0.5*(upperbound + lowerbound)
            matrix = RK4(t2, theta)
            currentt = t2
            if (matrix[1, -1] >= 0.):
                lowerbound = currentt
            else:
                upperbound = currentt
            
        n = n+1
        #print(n, lowerbound, upperbound, matrix[1, -1])
            
    #print("Maximum occurs at t = ", currentt)
    return matrix[0, -1]
            

def gradhow(x):
    h = np.finfo(float).eps
    h = (np.sqrt(h))
    return (howitzer(x + h) - howitzer(x))/(h)

def f(x):
    return -howitzer(x)

def gradf(x):
    return -gradhow(x)
    


# In[6]:


x = np.linspace(0, np.pi/2, 75)
y = np.linspace(0, np.pi/2, 75)
for i in range(75):
    y[i] = howitzer(x[i])

print(y)


# In[7]:


plt.plot(x,y)
print(x)
howitzer(np.pi/7)


# In[18]:


# applying BFGS
x = np.pi/7.004 # initialized value
C = 1.
fvalue = f(x)
grad = gradf(x)
n = 1
print(x, fvalue, abs(grad), n)
while (abs(grad) >= 1.e-6):
    n = n+1
    oldf = fvalue
    t = 1.
    deltax = -C*grad
    if (deltax > 0):
        deltax = -deltax
    d = t*deltax
    newx = x + d
    fvalue = f(newx)
    while (fvalue > oldf + 0.1*grad*d):
        t = t/2.
        d = t*deltax
        newx = x + d
        fvalue = f(newx)
    
    g = gradf(newx) - grad
    
    if (g == 0):
        break
        
    C = d/g
    x = newx
    grad = gradf(x)
    
    print(x, fvalue, abs(grad), n)
    
    if (n  == 1000):
        print("The loop has exceeded reached 1000 iterations.")
        break
        
    
print(x, fvalue, abs(grad), n)


# In[ ]:




