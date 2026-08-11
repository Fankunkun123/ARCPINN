import matplotlib.pyplot as plt
# import PointWithinPoly as pwp
import numpy as np
import pandas as pd
# import Point_in_polygen_2 as pwp2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
# from scipy.integrate import quad
import math as m
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pi = np.pi
xmin = -2
xmax = 2
numb = 500

omega1 = 7
omega2 = 5

uf_x = [];   uf_y = []
ub_x = [];   ub_y = []
for x in np.linspace(xmin, xmax, numb + 1, endpoint=True):
    if x == xmin or x == xmax:
        y = 0.5 * np.sin(omega1 * pi * x) * np.cos(omega2 * pi * x) + np.tanh(20 * x)
        ub_x.append(x)
        ub_y.append(y)
    else:
        y = 0.5 * np.sin(omega1 * pi * x) * np.cos(omega2 * pi * x) + np.tanh(20 * x)
        uf_x.append(x)
        uf_y.append(y)

ub_x = np.array(ub_x);  ub_y = np.array(ub_y)
uf_x = np.array(uf_x);  uf_y = np.array(uf_y)

print('ub_x=:', ub_x)
print('ub_y=:', ub_y)
print('uf_x=:', uf_x)
print('uf_y=:', uf_y)

X, Y = np.hstack((ub_x, uf_x)), np.hstack((ub_y, uf_y))  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y)

np.savez(f'./数据+结果/数据/w={omega1,omega2} 训练点={numb}_shuju.npz', ub_x, ub_y,uf_x, uf_y,X, Y)

data = np.load(f'./数据+结果/数据/w={omega1,omega2} 训练点={numb}_shuju.npz')

ub_x = data['arr_0']
ub_y = data['arr_1']

uf_x = data['arr_2']
uf_y = data['arr_3']

x_total = data['arr_4']
y_total = data['arr_5']

print('ub_x:',ub_x)
print('uf_x:',uf_x)
print('x_total=:',x_total)
print('y_total=:',y_total)


fig = plt.figure()

# 对x和y进行排序
sorted_points = sorted(zip(x_total, y_total))
x_total_sorted, y_total_sorted = zip(*sorted_points)

plt.plot(x_total_sorted, y_total_sorted, 'g', label='yf')
plt.xlabel('X', fontsize=14)
plt.ylabel('U', fontsize=14)
plt.savefig(f'./数据+结果/数据/w={omega1,omega2} 训练点={numb}_精确解.png', dpi=1000)


plt.figure()
plt.scatter(x_total,y_total,s=10)
# plt.savefig('F:/PINN/神经网络学习/代码/代码/一维方程/泊松方程/数据+结果/结果/PINN+MR(u5)/w1_15 w2_9_精确解撒点.png', dpi=1000)
plt.show()