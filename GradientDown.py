#梯度下降法
#假设目标函数为z = x**2+y**2
#起始点为（2,2）
#学习率α为0.1
from sympy import *
from math import sqrt
def Dis(xlist,ylist):
    return sqrt((xlist[0]-ylist[0])**2+(xlist[1]-ylist[1])**2)
if __name__=='__main__':
    x, y ,z= symbols('x y z')
    alpha = 0.1
    z = x ** 2 + y ** 2
    dx = diff(z, x)
    dy = diff(z,y)
    x = 1
    y = 3
    sita = [x,y]
    k = 0
    while true:
        tx = x
        ty = y
        x -= alpha*eval(str(dx))
        y -= alpha*eval(str(dy))
        sita[0] = x
        sita[1] = y
        k+=1
        if Dis([tx,ty],sita)<1e-05:
            break
    print(sita,k)
    # x,y = 2,2
    # z = x**2+y**2
    # alpha = 0.1


