import numpy as np

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.7,-0.5])
    b=1.0
    # calc=np.sum(w*x)+b
    calc=sum(x*w)+b
    if(calc>0):
        return 1
    else:
        return 0
    
print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))