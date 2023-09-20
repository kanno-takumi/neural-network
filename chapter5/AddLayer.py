class AddLayer:
    def __init__(self):
        pass #特に何も行わないと言う処理
    
    def forward(self,x,y):
        out = int(x + y)
        return out
    
    def backward(self,dout):#doutは微分
        dx = dout * 1  #あえて1をかけているのはx'=1のため
        dy = dout * 1
        return dx,dy