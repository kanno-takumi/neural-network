class MulLayer:#かけ算　機能のみを切り離している→引数で持ってくる。
    def __init__(self): #順伝播の入力を保持するために用いる
        self.x = None
        self.y = None
        
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = int(x * y)
        
        
        return out
    
    def backward(self,dout):
        dx = dout * self.y #dout/dxという意味だと思う
        dy = dout * self.x #dout/dyという意味だと思う
        
        return dx ,dy #1つ前の偏微分を出す