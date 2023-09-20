from MulLayer import MulLayer
from AddLayer import AddLayer

apple = 100 
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1


mul_apple_layer = MulLayer() #別々に呼び出す必要がある。initでforwardの処理を保管してbackwardで呼び出しているため
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer =MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(apple_orange_price,tax)
#print(price)

#backward
dprice = 1
dapple_orange_price , dtax = mul_tax_layer.backward(dprice) #forwardと順番が関係している
dapple_price,dorange_price = add_apple_orange_layer.backward(dapple_orange_price)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)
dorange,dorange_num = mul_orange_layer.backward(dorange_price)

print(price)
print(dapple_num,dapple,dorange,dorange_num,dtax)




