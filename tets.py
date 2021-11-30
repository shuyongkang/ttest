import numpy as np

data_org = np.arange(1,73,1 )
data_resh = np.reshape(data_org, [2, 3, 3, 4])

print(data_org)



data_resh1 = np.arange(1,28,1 )
for i in range(27):
    data_resh1[i] = 0



data_trans = np.transpose(data_resh,[1,2,3,0])
print(data_trans)





data2 = np.reshape(data_trans, [72])
print(data2)
print("-------------------------------------------")


for i in range(3):
    for j in range(3):
        for k in range(4):
            for p in range(2):
                value = data2[i*3*4*2 + j*4*2 + k*2 + p]
                print((i,j,k,p),np.argwhere(data_resh == value)[0], np.argwhere(data_org == value)[0],np.argwhere(data2 == value)[0])

       #找到他们的对应关系
