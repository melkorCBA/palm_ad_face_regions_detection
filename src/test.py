import numpy as np
a =np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,1,1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

a=a.reshape(6, 16)
newimage=a.copy()
print(a)
# v2 = a[1::3]
# print(v2)
# v2[0]=20
print(a.shape[0])
print(a.shape[1])
# # print(v2.base)
# print(a)
# print(v2)

v3=[a[i:i+2, j:j+2] for i in range(0,a.shape[0], 2) for j in range(0,a.shape[1],2)]
# print(v3)


score_image=np.zeros(int((a.shape[0]*a.shape[1])//(2*2)))
# print(a.shape[0])
# print(a.shape[1])
# print((a.shape[0]*a.shape[1])/(2*2))
for i, one in enumerate(v3):
    total=0
    for j in range(one.shape[0]):
        for k in range(one.shape[1]):
            total=total+one[j,k]
    score_image[i]=total
# print(score_image)

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      scoreIndex=((a.shape[1]//2))*(i//2)+(j//2)
      newimage[i][j]=score_image[scoreIndex]
# np.savetxt("foo.csv", newimage, delimiter=",")
print(newimage)

