import cv2
import tensorly as tl
import numpy as np
from numba import jit
from line_profiler import LineProfiler

'''
矩阵的shrinkage
'''
def shrinkage(X, t):
    # svd矩阵的SVD分解，其中S返回的形式为向量            
    U, S, Vh = np.linalg.svd(X,full_matrices=False)
    sigm = np.zeros((U.shape[1], Vh.shape[0]))
    for i in range(len(S)):
        sigm[i,i] = np.max(S[i]-t, 0)
    temp = np.dot(U, sigm)
    shrinkageX = np.dot(temp, Vh)
    return shrinkageX

# def shrinkage(X, t):
#     U, Sig, VT = np.linalg.svd(X,full_matrices=False)

#     Temp = np.zeros((U.shape[1], VT.shape[0]))
#     for i in range(len(Sig)):
#         Temp[i, i] = Sig[i]  
#     Sig = Temp

#     Sigt = Sig
#     imSize = Sigt.shape

#     for i in range(imSize[0]):
#         Sigt[i, i] = np.max(Sigt[i, i] - t, 0)

#     temp = np.dot(U, Sigt)
#     T = np.dot(temp, VT)
#     return T

'''
图片的读取，返回值为三维数组即张量
'''
def readImg(path): 
    # 读取图片 （TODO 方法的第二个参数填写）
    return cv2.imread(path)

'''
缺失部分图像数据
'''    
def missImg():
    path = "G:\python-code\seaside.jpg"
    X = readImg(path)  # 原始图像
    imgSize = X.shape 
    size = np.prod([imgSize[0],imgSize[1],imgSize[2]]) # 图片总的元素数据
    missDataSize = int(np.ceil(np.prod([size,0.3])))  # 缺失元素数量 缺失30%的数据
    nums = np.ones(size) # 生成全为1的数组
    nums[:missDataSize ] = 0  # 缺失的数据填充为0
    np.random.shuffle(nums)  # 对只含0,1的数组进行乱序排列
    T = tl.tensor(nums.reshape(imgSize)) # 生成只含有0,1的张量
    missDataTensor = T * X
    return T, missDataTensor

'''
生成一个元素全为0的张量
tShape:张量的大小,可以通过shape方法获取
'''
def createZeroTensor(tShape):
    return tl.zeros(tShape)

'''
K:最大迭代次数
a:核范数前的系数为一个数组
X:缺失部分数据的张量图片
Z:0-1张量
rho:罚参数
Y:初始时为0张量
'''
def HaLRTC(K,a,X,Z,rho,Y):
    Y1=Y2=Y3=Y
    print("Y is:===================================================")
    print(Y)
    for k in range(K):
        print('iteration number is:{num}'.format(num=k+1))
        # 1.更新Mi
        M1= tl.fold(shrinkage(tl.unfold(X, mode=0) + tl.unfold(Y1, mode=0) / rho, a[0] / rho), 0, X.shape)  
        M2= tl.fold(shrinkage(tl.unfold(X, mode=1) + tl.unfold(Y2, mode=1) / rho, a[1] / rho), 1, X.shape) 
        M3= tl.fold(shrinkage(tl.unfold(X, mode=2) + tl.unfold(Y3, mode=2) / rho, a[2] / rho), 2, X.shape) 
        # 2.更新X
        X_hat = (1-Z)*(M1+M2+M3-(Y1+Y2+Y3)/rho)/3+X
        # 3.更新Lagrange乘子
        Y1 = Y1-rho*(M1-X_hat)
        Y2 = Y2-rho*(M2-X_hat)
        Y3 = Y3-rho*(M3-X_hat)
        print('the {num} times iteration ending'.format(num=k+1))
    return X_hat

if __name__=="__main__":
    # X = missImg()
    # tShape = X.shape
    # #print(tShape)
    # size = np.prod([tShape[0],tShape[1],tShape[2]]) 
    # zeroArray = np.zeros(size) # 生成全为0的数组
    # T = tl.tensor(zeroArray.reshape(imgSize))
    # print("tensor ================================")
    # print(T)
    path = "G:\python-code\seaside.jpg"
    # 原始图片
    originl_X = readImg(path)
    # X为受损的图片
    Z, X = missImg()
    print("zero-one tensor:==================================================")
    print(Z)
    imgShape = X.shape
   # 对应alpha
    a = abs(np.random.rand(3, 1))
    a = a / np.sum(a)
    K = 5 # 迭代次数
   #print(Z)
    rho = 1e-6
    Y = createZeroTensor(X.shape)
    # X_hat为通过算法还原的图片
    X_hat = HaLRTC(K,a,X,Z,rho,Y)
    # 显示对比图
    im1 = cv2.resize(X, (400, 400))
    im2 = cv2.resize(X_hat, (400, 400))
    im3 = cv2.resize(originl_X, (400, 400))
    imgs =  np.hstack((im1,im2,im3))
    # # 展示多个
    # cv2.imshow("mutil_pic", imgs)
    #hmerge = np.hstack((im1, im2,im3)) #水平拼接
    # vmerge = np.vstack((im1, im2)) #垂直拼接

    cv2.imshow("compare_img", imgs)
    # cv2.imshow("test2", vmerge)
    #cv2.imshow("mutil_pic", imgs)
    cv2.waitKey(0) 
    # cv2.namedWindow('HaLRTC', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow("HaLRTC", X_hat.astype(np.uint8))
    # # 如果不加上这句，图片的显示没有停留，可能看不到图片显示
    # cv2.waitKey()


    # print('start compute....')
    # M1= tl.fold(shrinkage(tl.unfold(X, mode=0) + tl.unfold(Z, mode=0) / rho, a[0] / rho), 0, X.shape)  
    # print("M1 is====================================")
    # print(M1)
    # M2= tl.fold(shrinkage(tl.unfold(X, mode=1) + tl.unfold(Z, mode=1) / rho, a[1] / rho), 1, X.shape) 
    # print("M2 is====================================")
    # print(M2)
    # M3= tl.fold(shrinkage(tl.unfold(X, mode=2) + tl.unfold(Z, mode=2) / rho, a[2] / rho), 2, X.shape) 
    # print("M3 is====================================")
    # print(M3)


   

    
    
