import cv2
import tensorly as tl
import numpy as np
import datetime
from numba import jit
from line_profiler import LineProfiler

def shrinkage(matrix, t):
    """矩阵的shrinkage运算
    Args:
        matrix: 进行shrinkage运算的矩阵
        t: 

    Returns:
        shrinkageMatrix: 进行shrinkage运算以后的矩阵
    """         
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    sigm = np.zeros((U.shape[1], Vh.shape[0]))
    for i in range(len(S)):
        sigm[i,i] = np.max(S[i]-t, 0)
    temp = np.dot(U, sigm)
    shrinkageMatrix = np.dot(temp, Vh)
    return shrinkageMatrix

def readImg(img_path): 
    """对图片的数据进行部分缺失处理
    Args:
        img_path: 图片路径

    Returns:
        img_tensor_data: 图片的张量形式数据
    """
    img_tensor_data = cv2.imread(img_path)
    return img_tensor_data

def missImg(img_path, miss_percent):
    """对图片的数据进行部分缺失处理
    Args:
        img_path: 图片路径
        miss_percent: 图片缺失数据百分比，取值为[0,1]

    Returns:
        sparse_tensor: 稀疏张量，只含有元素0和1
        miss_data_tensor: 缺失部分数据图片的张量形式数据
    """
    X = readImg(img_path)  # 原始图像
    imgSize = X.shape 
    size = np.prod([imgSize[0],imgSize[1],imgSize[2]]) # 图片总的元素数据
    missDataSize = int(np.ceil(np.prod([size, miss_percent])))  # 缺失元素数量
    nums = np.ones(size) # 生成全为1的数组
    nums[:missDataSize] = 0  # 缺失的数据填充为0
    np.random.shuffle(nums)  # 对只含0,1的数组进行乱序排列
    sparse_tensor = tl.tensor(nums.reshape(imgSize)) # 生成只含有0,1的张量
    miss_data_tensor = sparse_tensor * X
    return sparse_tensor, miss_data_tensor

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
    """HaLRTC算法实现
    Args:
        img_path: 图片路径
        miss_percent: 图片缺失数据百分比，取值为[0,1]

    Returns:
        X_hat: 通过HaLRTC算法复原的图片张量形式数据
    """
    Y1=Y2=Y3=Y
    start_time = datetime.datetime.now()
    for k in range(K):
        i_start_time = datetime.datetime.now()
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
        i_end_time = datetime.datetime.now()
        cost = i_end_time - i_start_time 
        print('the {num} times iteration ending,time cost is:{time} second'.format(num=k+1,time= cost.seconds))
    
    end_time = datetime.datetime.now()
    print("total time cost: {} second".format((end_time - start_time).seconds))
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
    Z, X = missImg(path, 0.6)
    imgShape = X.shape
   # 对应alpha
    a = abs(np.random.rand(3, 1))
    a = a / np.sum(a)
    K = 1 # 迭代次数
   #print(Z)
    rho = 1e-6
    Y = createZeroTensor(X.shape)
    # X_hat为通过算法还原的图片
    X_hat = HaLRTC(K,a,X,Z,rho,Y)
    # 显示对比图
    # im1 = cv2.resize(X, (400, 400))
    # im2 = cv2.resize(X_hat, (400, 400))
    # im3 = cv2.resize(originl_X, (400, 400))
    # imgs =  np.hstack((im1,im2,im3))
    # # 展示多个
    # cv2.imshow("mutil_pic", imgs)
    #hmerge = np.hstack((im1, im2,im3)) #水平拼接
    # vmerge = np.vstack((im1, im2)) #垂直拼接

    #cv2.imshow("compare_img", imgs)
    # cv2.imshow("test2", vmerge)
    #cv2.imshow("mutil_pic", imgs)
    cv2.namedWindow('originl', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow("originl", originl_X.astype(np.uint8))
    cv2.namedWindow('miss', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow("miss", X.astype(np.uint8))
    cv2.namedWindow('completion', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow("completion", X_hat.astype(np.uint8))
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


   

    
    
