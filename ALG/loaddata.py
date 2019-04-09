import h5py ,matplotlib ,shutil ,os ,cv2 ,matplotlib.pyplot as plt ,numpy as np, glob

from sklearn.decomposition import PCA
from scipy.fftpack import fft,ifft
from PyQt5 import QtCore


def loadmat(file):
    global data
    #打开mat文件
    mat = h5py.File(file)
    origindata = mat['data'][:]
    num = int(origindata.shape[0]/4)

    data = np.ones((num,640,240))
    c=0

    while(c<int(origindata.shape[0]/4)):
        data[c,:,:] = origindata[c*4+3,:,:]
        c = c+1
    data = ((data - data.min()) / (data.max() - data.min()) * 255)


    print('数组大小'+ str(data.shape))
    print('compelet mat load')



    advance('PCA')

    # savepv('SD')


def loadvideo(filename):
    global data
    #打开视频文件
    video = cv2.VideoCapture(filename)
    num = int(video.get(7))

    rval, frame = video.read()

    c = 0
    data = np.ones((num,640,240))

    #读取每一帧的灰度信息
    while (rval):
        framedata = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).T

        # if(c == 0):
        #     firstframe = framedata[:]

        # else:

            # data[c-1,:,:] = framedata[:] - firstframe

        c = c+1

        rval, frame = video.read()
    video.release()
    data = ((data - data.min()) / (data.max() - data.min()) * 255)

    
    print('数组大小'+ str(data.shape))
    print("complet videoload")


    savepv('sub')


def advance(function):
    if function == 'PPT':
        PPT()
    elif function == 'TSR':
        TSR()
    else:
        PCA()


def PPT():
    global data
    # mat1 = np.transpose(data,(0,2,1))
    mat1 = data[:]
    # mat1 = mat1.reshape(data.shape[0],data.shape[1]*data.shape[2])
    mat2 = np.ones(mat1.shape)
    N = mat1.shape[0]
    for x in range(mat1.shape[1]):
        for y in range(mat1.shape[2]):
            ex = mat1[:,x,y]
            sum = complex(0,0)
            for i in range(N):
                yi = complex(0,-2*np.pi*i*data.shape[1]*data.shape[2])
                sum = ex[i]*np.exp(yi/N)+sum
            sum = sum / N
            mat2[:,x,y] = np.arctan(sum.imag/sum.real)
    mat2 = ((mat2 - mat2.min()) / (mat2.max() - mat2.min()) * 255)
    data = mat2[:]



def TSR():
    global data
    N = data.shape[0]
    x = np.arange(0.08,0.08*(N+1),0.08)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ex = data[:,i,j]
            z = np.polyfit(x,ex,8)
            p = np.poly1d(z)
            p = p.deriv().deriv()
            data[:,i,j] = p(x)


def PCA():
    global data
    data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    data.transpose(1,0)
    pca = PCA(n_components=8)
    ex = pca.fit_transform(data)
    print(ex.shape)

    # ex = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    # for x in range(data.shape[0]):
    #     data[x,:] = data[x,:]-np.mean(data[x,:])
    # F = 1/data.shape[0]*(ex.T.dot(ex))
    # [a,b] = np.linalg.eig(F)
    # b = b[:,:8]
    # # data =
    # # print(data.shape)


def savepv(name):
    global data

    # 获取总帧数
    framnum = data.shape[0]

    # 转换mat文件到本地图片
    for x in range(1, framnum + 1):

        fig,ax = plt.subplots()

        framedata = data[x - 1, :, :]

        # 输出伪彩色图像
        # ax.imshow(framedata.T, cmap='gray')
        ax.imshow(framedata.T, cmap=plt.cm.jet)
        plt.axis('off')

        height,width = framedata.T.shape
        fig.set_size_inches(width/100.0 , height/100.0)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        filename = '../IMG/' + str(x)+ name + '.jpg'
        plt.savefig(filename)
        plt.close()

    # videowriter = cv2.VideoWriter(name+'.avi',cv2.VideoWriter_fourcc(*'MJPG'),100,(640,240))
    #
    # img = glob.glob('../IMG/*.jpg')
    #
    # for x in img:
    #     frame = cv2.imread(x)
    #     videowriter.write(frame)
    # videowriter.release()




