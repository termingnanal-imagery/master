import h5py ,matplotlib ,shutil ,os ,cv2 ,matplotlib.pyplot as plt ,numpy as np, glob

from scipy.fftpack import fft,ifft
from PyQt5 import QtCore


def loadmat(file):
    global data
    #打开mat文件
    mat = h5py.File(file)
    origindata = mat['data'][:]
    origindata = ((origindata - origindata.min()) / (origindata.max() - origindata.min()) * 255)
    num = int(origindata.shape[0]/4)

    data = np.ones((num,640,240))
    c=0

    while(c<int(origindata.shape[0]/4)):
        data[c,:,:] = origindata[c*4,:,:]
        c = c+1


    print('数组大小'+ str(data.shape))
    print('compelet mat load')

    # savepv('nonesub')

    advance('PPT')


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

    savepv('nonesub')


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
    mat1 = mat1.reshape(data.shape[0],data.shape[1]*data.shape[2])
    mat2 = np.ones(mat1.shape,dtype=complex)
    i = 0
    fdata = np.fft.fft(mat2)
    print(fdata[:10,:10])
    print(fdata.shape)


def TSR():
    global data
    mat = h5py.File('../DATA/14.mat')
    origindata = mat['data'][:]
    tdata = origindata.reshape((origindata.shape[0],origindata.shape[1]*origindata.shape[2])).T
    print(tdata.shape)






def PCA():
    global data


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




