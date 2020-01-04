import numpy as np
import matplotlib.pyplot as plt

datalist = [0,0]#can not be none if vstack
def writedata(datalist):
    # python list that is needed to be save in the txt file
    # convert list to array
    data_array = np.array(datalist)
    # saving...
    np.savetxt('data.txt',data_array,fmt = '%.2f',delimiter=',')
    print ('Finish saving csv file')

def plotdata():
    data = np.loadtxt('data_exp.txt',dtype = None,delimiter=',')
    print("txt.shape",data.shape)

    #calculate the success rate
    count = 0
    rate = 0
    total = np. size(data[:,0])
    #print("total",total)
    for i in range(total):
        if data[i][1]<-0.9:
            count = count + 1
        i = i + 1
    rate = 1 -(count/total)
    print("successful rate",rate )

    plt.plot(data[:,0],data[:,1])
    plt.show()

def main():
    #global datalist
    #data_episode=[8,9]
    #datalist = np.vstack((datalist,data_episode))
    #writedata(datalist)
    plotdata()

if __name__ == '__main__':
    main()
