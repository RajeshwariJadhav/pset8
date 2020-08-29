import cv2 as cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt


def display_sensitivity(sensitivity, k):
    plt.plot(np.arange(400, 705, 10), sensitivity[1],color='r')
    plt.plot(np.arange(400, 705, 10), sensitivity[2], color = 'g')
    plt.plot(np.arange(400, 705, 10), sensitivity[3], color='b')
            
def main():
    print("Start main()")

    visible = np.array([i for i in range(400,701,10)])
    red = np.loadtxt('./r.csv',delimiter=',')
    blue = np.loadtxt('./b.csv',delimiter=',')
    green = np.loadtxt('./g.csv',delimiter=',')

    red_interp = np.interp(visible, red[:,0], red[:,1])
    green_interp = np.interp(visible, green[:,0], green[:,1])
    blue_interp = np.interp(visible, blue[:,0], blue[:,1])

    
    sensitivity = np.array([visible, red_interp, green_interp, blue_interp])
    

    filter_435 = np.loadtxt("./FilterData/GG435FilterLongPass.csv",delimiter=',')
    filter_455 = np.loadtxt("./FilterData/GG455FilterLongPass.csv",delimiter=',')
    filter_475 = np.loadtxt("./FilterData/GG475FilterLongPass.csv",delimiter=',')
    filter_495 = np.loadtxt("./FilterData/GG495FilterLongPass.csv",delimiter=',')
    filter_515 = np.loadtxt("./FilterData/OG515FilterLongPass.csv",delimiter=',')
    filter_530 = np.loadtxt("./FilterData/OG530FilterLongPass.csv",delimiter=',')
    filter_550 = np.loadtxt("./FilterData/OG550FilterLongPass.csv",delimiter=',')
    filter_570 = np.loadtxt("./FilterData/OG570FilterLongPass.csv",delimiter=',')
    filter_590 = np.loadtxt("./FilterData/OG590FilterLongPass.csv",delimiter=',')

    filter_610 = np.loadtxt("./FilterData/RG610FilterLongPass.csv",delimiter=',')

    filter_630 = np.loadtxt("./FilterData/RG630FilterLongPass.csv",delimiter=',')
    filter_665 = np.loadtxt("./FilterData/RG665FilterLongPass.csv",delimiter=',')

    #400 -455
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(sensitivity[i]-filter_455[:,1])
        for i in range(len(filtered_1)):
            for j in range(len(filtered_1[0])):
                filtered_1[i][j] = max(0,filtered_1[i][j])
    plt.plot(np.arange(400, 705, 10), filtered_1[3], color='b')

    # 455-495
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(sensitivity[i]-filter_495[:,1])
        for i in range(len(filtered_1)):
        	for j in range(len(filtered_1[0])):
        		filtered_1[i][j] = max(0,filtered_1[i][j])

    filtered_2 = []
    for i in range(sensitivity.shape[0]):
        filtered_2.append(np.multiply(filter_455[:,1], filtered_1[i]))
    plt.plot(np.arange(400, 705, 10), filtered_2[3], color='b')



    # 495-550
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(sensitivity[i]-filter_550[:,1])
        for i in range(len(filtered_1)):
            for j in range(len(filtered_1[0])):
                filtered_1[i][j] = max(0,filtered_1[i][j])

    filtered_2 = []
    for i in range(sensitivity.shape[0]):
        filtered_2.append(np.multiply(filter_495[:,1], filtered_1[i]))
    plt.plot(np.arange(400, 705, 10), filtered_2[2], color='g')


    # 550-610
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(sensitivity[i]-filter_610[:,1])
        for i in range(len(filtered_1)):
            for j in range(len(filtered_1[0])):
                filtered_1[i][j] = max(0,filtered_1[i][j])

    filtered_2 = []
    for i in range(sensitivity.shape[0]):
        filtered_2.append(np.multiply(filter_550[:,1], filtered_1[i]))
    plt.plot(np.arange(400, 705, 10), filtered_2[2], color='g')


    # 610-665
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(sensitivity[i]-filter_665[:,1])
        for i in range(len(filtered_1)):
            for j in range(len(filtered_1[0])):
                filtered_1[i][j] = max(0,filtered_1[i][j])

    filtered_2 = []
    for i in range(sensitivity.shape[0]):
        filtered_2.append(np.multiply(filter_610[:,1], filtered_1[i]))
    plt.plot(np.arange(400, 705, 10), filtered_2[1], color='r')


    #665-700
    filtered_2 = []
    for i in range(sensitivity.shape[0]):
        filtered_2.append(np.multiply(filter_665[:,1], sensitivity[i]))
    plt.plot(np.arange(400, 705, 10), filtered_2[1], color='r')


    plt.show()
  
if __name__ == "__main__":
    main()