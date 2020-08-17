import cv2 as cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def read_filter_csv(csv_dir):
    filter_X = []
    with open(csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            filter_X.append([float(elem) for elem in row])
    print("filter X shape: ", np.array(filter_X).shape)
    filter_X = np.transpose(filter_X)
    new_filter = np.array(filter_X[:,0:2])
    print("new filter before for loop shape: ", new_filter.shape)
    for i in range(1,6):
        new_filter = np.append(new_filter, filter_X[:, 2*i:(2*i+2)], axis=0)
    
    # We only want 400 to 700
    new_filter = new_filter[20:51]
    return np.transpose(new_filter)[1]

def display_sensitivity(sensitivity):
    plt.plot(np.arange(400, 705, 10), sensitivity[0], color='r')
    plt.plot(np.arange(400, 705, 10), sensitivity[1], color='g')
    plt.plot(np.arange(400, 705, 10), sensitivity[2], color='b')
    plt.show()
            
def main():
    print("Start main()")
    dir_spectral = os.getcwd() + "/coneSensitivity.csv"
    sensitivity = []
    
    with open(dir_spectral) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sensitivity.append([float(elem) for elem in row])
    sensitivity = np.transpose(sensitivity[::2])
    
    # These filters start at 200, step=10 until 1200 after which, step=50
    filter_435 = read_filter_csv(os.getcwd()+"/FilterData/GG435FilterLongPass.csv")
    filter_475 = read_filter_csv(os.getcwd()+"/FilterData/GG475FilterLongPass.csv")
    filter_630 = read_filter_csv(os.getcwd()+"/FilterData/RG630FilterLongPass.csv")
    filter_665 = read_filter_csv(os.getcwd()+"/FilterData/RG665FilterLongPass.csv")

    print("Filter_435 shape: ", filter_435.shape)
    print("filter 435: ", filter_435)
    print("Sensitivity shape: ", sensitivity.shape)

    # Original
    display_sensitivity(sensitivity[1:])
    
    # 475 filter
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(np.multiply(filter_435, sensitivity[i]))
    display_sensitivity(filtered_1)
    
if __name__ == "__main__":
    main()