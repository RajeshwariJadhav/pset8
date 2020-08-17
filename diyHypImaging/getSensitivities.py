import cv2 as cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# def read_filter_csv(csv_dir):
#     filter_X = []
#     with open(csv_dir) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter = ',')
#         for row in csv_reader:
#             filter_X.append([float(elem) for elem in row])
#     print("filter X shape: ", np.array(filter_X).shape)
#     filter_X = np.transpose(filter_X)
#     new_filter = np.array(filter_X[:,0:2])
#     print("new filter before for loop shape: ", new_filter.shape)
#     for i in range(1,6):
#         new_filter = np.append(new_filter, filter_X[:, 2*i:(2*i+2)], axis=0)
    
#     # We only want 400 to 700
#     new_filter = new_filter[20:51]
#     return np.transpose(new_filter)[1]

def display_sensitivity(sensitivity):
    plt.plot(np.arange(400, 705, 10), sensitivity[1],color='r')
    plt.plot(np.arange(400, 705, 10), sensitivity[2], color='g')
    plt.plot(np.arange(400, 705, 10), sensitivity[3], color='b')
            
def main():
    print("Start main()")
    dir_spectral = os.getcwd() + "/coneSensitivity.csv"
    sensitivity = []
    
    with open(dir_spectral) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sensitivity.append([float(elem) for elem in row])
    sensitivity = np.transpose(sensitivity[::2]) #at steps of 10
    
    filter_435 = np.loadtxt("./FilterData/GG435FilterLongPass.csv",delimiter=',')
    filter_455 = np.loadtxt("./FilterData/GG455FilterLongPass.csv",delimiter=',')
    filter_475 = np.loadtxt("./FilterData/GG475FilterLongPass.csv",delimiter=',')
    filter_495 = np.loadtxt("./FilterData/GG495FilterLongPass.csv",delimiter=',')
    filter_515 = np.loadtxt("./FilterData/OG515FilterLongPass.csv",delimiter=',')
    filter_530 = np.loadtxt("./FilterData/OG530FilterLongPass.csv",delimiter=',')
    filter_630 = np.loadtxt("./FilterData/RG630FilterLongPass.csv",delimiter=',')
    filter_665 = np.loadtxt("./FilterData/RG665FilterLongPass.csv",delimiter=',')
  
    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(np.multiply(filter_435[:,1], sensitivity[i]))
    display_sensitivity(filtered_1)

    filtered_1 = []
    for i in range(sensitivity.shape[0]):
        filtered_1.append(np.multiply(filter_515[:,1], sensitivity[i]))
    display_sensitivity(filtered_1)


    plt.show()
  
if __name__ == "__main__":
    main()
