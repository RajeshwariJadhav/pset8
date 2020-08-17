import cv2 as cv2
import numpy as np
import os

print("Hello")
def main():
    print("Start main()")
    dir_spectral = os.getcwd() + "/coneSensitivity.csv"
    sensitivity = []
    
    with open(dir_spectral) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            sensitivity.append(row)
        print("Sensitivity: ", sensitivity)
        
if __name__ == "__main__":
    main()