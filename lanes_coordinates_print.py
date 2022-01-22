import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny = cv2.Canny(blur,low_threshold,high_threshold)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    #print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = [] #empty list here
    right_fit = [] #empty list here
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)#degree 1
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        #slope of the line on the left side -ve and on right side +ve
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    #print(left_fit)
    #print(right_fit)
    left_fit_average = np.average(left_fit,axis = 0)#axis = 0 because to traverse column wise
    right_fit_avergae = np.average(right_fit,axis = 0)
    #print(left_fit_average,'left')
    #print(right_fit_avergae,'right')
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_avergae)
    return np.array([left_line,right_line])


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
average_lines = average_slope_intercept(lane_image,lines)

cv2.imshow('result',cropped_image)
cv2.waitKey(0)
