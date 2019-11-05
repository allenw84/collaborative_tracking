#!/usr/bin/env python
# coding: utf-8
import numpy as np
import collections
import cv2

def __color(img):
    #img = cv2.imread(img_name)
    cv2.imshow("FUCK",img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([100,43,46])
    upper_blue = np.array([124,255,255])

    # define range of green color in HSV 
    lower_green = np.array([35,43,46])
    upper_green = np.array([77,255,255])

    # define range of red1 color in HSV 
    lower_red = np.array([156,43,46])
    upper_red = np.array([180,255,255])

    # define range of yellow color in HSV
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])

    # define range of purple
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])

    # define rnage of orange
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])

    # define range of white
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])

    # define range of black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])

    # define range of gray
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Threshold the HSV image to get only red colors   
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    res_blue = cv2.bitwise_and(img, img, mask= mask_blue)
    res_green = cv2.bitwise_and(img, img, mask= mask_green)
    res_red = cv2.bitwise_and(img, img, mask= mask_red)
    rows, cols = mask_red.shape

    c_red = c_blue = c_yellow = c_purple = c_orange = c_white = c_black = c_gray = c_green = 0
    for i in range(0,rows):
        for j in range(0,cols):
            if mask_red[i,j] == 255:
                c_red = c_red + 1
            if mask_yellow[i,j] == 255:
                c_yellow = c_yellow + 1
            if mask_black[i,j] == 255:
                c_black = c_black + 1
            if mask_blue[i,j] == 255:
                c_blue = c_blue + 1
            if mask_gray[i,j] == 255:
                c_gray = c_gray + 1
            if mask_green[i,j] == 255:
                c_green = c_green + 1
            if mask_orange[i,j] == 255:
                c_orange = c_orange + 1
            if mask_purple[i,j] == 255:
                c_purple = c_purple + 1
            if mask_white[i,j] == 255:
                c_white = c_white + 1

    s = mask_red.size
    if c_red != 0:
        p_red = c_red/s
    if c_yellow != 0:
        p_yellow = c_yellow/s
    if c_black != 0:
        p_black = c_black/s
    if c_blue != 0:
        p_blue = c_blue/s
    if c_gray != 0:
        p_gray = c_gray/s
    if c_green != 0:
        p_green = c_green/s
    if c_orange != 0:
        p_orange = c_orange/s
    if c_purple != 0:
        p_purple = c_purple/s
    if c_white != 0:
        p_white = c_white/s
    '''
    print('red percentage is: {:.2%}'.format(p_red))
    print('yellow percentage is: {:.2%}'.format(p_yellow))
    print('black percentage is: {:.2%}'.format(p_black))
    print('blue percentage is: {:.2%}'.format(p_blue))
    print('gray percentage is: {:.2%}'.format(p_gray))
    print('green percentage is: {:.2%}'.format(p_green))
    print('orange percentage is: {:.2%}'.format(p_orange))
    print('purple percentage is: {:.2%}'.format(p_purple))
    print('white percentage is: {:.2%}'.format(p_white))
    '''
    color = [p_red, p_yellow, p_black, p_blue, p_gray, p_green, p_orange, p_purple, p_white]
    tmp = 0.0
    
    for j in range (8, 0, -1):
        jump = False
        for i in range (0, j):
            if color[i] < color[i+1]:
                tmp = color[i]
                color[i] = color[i+1]
                color[i+1] = tmp
                jump = True
        if jump == False:
            break
        
    color_dic = {p_red:'red', p_yellow:'yellow', p_black:'black', p_blue:'blue', p_gray:'gray', p_green:'green', p_orange:'orange', p_purple:'purple', p_white:'white'}
    # print('most color is:',color_dic[most_color])
    color_list = []
    
    if color[0] >= 0.5 :
        color_list.append(color_dic[color[0]])
        return color_list
    elif color[0] >= 0.33 & color[1] >= 0.33:
        color_list.append(color_dic[color[0]])
        color_list.append(color_dic[color[1]])
        return color_list
    else:
        color_list.append(color_dic[color[0]])
        color_list.append(color_dic[color[1]])
        color_list.append(color_dic[color[2]])
        color_list.append(color_dic[color[3]])
        return color_list

#print(color("person3.jpg"))

def color(img):
    return ['red']

def merge_color(img, msg):
    for f in msg:
        roiImg = img[f[2][0]:f[2][2],f[2][1]:f[2][3]]  #
        # roiImg = img[f[2][2]:f[2][3],f[2][0]:f[2][1]]
        c = color(roiImg)
        f[2] = c
    
    return msg

