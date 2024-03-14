import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os

def computehoughTransformCoord(x1, y1, x2, y2):

    if x1 == x2:
        return(np.pi/2, abs(y1-y2), abs(x1))

    k =  (y1 - y2) / (x1 - x2)
    theta = np.arctan(-k)

    A = y2 - y1
    B = x2 - x1
    C = abs(x2*y1 - x1*y2)
    length = np.sqrt(A**2 + B**2)
    rho = abs(C / length)
    
    return np.degrees(theta), length, rho

def simulateSigma(image0, noise_std, low, high, Blur = False, Blur_std = 0.5, iterNum = 2000, draw = False, List = False):

    height, width = image0.shape
    bound = int(width / 2)

    # thetaT = 45
    # rhoT = height/np.sqrt(2)-0.5

    # for i in range(0, height):
    #     image0[i, 0: height-i] = low
    #     image0[i, height-i : height] = high

    image0[0:bound, :] = high
    image0[bound:, :] = low

    thetaT = 0
    rhoT = 31.5

    lsd = cv2.createLineSegmentDetector(0)
    thetaList = []
    rhoList = []

    for _ in range(0, iterNum):
        image = image0.copy()

        if Blur == True:
            image = cv2.GaussianBlur(image, (5, 5), Blur_std)

        image = image + np.round(np.random.normal(0, noise_std, image0.shape)).astype(int)
        image = np.uint8(image)
        
        if draw == True:
            cv2.imwrite("blur_image.png", image)

        lines, width, prec, nfa = lsd.detect(image)

        drawn_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if lines is None:
            print("================empty lines==============")
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                theta, length, rho = computehoughTransformCoord(x1, y1, x2, y2)

                if abs( rho - rhoT ) > 3 or abs(theta - thetaT) > 7:
                    continue

                thetaList.append(theta)
                rhoList.append(rho)

                if draw == True:
                    x1, y1, x2, y2 = map(int, line[0])
                    drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

                # print(f"theta : {theta - thetaT}, rho : {rho - rhoT}")
        
        if draw == True:
            cv2.imwrite("drawn_img.png", drawn_img)

    thetaList = np.asarray(thetaList)
    rhoList = np.asarray(rhoList)
    thetaSigma = np.std(thetaList)
    rhoSigma = np.std(rhoList)
    # print(np.mean(rhoList))
    # print(thetaSigma, rhoSigma)

    if List == True:
        return thetaList, rhoList

    return thetaSigma, rhoSigma