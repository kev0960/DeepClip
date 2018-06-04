import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)

import copy
import flowfilter.gpu.flowfilters as gpufilter


cap = cv2.VideoCapture("e.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# GPU filter object with 2 pyramid levels
gpuF = gpufilter.PyramidalFlowFilter(256, 256, 2)
gpuF.gamma = [1, 1]                                          # gains for each level
gpuF.maxflow = 1.0                                      # maximum optical flow value
gpuF.smoothIterations = [2,2]                               # smooth iterations per level

i = -1
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    i += 1
    if i % 5 == 0 :
        gpuF.loadImage(prvs)
        gpuF.loadImage(next)

        gpuF.compute()
        flow = gpuF.getFlow()

        med_x = np.median(flow[..., 0])
        med_y = np.median(flow[..., 1])

        print(med_x, med_y)
        flow[..., 0] -= med_x
        flow[..., 1] -= med_y

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(1500)
    '''
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cloned_frame = copy.copy(frame2)
    for i in range(0, cloned_frame.shape[0], 10):
        for j in range(0, cloned_frame.shape[1], 10):
            pt1 = (i, j)
            pt2 = (int(i + mag[i][j] * np.cos(ang[i][j])), int(j + mag[i][j] * np.sin(ang[i][j])))

            cv2.arrowedLine(cloned_frame, pt1, pt2, (0, 255, 255), 1, tipLength=2)

    cv2.imshow('frame2',bgr)
    #cv2.imshow('frame2', cloned_frame)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    '''
    prvs = next
cap.release()
cv2.destroyAllWindows()
