import numpy as np
import cv2 as cv
import glob
import pickle



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
#15,22  640,480
chessboardSize = (22,15)
frameSize = (1920,1080)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('images/*.png')

for image in images:

    img = cv.imread(image)
    
    # img=cv.resize(img,(640,480))
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # img = cv.filter2D(img, -1, kernel)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10  )
    # gray=cv.boxFilter(img, 0, (7,7), img, (-1,-1), False, cv.BORDER_DEFAULT)

    # gray=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
    # cv.imshow("image",gray)

    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

    # gray = cv.filter2D(gray, -1, kernel_sharpening)
    # print(img)
    # print(img.shape)
    # img = cv.filter2D(image, -1, kernel)
    # cv.imshow("image",gray)
    # cv.waitKey(1000)
    

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize,None)
    print(ret)
    # break

    # cv.imshow('image',img)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.imshow('imgra', gray)
        cv.waitKey(10000)

    


cv.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))

############## UNDISTORTION #####################################################

img = cv.imread('images/Screenshot (48).png')


h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)

dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# print(dst)
# crop the image
x, y, w, h = roi
dst = dst[ x:x+w,y:y+h]
# print(dst)

cv.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )