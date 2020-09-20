import cv2
import numpy as np

capture = cv2.VideoCapture(0)

imageNormalSize = 120
getImage = cv2.imread('baby.jpeg')
smallImage = cv2.resize( getImage, (imageNormalSize, imageNormalSize))


smallImageoldx = 200
smallImageoldy = 170

smallImageoldwidth = smallImageoldx + 120
smallImageoldhigh = smallImageoldy + 120

resizeImageMore = 0
distanceInImage = 0

def duplicateArea(areaList):
    
    newAreaList = []
    for al in areaList:
        newArea = 1
        for inNew, area in enumerate(newAreaList):
            lowArea = area[0] - 25
            maxArea =  area[0] + 25
            if lowArea < al[0] < maxArea:
                newArea = 0
        if newArea == 1:
            newAreaList.append([al[0],al[1]])
    return newAreaList

def calculateDistance(topLists):
    result = [0,0,0]
    if len(topLists) > 1:
        top1x = topLists[0][0]
        top2x = topLists[1][0]
        distance = top1x - top2x
        if top2x > top1x:
            distance = top2x - top1x

        distanceX = int(top1x/2) + int(top2x/2)
        distanceY = int(topLists[0][1]/2) + int(topLists[1][1]/2)  
        result = [distance,distanceX, distanceY]
    return result


def getContours(img):
    getDistanceFromTopFinger = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area >15000:
            topArea = []

            approx = cv2.convexHull(cnt)

            for indexapprox ,i in enumerate(approx):
                newArea = 1
                cv2.drawContours(imageContour, approx, indexapprox, (0 ,255 ,0), 5)
                if len(topArea) == 0:
                    topArea.append([i[0][0],i[0][1]])
                else:
                    for index, area in enumerate(topArea):
                        lowArea = area[0] - 20
                        maxArea =  area[0] + 20
                        if lowArea < i[0][0] < maxArea:
                            newArea = 0
                            if i[0][1] > area[1]:
                                topArea[index] = [i[0][0], i[0][1]]
                    if newArea == 1:
                        topArea.append([i[0][0],i[0][1]])

            only1TopArea = duplicateArea(topArea)
            newTopdata = sorted(only1TopArea, key=lambda x: x[1] , reverse=True) 
            getDistanceFromTopFinger = calculateDistance(newTopdata)

            for new in newTopdata:
                cv2.circle(imageContour, ( new[0] , new[1] ), 3, (150,150,255), 4)   

    global smallImageoldx
    global smallImageoldy
    global smallImageoldhigh
    global smallImageoldwidth
    global distanceInImage
    global resizeImageMore
    global imageNormalSize
    global smallImage

    if len(getDistanceFromTopFinger) > 0 and getDistanceFromTopFinger[1] in range(smallImageoldx, smallImageoldwidth  ) and getDistanceFromTopFinger[2] in range(smallImageoldy, smallImageoldhigh ):
        if distanceInImage == 0:
            distanceInImage = getDistanceFromTopFinger[0]
    else:
        distanceInImage = 0
        resizeImageMore = 0
    imageNormalSize = 120
    if distanceInImage > 0 :
        resizeImageMore = getDistanceFromTopFinger[0] - distanceInImage
        imageNormalSize = imageNormalSize + resizeImageMore
        

    smallImage = cv2.resize( getImage, (imageNormalSize, imageNormalSize))
    newImageHeight = smallImageoldhigh + resizeImageMore
    newImagewidht = smallImageoldwidth + resizeImageMore
    imageContour[smallImageoldy: newImageHeight, smallImageoldx: newImagewidht] = smallImage


while True:
    _, frame = capture.read()
    imageContour = frame.copy()
    
    medianImage = cv2.medianBlur(frame,17)
    gaussianImage = cv2.GaussianBlur(medianImage,(5,5),0)
    bilateralImage = cv2.bilateralFilter(gaussianImage,9,75,75)
    imgHSV = cv2.cvtColor(bilateralImage, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 33, 35]) 
    upper = np.array([39, 186 , 145])

    mask = cv2.inRange(imgHSV, lower, upper)

    getContours(mask)


    cv2.imshow("video",imageContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()