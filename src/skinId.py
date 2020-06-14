import sys
import numpy as np
import cv2
import copy

#order
#1.colour segmentation
#2.Density Regularization
#3.Luminance Regularization
#4.Geometric Correction
#5.Contour Extraction

class Enums:
        posValue=255
        negValue=0
        kernelSize=int(4)
        zero= 0
        full = kernelSize*kernelSize*posValue
        # def setFull(self, val):
        #     self.full=val
        


#paper Face Segmentation Using Skin-Color Map in Videophone Applications by Chai and Ngan
skin_ycrcb_mint = np.array((80,133,77))
skin_ycrcb_maxt = np.array((255, 173, 127))

def main():

   

    font = cv2.FONT_HERSHEY_SIMPLEX   #font
    frame_number=0 #holds current number

    #load video file
    cap=cv2.VideoCapture('assets/cute little girl waving.mp4')

    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cap.get(3)),int(cap.get(4))))

    #total frames
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        _, frame=cap.read()

        #output frame
        # displayFrame=copy.deepcopy(frame)

        #kernel size
    
        #convert to ycrcb
        im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        #apply color segmentation
        #binary grayscale
        #skin colour 255, non-skin colour 0
        #binary  image
        binaryImage=convertBinary(im_ycrcb, Enums.posValue, Enums.negValue, skin_ycrcb_mint, skin_ycrcb_maxt)
        

        divisibleImage=setImageSizeDivisible(binaryImage, Enums.kernelSize)
        #getDensityVector for 4 by 4 kernel
        densityScoreVector=getDensityVector(divisibleImage, Enums.kernelSize)
        # np.savetxt("foo.csv", densityScoreVector, delimiter=",")
        
        # gaussian_blur = cv2.GaussianBlur(binaryImage,(4,4),0)
        

        #classify regions
        noiseRemovedImage=clusteringBlocks(densityScoreVector, 255,0)
        # np.savetxt("foo.csv", noiseRemovedImage, delimiter=",")
        # print(noiseRemovedImage.shape)
        # print(binaryImage.shape)

        #more erosion using 5*5 kernel
        kernel2=np.ones((5,5), np.uint8)
        erosion=cv2.erode(noiseRemovedImage, kernel2, iterations=2)

        # opening=cv2.morphologyEx(noiseRemovedImage, cv2.MORPH_OPEN, kernel2, iterations=6)
        closing=cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2, iterations=4)
        opening=cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2, iterations=1)


        #findcontours
        contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        displayFrame=copy.deepcopy(frame)


        #draw contor borders
        for cnt in contours:
            rect=cv2.minAreaRect(cnt)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            cv2.drawContours(displayFrame, [box], 0, (255,0,0),3)

        #draw contours
        cv2.drawContours(displayFrame,contours,-1,(0,255,0),3)

        #displayFrame
        # displayFrame=copy.deepcopy(noiseRemovedImage)

        #developer info----
        frame_number+=1 #increment frame number
        cv2.putText(displayFrame, 'frame number: '+str(frame_number)+"/"+str(total_frames), (50, 50), font, 0.8, (255, 0, 0), 2)
        #developer info---- 

        #write frame
        out.write(displayFrame)

        #show frame  
        cv2.imshow('img',displayFrame)
        # print(displayFrame.shape)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()  
    out.release()

    

def convertBinary(image, posValue, negValue, minCValue, maxCValue):
    binaryImage=np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            binaryImage[y,x]=negValue if(False in np.greater(image[y,x], minCValue) or False in np.less(image[y,x], maxCValue)) else posValue
    return binaryImage


def setImageSizeDivisible(image, size):
    tempImage=copy.deepcopy(image)
    #check hight is divisible by size
    rowRemainder=tempImage.shape[0]%size
    columnRemainder=tempImage.shape[1]%size
    if(rowRemainder !=0):
        appendRowPart=np.zeros(tempImage.shape[1])
        tempImage=np.vstack((tempImage,appendRowPart))
        setImageSizeDivisible(tempImage, size)
    if(columnRemainder!=0):
        appendColumnPart=np.zeros(tempImage.shape[0]).reshape(tempImage.shape[0],1)
        tempImage=np.hstack((tempImage,appendColumnPart))
        setImageSizeDivisible(tempImage, size)
    return tempImage

def partitionImage(image, size):
    width=image.shape[1]
    hight=image.shape[0]
    return [image[i:i+size, j:j+size] for i in range(0,hight, size) for j in range(0,width,size)]



def clusteringBlocks(image, posVal, negVal):
    #assign values to pixels arcoding to D
        #D=0 -> zero
        #0<D<16*255 -> intermidate look for 3*3 neighbor 2 full values to be positive
        #D=16*255 -> full
    
    resultImage=binaryImage=np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for i in range(1,image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            # print(image[i,j])
            if(image[i,j]==Enums.zero):
                # print("zero match")
                resultImage[i,j]=negVal
            #Erode 
            elif(image[i,j]==Enums.full):
                if(neighborFullCheck(image,i,j,Enums.zero, Enums.full)<5):
                    resultImage[i,j]=negVal
                else:
                    resultImage[i,j]=posVal
            #Dilate 
            elif(Enums.zero<image[i,j] and image[i,j]<Enums.full):
                if(neighborFullCheck(image, i,j, Enums.zero, Enums.full)>2):
                    resultImage[i,j]=posVal
                else:
                    resultImage[i,j]=negVal
    return resultImage

def neighborFullCheck(image, rowCord, colCord, zero, full):
    count=0
    if(image[rowCord-1,colCord-1]==full):
        count=count+1
    if(image[rowCord-1,colCord]==full):
        count=count+1
    if(image[rowCord-1,colCord+1]==full):
        count=count+1
    if(image[rowCord,colCord-1]==full):
        count=count+1
    if(image[rowCord,colCord+1]==full):
        count=count+1
    if(image[rowCord+1,colCord-1]==full):
        count=count+1
    if(image[rowCord+1,colCord]==full):
        count=count+1
    if(image[rowCord+1,colCord+1]==full):
        count=count+1
    return count
   


def calculateDensityLevels(blockArray, image, size):
    noOfClusters=(image.shape[0]*image.shape[1])//(size*size)
    # print(noOfClusters)
    score_image=np.zeros(int(noOfClusters))
    newimage=np.zeros((image.shape[0], image.shape[1]))

     #calculate totals
    for i, one in enumerate(blockArray):
        total=0
        for j in range(one.shape[0]):
            for k in range(one.shape[1]):
                total=total+one[j,k]
        score_image[i]=total
        # np.savetxt("foo.csv", score_image, delimiter=",")

    #merge blocks together
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            scoreIndex=((image.shape[1]//size))*(r//size)+(c//size)
            newimage[r][c]=score_image[scoreIndex]
    
    return newimage


def getDensityVector(image, size):
    opImage=image.copy()
    #set divisible array size
    # print(tempImage.shape)
    #partition image to size*size blocks
    blockArray=partitionImage(image, size)
    # print(len(blockArray))
    # tarr=blockArray[0].copy()
    # for i in range(1, len(blockArray)):
    #     tarr=np.vstack((tarr,blockArray[i]))
    # np.savetxt("foo.csv", tarr, delimiter=",")
    # print("done writiing")
    # print(blockArray)
    return calculateDensityLevels(blockArray, image, size)

main()


    


