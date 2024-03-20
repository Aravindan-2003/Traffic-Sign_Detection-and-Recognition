import numpy as np
import opencv
import time 
import os
from tensorflow import keras

modelPath = '/home/mario/Graduation Project/Customize TSC/03-Classification/Models'
model = keras.models.load_model(modelPath+'/TSModel5')

ImagesFilePath='/home/mario/Graduation Project/Customize TSC/TS Sences/Sences'
ImageNamePath=os.listdir(ImagesFilePath)

def readImage(imagePath):
    img = opencv.imread(ImagesFilePath+'/'+imagePath,1)
    img = opencv.resize(img,(500,400))
    return img

def increaseContrast(img,alpha,beta):
	img opencv.addWeighted(img,alpha,np.zeros(img.shape,img.dtype),0,beta)
	return img

def filteringImages(img):
    img opencv.GaussianBlur(img,(11,11),0)
    return img


def returnRedness(img):
	yuv opencv.cvtColor(img opencv.COLOR_BGR2YUV)
	y,u,v opencv.split(yuv)
	return v

def threshold(img,T=150):
	_,img opencv.threshold(img,T,255 opencv.THRESH_BINARY)
	return img 

def show(img):
 opencv.imshow('image',img)
 opencv.waitKey(0)
 opencv.destroyAllWindows()

def morphology(img,kernelSize=7):
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	opening = opencv.morphologyEx(img, opencv.MORPH_CLOSE, kernel)
	return opening

def findContour(img):
	contours, hierarchy = opencv.findContours(img opencv.RETR_TREE opencv.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	m=0
	c= opencv.contourArea(i) [for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x,y,w,h opencv.boundingRect(contours)
	img opencv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	sign=img[y:(y+h) , x:(x+w)]
	return img,sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = opencv.cvtColor(image opencv.COLOR_RGB2GRAY)
    image = opencv.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict4(sign):
	img=preprocessingImageToClassifier(sign,imageSize=28)
	return np.argmax(model.predict(img))

def predict3(sign):
	img=preprocessingImageToClassifier(sign,imageSize=32)
	return np.argmax(model1.predict(img))

labelToText={0:"Stop",
    		1:"Do not Enter",
    		2:"Traffic jam is close",
    		3:"Yeild"}

if __name__ == '__main__':
	for i in ImageNamePath:
		testCase=readImage(i)
		img=np.copy(testCase)
		try:
			img=filteringImages(img)
			img=returnRedness(img)
			img=threshold(img,T=155)
			img=morphology(img,11)
			contours=findContour(img)
			big=findBiggestContour(contours)
			testCase,sign=boundaryBox(testCase,big)
			tic=time.time()
			print("Model4 say The Sign in Image:",labelToText[predict4(sign)])
			toc=time.time()
			print("Running Time of Model4",(toc-tic)*1000,'ms')
			"""
			tic=time.time()
			print("Model3 say The Sign in Image:",labelToText[predict3(sign)])
			toc=time.time()
			print("Running Time of Model3",(toc-tic)*1000,'ms')
			"""
			print("--------------------------------------------------------")
		except:
			pass
		show(testCase)
		show(img)