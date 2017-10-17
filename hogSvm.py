import os
import random
import cv2
import numpy as np

def hogDescriptorGen():
    #dont tuch
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64

    #can play with
    winSize = (200,600) #im size
    blockSize = (100,300) #imszie/2
    blockStride = (50,150) #imszie/4
    cellSize = (50,150) #imszie/4
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

    return hog

def generateData(inputData, hog,split):

    data = []
    dataLabels = []
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []

    # data = np.array
    yi = 0

    for class_ in os.listdir(inputData):
        classPath = os.path.join(inputData, class_)
        print class_
        if os.path.isdir(classPath):
            for sample in os.listdir(classPath):
                # print sample
                imPath = os.path.join(classPath, sample)
                # print imPath
                img = cv2.imread(imPath)
                descriptor = hog.compute(img)
                xi = [float(x[0]) for x in descriptor]
                xi.insert(0,float(yi))


                # if random.random() < 0.2:
                #     testData.app
                #
                # end(xi)
                #     testLabels.append(yi)
                # else:
                #     trainData.append(xi)
                #     trainLabels.append(yi)

                # sample = [float(yi),xi]

                # print sample
                data.append(xi)
        yi += 1

    print len(data)
    # for x in data:
        # if(x is not float and x is not []):
        #     print x

    nd_a = np.array([[x for x in y] for y in data])

    train_n = int(split*len(data))
    # NPdata = np.array(data,np.float32)
    trainData, testData = np.split(nd_a, [train_n])

    print "test data size:", len(testData), ", train data size:", len(trainData)
    print len(data)

    return trainData, testData

# def save2Csv(data):
#
# def loadFromCsv(path):


DIR = './input/'

hog = hogDescriptorGen()

trainData, testData = generateData(DIR,hog,0.60)

trainLabels = [x[0] for x in trainData]
trainData = [x[1:] for x in trainData]
testLabels = [x[0] for x in testData]
testData = [x[1:] for x in testData]


# print trainData

# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF)
svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
svm.setC(1)
# Set parameter Gamma
svm.setGamma(1)

print "============================="
print "start SVM sraining"
print "============================="
# Train SVM on training data
svm.train(np.float32(trainData),cv2.ml.ROW_SAMPLE, np.int32(trainLabels))
print "============================="
print "done SVM training"
print "============================="



# Save trained model
svm.save("trees_obst.yml")

print "SVM test"

# Test on a held out test set
testResponse = svm.predict(np.float32(testData))[1].ravel()

print len(testResponse)

error = 0

for i in range(len(testResponse)):
    if testResponse[i] != testLabels[i]:
        error +=1

print error
print "acc:", float(len(testResponse)-error)/len(testResponse), "%"
