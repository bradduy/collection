import json
import cv2
import glob
import os


def extractFramesInVideo():
    videoPath = './video.mp4'

    cap = cv2.VideoCapture(videoPath)
    i = 0
    print('---------START---------')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        number = str('{:06d}'.format(i))
        cv2.imwrite('frame_' + number + '.png', frame)
        i += 1
        print('---------on---------', number)
    print('---------END---------')
    cap.release()
    cv2.destroyAllWindows()


def convertJsonFileToTxtFile():
    jsonFilePath = './data.json'

    with open(jsonFilePath) as f:
        jsonFile = json.load(f)
    textFile = open("convertedData.txt", "a")
    name = jsonFile['name']
    textFile.write("The author is " + name + "\n")
    textFile.close()


def resizedImage():
    inputFolder = '/home/user/'
    resizedFolder = 'resized_images'

    os.mkdir(resizedFolder)

    for img in glob.glob(inputFolder + '/*.jpg'):
        tail = img.split(inputFolder, 1)[1]
        img = cv2.imread(img)
        resizedImg = cv2.resize(img, (224, 224))
        cv2.imwrite(inputFolder + resizedFolder + tail, resizedImg)
        cv2.imshow('image', resizedImg)
        cv2.waitKey(30)
    cv2.destroyAllWindows()

