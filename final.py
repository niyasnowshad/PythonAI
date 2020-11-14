import cv2 as cv
import numpy as np
import face_recognition
import os
from PIL import ImageGrab

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode)>0:
            encodes=encode[0]
        encodeList.append(encodes)
    return encodeList

def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv.cvtColor(capScr, cv.COLOR_RGB2BGR)
    return capScr

def motiondetect():
    i = input('1. Webcam \n'
              '2. Video file \n'
              'enter your choice : ' )
    if i == '1':
        cap = cv.VideoCapture(0)
    elif i == '2':
        cap = cv.VideoCapture('traffic.mp4')
    else:
        print('wrong input. try again!')

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        motiondiff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(motiondiff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

        dilate = cv.dilate(thresh, None, iterations=3)

        contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        cv.imshow("Motion Detection", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv.waitKey(1) & 0xFF == ord("e"):
            break

    cap.release()
    cv.destroyAllWindows()

def facerecogwbc():
    path = 'DataSet'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)

    for cl in myList:
        curImg = cv.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    encodeListKnown = findEncodings(images)
    print(len(encodeListKnown), 'encoding complete')

    cap = cv.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()

            else:
                name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Webcam', img)
        if cv.waitKey(1) & 0xFF == ord("e"):
            break

def facerecogss():
    path = 'DataSet'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)

    for cl in myList:
        curImg = cv.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    encodeListKnown = findEncodings(images)
    print(len(encodeListKnown), 'encoding complete')

    cap = cv.VideoCapture(0)

    while True:
        img = captureScreen()
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()

            else:
                name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Share Screen', img)
        if cv.waitKey(1) & 0xFF == ord("e"):
            break

def faceeye():
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    i = input('1. Webcam \n'
              '2. Video file \n'
              'enter your choice : ')
    if i == '1':
        cap = cv.VideoCapture(0)
    elif i == '2':
        cap = cv.VideoCapture('traffic.mp4')
    else:
        print('wrong input. try again!')

    while cap.isOpened():
        _, frame = cap.read()
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 4)

        for (x, y, w, h) in face_detect:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)

            eye_detect = eye_cascade.detectMultiScale(gray_img)

            for (ex, ey, ew, eh) in eye_detect:
                cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), thickness=3)

        cv.imshow("Face and Eye Tracker", frame)

        if cv.waitKey(1) & 0xFF == ord("e"):
            break

    cap.release()
    cv.destroyAllWindows()

def motiontrack():
    i = input('1. Webcam \n'
              '2. Video file \n'
              'enter your choice : ')
    if i == '1':
        cap = cv.VideoCapture(0)
    elif i == '2':
        cap = cv.VideoCapture('traffic.mp4')
    else:
        print('wrong input. try again!')

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        motiondiff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(motiondiff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

        dilate = cv.dilate(thresh, None, iterations=3)

        contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)

            if cv.contourArea(contour) < 700:
                continue

            cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame1, "Motion: {}".format("Yes"), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow("Motion", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv.waitKey(1) & 0xFF == ord("e"):
            break

    cap.release()
    cv.destroyAllWindows()

def objectdetect():
    thres = 0.45  # Threshold to detect object
    nms_threshold = 0.2
    cap = cv.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720)
    # cap.set(10,150)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        # print(type(confs[0]))
        # print(confs)

        indices = cv.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        # print(indices)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            cv.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Object Detection", img)
        if cv.waitKey(1) & 0xFF == ord("e"):
            break

    cap.release()
    cv.destroyAllWindows()


x=input('1. motion detection \n'
        '2. face recognition with webcam\n'
        '3. face recognition with sharescreen \n'
        '4. face and eye tracker \n'
        '5. motion tracker \n'
        '6. object detection \n'
        'enter your choice : ')
if x=='1':
    print('press E to exit')
    motiondetect()
elif x=='2':
    print('press E to exit')
    facerecogwbc()
elif x=='3':
    print('press E to exit')
    facerecogss()
elif x=='4':
    print('press E to exit')
    faceeye()
elif x=='5':
    print('press E to exit')
    motiontrack()
elif x=='6':
    print('press E to exit')
    objectdetect()
else:
    print('incorrect output')