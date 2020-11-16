import os
import cv2
import queue
import random
import threading
import face_recognition
import numpy as np
from sklearn import svm
import joblib


q = queue.Queue()


# 加载人脸图片并进行编码
def Encode():
    print("Start Encoding")
    image_path = 'C:\\Users\\Administrator\\Desktop\\face_recognition-master\\examples\\knn_examples\\test\\'
    person_list = os.listdir(image_path)
    #print(person_list)
    for person in person_list:
        image_list = os.listdir(image_path)
        for image in image_list:
            #print(person +'   '+ image)
            face = face_recognition.load_image_file(image_path + image)
            face_locations = face_recognition.face_locations(face)
            face_enc = face_recognition.face_encodings(face, face_locations)[0]
            np.save(image.split(".")[0], face_enc)
            #print(image.split(".")[0])

# 训练SVC
def Train_SVC():
    print("Start Training")
    encodings = []
    names = []
    name_dict = {}
    # 加载人脸数据库并学习
    data_path = "C:\\Users\\Administrator\\Desktop\\face_recognition-master\\examples\\knn_examples\\test\\"
    person_list = os.listdir(data_path)
    print(person_list)
    for i, person in enumerate(person_list):
        data_list = os.listdir(data_path)
        for data in data_list:
            print(i, data)
            encodings.append(np.load(data_path + person).tolist())
            names.append(int(i))
            name_dict[i] = person

    clf = svm.SVC(C=20, probability=True)
    clf.fit(encodings, names)
    joblib.dump(clf, "my_model.m")

    f = open('name.txt', 'w')
    f.write(str(name_dict))
    f.close()


# 线程1获取网络摄像头图像
def Receive():
    print("Start Reveive")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture("rtsp://admin:a123456789@121.248.50.30/h264/ch1/main/av_stream")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


# 线程2进行人脸检测识别并显示
def Display():
    print("Start DisPlaying")

    clf = joblib.load("my_model.m")

    f = open('name.txt', 'r')
    name_dict = eval(f.read())
    f.close()

    face_locations = []
    face_names = []
    count = 0
    threshold = 1/(0.75 * len(name_dict))

    while True:
        if not q.empty():

            count += 1

            frame = q.get()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # 每0.2秒进行一次人脸检测
            if count % 5 == 0:
                face_locations = face_recognition.face_locations(rgb_small_frame)

            # 每0.4秒进行一次人脸识别
            if count % 10 == 0:

                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    #print(clf.predict[face_encoding])
                    print(clf.predict_proba([face_encoding]))  #
                    if np.max(clf.predict_proba([face_encoding])) > threshold:
                        face_names.append(name_dict[int(clf.predict([face_encoding]))])
                    else:
                        face_names.append("Unknown")

            # 显示人脸定位框及姓名
            for (top, right, bottom, left), name in zip(face_locations, face_names):

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    #Encode()
    Train_SVC()
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
