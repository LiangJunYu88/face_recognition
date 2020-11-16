import face_recognition
import cv2
import numpy as np
import time

video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# VideoCapture打开摄像头，0为笔记本内置摄像头，1为外USB摄像头，或写入视频路径

mayun_img = face_recognition.load_image_file("../known/mayun.jpg")
jobs_img = face_recognition.load_image_file("../known/jobs.jpg")
ljy_img=face_recognition.load_image_file("../known/ljy.jpg")

# lcy_img=face_recognition.load_image_file("../known/lcy.jpg")
# lhk_img=face_recognition.load_image_file("../known/lhk.jpg")
# lgy_img=face_recognition.load_image_file("../known/lgy.jpg")
# lz_img=face_recognition.load_image_file("../known/lz.jpg")


#对图片进行编码

mayun_face_encoding = face_recognition.face_encodings(mayun_img)[0]
jobs_face_encoding = face_recognition.face_encodings(jobs_img)[0]
ljy_face_encoding = face_recognition.face_encodings(ljy_img)[0]

# lcy_face_encoding = face_recognition.face_encodings(lcy_img)[0]
# lhk_face_encoding = face_recognition.face_encodings(lhk_img)[0]
# lgy_face_encoding = face_recognition.face_encodings(lgy_img)[0]
# lz_face_encoding = face_recognition.face_encodings(lz_img)[0]


#存为数组
known_faces = [
    mayun_face_encoding,
    jobs_face_encoding,
    ljy_face_encoding,
    # lcy_face_encoding,
    # lhk_face_encoding,
    # lgy_face_encoding,
    # lz_face_encoding,
]
print(known_faces)

# opencv的人脸检测
# def detect_byOpencv(img):
#     start_detect_time = time.time()
#     face_cascade=cv2.CascadeClassifier('C:/ProgramData/Anaconda3/envs/Python3.7/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     loctions = []
#     for (x, y, w, h) in faces:
#         top = y
#         right = x + w
#         bottom = y + h
#         left = x
#         loctions.append((top, right, bottom, left))
#     print("opencv detect:", time.time() - start_detect_time)
#     return loctions

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    ret, frame = video_capture.read()
    #  video_capture.read()按帧读取视频，ret,frame是获video_capture.read()方法的两个返回值。
    #  其中ret是布尔值，如果读取帧是正确的则返回True，
    #  如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。

    #start3=time.clock()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  #缩放图片
    # end3=time.clock()
    # timeResize=end3-start3
    # print("缩放图片耗时：",timeResize,"s")
    # 对截取到的图像进行处理

    if process_this_frame:

        #face_locations=detect_byOpencv(small_frame)

        face_locations = face_recognition.face_locations(small_frame)  #人脸定位
        face_encodings = face_recognition.face_encodings(small_frame, face_locations) #计算编码

        similarities = []

        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.60)
            print(match)
            if match[0]:
                name = "mayun"
            elif match[1]:
                name = "jobs"
            elif match[2]:
                name = "ljy"
            # elif match[3]:
            #     name = "lcy"
            # elif match[4]:
            #     name = "lhk"
            # elif match[5]:
            #     name = "lgy"
            # elif match[6]:
            #     name = "lz"
            else:
                name = "unknown"
            print(name)


            for known_face in known_faces:

                feature_1=np.array(face_encoding)
                feature_2=np.array(known_face)
                similarity=(np.dot(feature_1,feature_2)) / (np.sqrt(np.dot(feature_1,feature_1)) * np.sqrt(np.dot(feature_2 , feature_2)) )
                similarities.append(similarity)
            max_sim = str(max(similarities)*100)[:5]+'%'
            print(max_sim)


            #face_distances=face_recognition.face_distance(known_faces,face_encoding)
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, top-35), (right+60, top), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
        if name == "unknown":
            cv2.putText(frame, "unrecognized", (left + 6, top - 10), font, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, format(max_sim), (left + 6, top - 10), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


