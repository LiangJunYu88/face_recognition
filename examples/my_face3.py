import face_recognition
import cv2
import numpy as np
import time

video_capture = cv2.VideoCapture(0)
# VideoCapture打开摄像头，0为笔记本内置摄像头，1为外USB摄像头，或写入视频路径
start1=time.clock()
mayun_img = face_recognition.load_image_file("../known/mayun.jpg")
jobs_img = face_recognition.load_image_file("../known/jobs.jpg")
ljy_img=face_recognition.load_image_file("../known/ljy.jpg")
end1=time.clock()
print("导入图片耗时：",end1-start1,"s")
# lcy_img=face_recognition.load_image_file("../known/lcy.jpg")
# lhk_img=face_recognition.load_image_file("../known/lhk.jpg")
# lgy_img=face_recognition.load_image_file("../known/lgy.jpg")
# lz_img=face_recognition.load_image_file("../known/lz.jpg")


#对图片进行编码
start2=time.clock()
mayun_face_encoding = face_recognition.face_encodings(mayun_img)[0]
jobs_face_encoding = face_recognition.face_encodings(jobs_img)[0]
ljy_face_encoding = face_recognition.face_encodings(ljy_img)[0]
end2=time.clock()
print("编码图片耗时：",end2-start2,"s")
# lcy_face_encoding = face_recognition.face_encodings(lcy_img)[0]
# lhk_face_encoding = face_recognition.face_encodings(lhk_img)[0]
# lgy_face_encoding = face_recognition.face_encodings(lgy_img)[0]
# lz_face_encoding = face_recognition.face_encodings(lz_img)[0]


#存为数组
known_faces=[
    mayun_face_encoding,
    jobs_face_encoding,
    ljy_face_encoding,
    # lcy_face_encoding,
    # lhk_face_encoding,
    # lgy_face_encoding,
    # lz_face_encoding,
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    #  video_capture.read()按帧读取视频，ret,frame是获video_capture.read()方法的两个返回值。
    #  其中ret是布尔值，如果读取帧是正确的则返回True，
    #  如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  #缩放图片
    # 对截取到的图像进行处理

    if process_this_frame:


        face_locations = face_recognition.face_locations(small_frame)  #人脸定位




        face_encodings = face_recognition.face_encodings(small_frame, face_locations) #计算编码



        face_names = []
        similarities=[]


        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([mayun_face_encoding, jobs_face_encoding,ljy_face_encoding],face_encoding,tolerance=0.60)

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

            #出现多张人脸会报错
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

video_capture.release()
cv2.destroyAllWindows()
