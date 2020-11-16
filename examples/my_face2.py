# 1 加载库
import cv2
import numpy as np
import face_recognition

# 2 加载图片
jobs = cv2.imread("../known/jobs.jpg")
lincanyuan = cv2.imread("../known/lcy.jpg")
#leiguangyu=cv2.imread("../known/lgy.jpg")
#liaohuikang=cv2.imread("../known/lhk.jpg")
liangjunyu=cv2.imread("../known/ljy.jpg")
liuzhuang=cv2.imread("../known/lz.jpg")
mayun=cv2.imread("../known/mayun.jpg")

# 3 BGR 转 RGB
jobs_RGB = jobs[:, :, ::-1]
lincanyuan_RGB = lincanyuan[:, :, ::-1]
#leiguangyu_RGB = leiguangyu[:, :, ::-1]
#liaohuikang_RGB = liaohuikang[:, :, ::-1]
liangjunyu_RGB = liangjunyu[:, :, ::-1]
liuzhuang_RGB = liuzhuang[:, :, ::-1]
mayun_RGB = mayun[:, :, ::-1]

# 4 检测人脸
jobs_face = face_recognition.face_locations(jobs_RGB)
lincanyuan_face = face_recognition.face_locations(lincanyuan_RGB)
#leiguangyu_face = face_recognition.face_locations(leiguangyu_RGB)
#liaohuikang_face = face_recognition.face_locations(liaohuikang_RGB)
liangjunyu_face = face_recognition.face_locations(liangjunyu_RGB)
liuzhuang_face = face_recognition.face_locations(liuzhuang_RGB)
mayun_face = face_recognition.face_locations(mayun_RGB)

# 5 人脸特征编码
jobs_encoding = face_recognition.face_encodings(jobs_RGB, jobs_face)[0]
lincanyuan_encoding = face_recognition.face_encodings(lincanyuan_RGB, lincanyuan_face)[0]
#leiguangyu_encoding = face_recognition.face_encodings(leiguangyu_RGB, leiguangyu_face)[0]
#liaohuikang_encoding = face_recognition.face_encodings(liaohuikang_RGB, liaohuikang_face)[0]
liangjunyu_encoding = face_recognition.face_encodings(liangjunyu_RGB, liangjunyu_face)[0]
liuzhuang_encoding = face_recognition.face_encodings(liuzhuang_RGB, liuzhuang_face)[0]
mayun_encoding = face_recognition.face_encodings(mayun_RGB, mayun_face)[0]

# 6 把所有人脸放在一起，当做数据库使用
encodings = [jobs_encoding, lincanyuan_encoding,liangjunyu_encoding,liuzhuang_encoding,mayun_encoding]
names = ["jobs", "lin can yuan","liang jun yu","liu zhuang","ma yun"]

# 7 打开摄像头，读取视频流
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Camera Error !")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # 8 BGR 传 RGB
    frame_RGB = frame[:, :, ::-1]
    # 9 人脸检测
    faces_locations = face_recognition.face_locations(frame_RGB)
    # 10 人脸特征编码
    faces_encodings = face_recognition.face_encodings(frame_RGB, faces_locations)
    # 11 与数据库中的所有人脸进行匹配
    for (top, right, bottom, left), face_encoding in zip(faces_locations, faces_encodings):
        # 12 进行匹配
        matches = face_recognition.compare_faces(encodings, face_encoding)
        # 13 计算距离
        distances = face_recognition.face_distance(encodings, face_encoding)
        min_distance_index = np.argmin(distances)  # 0, 1, 2
        # 14 判断：如果匹配，获取名字
        name = "Unknown"
        if matches[min_distance_index]:
            name = names[min_distance_index]
        # 15 绘制人脸矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        # 16 绘制、显示对应人脸的名字
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), 3)
        # 17 显示名字
        cv2.putText(frame, name, (left + 10, bottom - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    # 18 显示整个效果
    cv2.imshow("face recognition", frame)
    # 19 判断 Q , 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 20 关闭所有资源
cap.release()
cv2.destroyAllWindows()
