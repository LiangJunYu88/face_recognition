"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
import cv2
import time
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: 模型所保存的地址
    :param n_neighbors: 在训练时如果没有指定权重，则自动选择权重
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: knn_clf返回训练后的模型
    """
    X = []
    y = []
    # 循环遍历训练集中的每一个人
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 循环遍历当前训练集中的每个人
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image, model="cnn") # =1说明只有1张人脸，可用于训练

            if len(face_bounding_boxes) != 1:
                # 如果该训练集中没有人或者有很多人，则跳过该图像
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # 将图片中的人脸的编码加入到训练集中,并存入train_face_encodings用于后面的识别
                # for train_face_encoding in train_face_encodings:
                #     train_face_encoding=face_recognition.face_encodings(image)
                #     train_face_encodings.append(train_face_encoding)
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, model="large")[0])
                y.append(class_dir)

    # 确定KNN分类器中的权重
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))  #对训练集的长度开方，再四舍五入，取整
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # 建立并训练KNN训练集
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')

    knn_clf.fit(X, y)

    # 保存KNN分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f: # 'wb'打开f
            pickle.dump(knn_clf, f)  #序列化knn_clf，并将模型保存到f中，

    return knn_clf

#对for_rec里的图片编码
def encoding_known_face(known_faces_dir):
    known_faces = [] #用于存放for_rec里面的图片的编码
    known_locations = [] #用于存放for_rec里面的图片的location
    #known_face_names = [] #用于存放for_rec里面图片的名字
    len_of_for_rec = len(os.listdir(known_faces_dir))
    time_load = 0
    time_location_known_faces = 0
    for class_dir in os.listdir(known_faces_dir):
        if not os.path.isdir(os.path.join(known_faces_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(known_faces_dir, class_dir)):
            #start5 = time.clock()
            img = face_recognition.load_image_file(img_path)
            #end5 = time.clock()
            #time5 = end5 - start5
            #time_load += time5

            #start4 = time.clock()
            known_locations.append(face_recognition.face_locations(img,model="cnn")[0])
            #end4 = time.clock()
            #time4 = end4 - start4
            #time_location_known_faces += time4

            known_faces.append(face_recognition.face_encodings(img, known_locations)[0])
            #known_face_names.append(os.path.basename(os.path.splitext(img_path)[0])) #获取文件名存入face_names中
    #print("加载图片需要: " + str(time_load) + ' s')
    #print("定位人脸需要: " + str(time_location_known_faces) + ' s')

    #print(known_faces)
    #print(face_names)
    #print(len_of_for_rec)
    return known_faces, len_of_for_rec

name = "name"
def process_frame(known_faces_dir, knn_clf=None, model_path=None, distance_threshold=0.60):
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # 加载KNN模型
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)


    #video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #face_names = [] #用于存放识别出的对应的姓名
    face_locations = []
    #known_locations = []
    face_encodings = []

    start2 = time.clock()

    known_faces, len_of_for_rec = encoding_known_face(known_faces_dir)#用于存放数据库人脸的编码和姓名
    end2 = time.clock()
    encoding_time = end2 - start2
    print("编码时间需要：" + str(encoding_time) + ' s')
    #print(len_of_for_rec)
    process_this_frame = True
    global name
    threshold = 1/(0.75 * len_of_for_rec)
    #print(threshold)
    #print(known_face_names)
    #print(known_faces)


    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_this_frame:
            #start3 = time.clock()
            face_locations = face_recognition.face_locations(small_frame, model="cnn")  # 人脸定位
            face_encodings = face_recognition.face_encodings(small_frame, face_locations, model="small")  # 计算编码
            face_names = []
            max_sims = []
            #print(face_encodings)
            for face_encoding in face_encodings:
                #closest_distances = knn_clf.kneighbors(known_faces, n_neighbors=1)
                #print(closest_distances)

                #face_names.append(knn_clf.predict([face_encoding])) #存放匹配的脸的名字
                #print(knn_clf.predict([face_encoding]))  #返回人脸的标签，即:train文件夹对应的名字
                #start3 = time.clock()
                if np.max(knn_clf.predict_proba([face_encoding])) > threshold:  # 1 > threshold
                    #print(knn_clf.predict_proba([face_encoding]))  #[[0.0.1.0.0....]]
                    if knn_clf.predict([face_encoding]):
                        #print(knn_clf.predict([face_encoding]))   #['liang junyu']
                        name = str(knn_clf.predict([face_encoding]))  #['*'] 要处理[,'
                    else:
                        name = "unknown"
                else:
                    name = "unknown"
                #print(knn_clf.predict_proba([face_encoding]))  #返回人脸的概率，即：是train某个脸的概率
                face_names.append(name)
                similarities = []

                #face_distance = face_recognition.face_distance(known_faces, face_encoding)
                #print(max(face_distance))

                for known_face in known_faces:  #余弦相似度
                    feature_1 = np.array(face_encoding)
                    feature_2 = np.array(known_face)
                    similarity = (np.dot(feature_1, feature_2)) / (
                            np.sqrt(np.dot(feature_1, feature_1)) * np.sqrt(np.dot(feature_2, feature_2)))
                    similarities.append(similarity)
                #print(similarities)
                max_sim = str(max(similarities) * 100)[:5] + '%'
                #print(max_sim)
                max_sims.append(max_sim)

                #end3 = time.clock()
                #process_time = end3 - start3
                #print("人脸识别需要：" + str(process_time) + ' s')
                #print(max_sim)

        #process_this_frame = not process_this_frame

        #print(face_names)
        for (top, right, bottom, left), name, sim in zip(face_locations, face_names, max_sims):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            #name = face_names[-1]#name存放最新识别到的名字，放在列表最后面

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, top - 35), (right + 60, top), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            # 打印名字
            cv2.putText(frame, name.replace("'", '').replace("[",'').replace("]",'').title(), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # 打印相似度
            if name == "unknown":
                cv2.putText(frame, "unrecognized", (left + 6, top - 10), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, sim, (left + 6, top - 10), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
#     """
#     Recognizes faces in given image using a trained KNN classifier
#
#     :param X_img_path: 测试集的图片地址
#     :param knn_clf: 训练好的模型
#     :param model_path: 模型地址
#     :param distance_threshold: 给出当前测试图片中的人脸向量与模型中的距离
#     :return:图像中已识别面的名称和脸的定位列表: [(name, bounding box), ...].
#         For faces of unrecognized persons, the name 'unknown' will be returned.
#     """
#
#
#     if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
#         raise Exception("Invalid image path: {}".format(X_img_path))
#
#     if knn_clf is None and model_path is None:
#         raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
#
#     # 加载KNN模型
#     if knn_clf is None:
#         with open(model_path, 'rb') as f:
#             knn_clf = pickle.load(f)
#
#     # 加载图片文件夹以及人脸
#     X_img = face_recognition.load_image_file(X_img_path)
#     X_face_locations = face_recognition.face_locations(X_img)
#
#     # 如果图片中没有人脸，则返回空的结果集
#     if len(X_face_locations) == 0:
#         return []
#
#     # 找出测试集中的人脸编码
#     faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
#
#     # 利用KNN模型找出测试集中最匹配的人脸
#     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
#
#     # 预测类并删除不在阈值范围内的分类
#     return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
#
#
# def show_prediction_labels_on_image(img_path, predictions):
#     """
#     在图片给出标签并展示人脸.
#
#     :param img_path: path to image to be recognized
#     :param predictions: results of the predict function
#     :return:
#     """
#     pil_image = Image.open(img_path).convert("RGB")
#     draw = ImageDraw.Draw(pil_image)
#
#     for name, (top, right, bottom, left) in predictions:
#         # Draw a box around the face using the Pillow module
#         draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
#
#         # There's a bug in Pillow where it blows up with non-UTF-8 text
#         # when using the default bitmap font
#         name = name.encode("UTF-8")
#
#         # Draw a label with a name below the face
#         text_width, text_height = draw.textsize(name)
#         draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
#         draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
#
#     # Remove the drawing library from memory as per the Pillow docs
#     del draw
#
#     # Display the resulting image
#     pil_image.show()



if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    #start1 = time.clock()
    #classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    #end1 = time.clock()
    #train_time = end1 - start1
    #print("训练需要：" + str(train_time) + ' s')
    print("Training complete!")

    #start2 = time.clock()
    #encoding_known_face(known_faces_dir="knn_examples\\")
    #end2 = time.clock()
    #encoding_time = end2 - start2
    #print("编码需要：" + str(encoding_time) + ' s')
    #print("Encoding complete!")

    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    process_frame(known_faces_dir="./knn_examples/train", model_path="./trained_knn_model.clf")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    # for image_file in os.listdir("knn_examples\\test"):
    #     full_file_path = os.path.join("knn_examples\\test", image_file)
    #     #print("111")
    #     print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        #predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        #for name, (top, right, bottom, left) in predictions:
            #print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        #show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
