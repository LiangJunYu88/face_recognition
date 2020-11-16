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
from numpy import *
import dlib
import matplotlib.image as mpimg
import numpy as np

def get_img_pairs_list(pairs_txt_path, img_path):
    """ 指定图片组合及其所在文件，返回各图片对的绝对路径
        Args:
            pairs_txt_path：图片pairs文件，里面是6000对图片名字的组合
            img_path：图片所在文件夹
        return:
            img_pairs_list：深度为2的list，每一个二级list存放的是一对图片的绝对路径
    """
    file = open(pairs_txt_path, 'r')
    img_pairs_list, labels = [], []
    while 1:
        img_pairs = []
        line = file.readline().replace('\n', '')
        if line == '':
            break
        line_list = line.split('\t')
        if len(line_list) == 3:
            # 图片路径示例：
            # 'C:\Users\thinkpad1\Desktop\image_set\lfw_funneled\Tina_Fey\Tina_Fey_0001.jpg'
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[1])[-4:] + '.jpg')
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[2])[-4:] + '.jpg')
            labels.append(1)
        elif len(line_list) == 4:
            img_pairs.append(
                img_path + '\\' + line_list[0] + '\\' + line_list[0] + '_' + ('000' + line_list[1])[-4:] + '.jpg')
            img_pairs.append(
                img_path + '\\' + line_list[2] + '\\' + line_list[2] + '_' + ('000' + line_list[3])[-4:] + '.jpg')
            labels.append(0)
        else:
            continue

        img_pairs_list.append(img_pairs)
    return img_pairs_list

def acc_test(knn_clf=None, model_path=None, distance_threshold=0.60):
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # 加载KNN模型
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)


    img_pairs_list = get_img_pairs_list(pairs_txt_path="C:\\Users\\Administrator\\Desktop\\face_recognition-master\\examples\\knn_examples\\pairs.txt", img_path="C:\\Users\\Administrator\\Desktop\\face_recognition-master\\examples\\knn_examples\\train-LFW")

    #print(len(img_pairs_list))  #[[pic1,pic2],[]...]
    #
    # #img_locations_pairs = []  # 样本是： 正300  负300  以此循环10次 共正3000 负3000
    results = []
    img_encoding1s = []
    img_encoding2s = []
    for i in range(5951):
        img_pairs = img_pairs_list[i]
        #print(img_pairs[0])
        #print(img_pairs[1])
        #img1 = mpimg.imread(str(img_pairs[0]))
        img1 = face_recognition.load_image_file(str(img_pairs[0]))
        img2 = face_recognition.load_image_file(str(img_pairs[1]))

        #img1_location = face_recognition.face_locations(img1)
        #img2_location = face_recognition.face_locations(img2)

        img_encoding1 = face_recognition.face_encodings(img1)[0]
        img_encoding1s.append(img_encoding1)
        #print("111")
        print(i)
        img_encoding2 = face_recognition.face_encodings(img2)[0]
        img_encoding2s.append(img_encoding2)
        #print("222")

        #knn_clf.predict(img_encoding1, img_encoding2)

    for i in range(5951):
        result = face_recognition.compare_faces([img_encoding1s[i]], img_encoding2s[i])
        results.append(result)
    print(results)

    # TP = 0
    # TN = 0
    # FP = 0
    # FN = 0
    # for i in range(0, 299):
    #     if results[i] == True:
    #         TP += 1   #285
    #     else:
    #         FN += 1   #13
    # for i in range(299, 596):
    #     if results[i] == True:
    #         FP += 1    #0
    #     else:
    #         TN += 1   #297
    # for i in range(596, 895):
    #     if results[i] == True:
    #         TP += 1
    #     else:
    #         FN += 1
    # for i in range(895, 1190):
    #     if results[i] == True:
    #         FP += 1
    #     else:
    #         TN += 1
    # for i in range(1190, 1489):
    #     if results[i] == True:
    #         TP += 1
    #     else:
    #         FN += 1
    # for i in range(1489, 1785):
    #     if results[i] == True:
    #         FP += 1
    #     else:
    #         TN += 1
    # for i in range(1785, 1786):
    #     if results[i] == True:
    #         TP += 1
    #     else:
    #         FN += 1
    # for i in range(2084, 2384):
    #     if results[i] == True:
    #         FP += 1
    #     else:
    #         TN += 1
    # for i in range(2384, 2684):
    #     if results[i] == True:
    #         TP += 1
    #     else:
    #         FN += 1

    # precesion = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # TNR = TN / (FP + TN)
    # FPR = FP / (FP + TN)
    #
    # ACC = (TP + TN) / (TP + TN + FP + FN)
    # print("TP= " + str(TP) + " TN= " + str(TN) + " FP= " + str(FP) + " FN= " + str(FN))
    # print("precesion=" + str(precesion))
    # print("recall=" + str(recall))
    # print("TNR=" + str(TNR))
    # print("FPR=" + str(FPR))
    # print("ACC=" + str(ACC))
# def face_verification(img_pairs_list):
#     model = './model/'
#     model_facenet = r'XXX\XXX\20180402-114759.pb'  # 模型在你电脑中的路径
#     # mtcnn相关参数
#     minsize = 40
#     threshold = [0.4, 0.5, 0.6]  # pnet、rnet、onet三个网络输出人脸的阈值，大于阈值则保留，小于阈值则丢弃
#     factor = 0.709  # scale factor
#
#     # 创建mtcnn网络
#     with tf.Graph().as_default():
#         sess = tf.Session()
#         with sess.as_default():
#             pnet, rnet, onet = detect_face.create_mtcnn(sess, model)
#
#     margin = 44
#     image_size = 160
#
#     with tf.Graph().as_default():
#
#         with tf.Session() as sess:
#
#             # 根据模型文件载入模型
#             facenet.load_model(model_facenet)
#             # 得到输入、输出等张量
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#
#             # 设置可视化进度条相关参数
#             jd = '\r   %2d%%\t [%s%s]'
#             bar_num_total = 50
#             total_num = len(img_pairs_list)
#             result, dist = [], []
#
#             for i in range(len(img_pairs_list)):
#
#                 # 画进度条
#                 if i % round(total_num / bar_num_total) == 0 or i == total_num - 1:
#                     bar_num_alright = round(bar_num_total * i / total_num)
#                     alright = '#' * bar_num_alright
#                     not_alright = '□' * (bar_num_total - bar_num_alright)
#                     percent = (bar_num_alright / bar_num_total) * 100
#                     print(jd % (percent, alright, not_alright), end='')
#
#                 # 读取一对人脸图像
#                 img_pairs = img_pairs_list[i]
#                 img_list = []
#                 img1 = cv2.imread(img_pairs[0])
#                 img2 = cv2.imread(img_pairs[1])
#
#                 img_size1 = np.asarray(img1.shape)[0:2]
#                 img_size2 = np.asarray(img2.shape)[0:2]
#
#                 # 检测该对图像中的人脸
#                 bounding_box1, _1 = detect_face.detect_face(img1, minsize, pnet, rnet, onet, threshold, factor)
#                 bounding_box2, _2 = detect_face.detect_face(img2, minsize, pnet, rnet, onet, threshold, factor)
#
#                 # 未检测到人脸，则将结果标为-1，后续计算准确率时排除
#                 if len(bounding_box1) < 1 or len(bounding_box2) < 1:
#                     result.append(-1)
#                     dist.append(-1)
#                     continue
#
#                 # 将图片1加入img_list
#                 det = np.squeeze(bounding_box1[0, 0:4])
#                 bb = np.zeros(4, dtype=np.int32)
#                 bb[0] = np.maximum(det[0] - margin / 2, 0)
#                 bb[1] = np.maximum(det[1] - margin / 2, 0)
#                 bb[2] = np.minimum(det[2] + margin / 2, img_size1[1])
#                 bb[3] = np.minimum(det[3] + margin / 2, img_size1[0])
#                 cropped = img1[bb[1]:bb[3], bb[0]:bb[2], :]
#                 aligned = cv2.resize(cropped, (image_size, image_size))
#                 prewhitened = facenet.prewhiten(aligned)
#                 img_list.append(prewhitened)
#
#                 # 将图片2加入img_list
#                 det = np.squeeze(bounding_box2[0, 0:4])
#                 bb = np.zeros(4, dtype=np.int32)
#                 bb[0] = np.maximum(det[0] - margin / 2, 0)
#                 bb[1] = np.maximum(det[1] - margin / 2, 0)
#                 bb[2] = np.minimum(det[2] + margin / 2, img_size2[1])
#                 bb[3] = np.minimum(det[3] + margin / 2, img_size2[0])
#                 cropped = img2[bb[1]:bb[3], bb[0]:bb[2], :]
#                 aligned = cv2.resize(cropped, (image_size, image_size))
#                 prewhitened = facenet.prewhiten(aligned)
#                 img_list.append(prewhitened)
#
#                 images = np.stack(img_list)
#
#                 # 将两个人脸转化为512维的向量
#                 feed_dict = {images_placeholder: images, phase_train_placeholder: False}
#                 emb = sess.run(embeddings, feed_dict=feed_dict)
#
#                 # 计算两个人脸向量的距离
#                 ed = np.sqrt(np.sum(np.square(np.subtract(emb[0], emb[1]))))
#                 dist.append(ed)
#                 # 根据得出的人脸间的距离，判断是否属于同一个人
#                 if ed <= 1.1:
#                     result.append(1)
#                 else:
#                     result.append(0)
#     return result, dist

if __name__ == "__main__":
    #get_img_pairs_list(pairs_txt_path="knn_examples/pairs.txt", img_path="knn_examples/train-LFW")
    acc_test(model_path="trained_knn_model3.clf")

