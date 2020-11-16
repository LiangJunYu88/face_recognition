import os
import os.path
import pickle

pthmodel_path = "C:/Users/Administrator/Desktop/face.evoLVe.PyTorch-master/model/ms1m-ir50/backbone_ir50_ms1m_epoch120.pth"
with open(pthmodel_path, 'wb') as f:  # 'wb'打开f
    pickle.dump(pthmodel_path, f)  # 序列化knn_clf，并将模型保存到f中，
