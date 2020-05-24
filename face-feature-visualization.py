import os
import sys

import cv2
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from openvino.inference_engine import IENetwork, IECore

# DL models for face detection and re-identification
model_det  = 'face-detection-adas-0001'             # data, Shape:[1, 3, 384, 672] / detection_out, Shape:[1, 1, 200, 7]
model_reid = 'face-reidentification-retail-0095'    # 0, Shape:[1, 3, 128, 128] / 658, Shape:[1, 256, 1, 1]
model_det  = 'intel/' + model_det  + '/FP16/' + model_det
model_reid = 'intel/' + model_reid + '/FP16/' + model_reid

def findFacesAndGenerateFeatures(img, ID, exec_det, net_det, exec_reid, net_reid):
    input_name_det  = next(iter(net_det.inputs))
    input_shape_det = net_det.inputs[input_name_det].shape
    out_name_det    = next(iter(net_det.outputs))
    out_shape_det   = net_det.outputs[out_name_det].shape

    input_name_reid  = next(iter(net_reid.inputs))
    input_shape_reid = net_reid.inputs[input_name_reid].shape
    out_name_reid    = next(iter(net_reid.outputs))
    out_shape_reid   = net_reid.outputs[out_name_reid].shape

    inBlob = cv2.resize(img, (input_shape_det[3], input_shape_det[2]))
    inBlob = inBlob.transpose((2,0,1))
    res_det = exec_det.infer(inputs={input_name_det:inBlob})[out_name_det][0][0]
    ret = []
    idx = 0
    for obj in res_det:
        if obj[2]>0.6:
            xmin = abs(int(obj[3] * img.shape[1]))
            ymin = abs(int(obj[4] * img.shape[0]))
            xmax = abs(int(obj[5] * img.shape[1]))
            ymax = abs(int(obj[6] * img.shape[0]))
            class_id = int(obj[1])
            face = img[ymin:ymax,xmin:xmax].copy()
            
            inFace=cv2.resize(face, (input_shape_reid[3], input_shape_reid[2]))
            inFace=inFace.transpose((2,0,1))
            inFace=inFace.reshape(input_shape_reid)
            res_reid = exec_reid.infer(inputs={input_name_reid: inFace})[out_name_reid]
            res_reid = res_reid.reshape((256))

            name = ID+str(idx)
            idx+=1
            ret.append({'name': name, 'img':face, 'feature': res_reid})
    return ret

def createDB(root_dir, exec_det, net_det, exec_reid, net_reid):
    face_db = []
    for dir in os.listdir(root_dir):
        face_db.append([])
        for file in os.listdir(os.path.join(root_dir, dir)):
            if '.jpg' in file or '.png' in file or '.bmp' in file:
                img = cv2.imread(os.path.join(root_dir, dir, file))
                ID = os.path.join(dir, file)
                faces = findFacesAndGenerateFeatures(img, ID, exec_det, net_det, exec_reid, net_reid)
                face_db[-1]+=faces
    return face_db

def displayFacesInDB(face_db):
    num_people = len(face_db)
    max_faces=0
    for person in face_db:
        if len(person)>max_faces:
            max_faces = len(person)
    plt.figure(figsize=(10,8))
    for row, person in enumerate(face_db):
        for col, face in enumerate(person):
            plt.subplot(num_people, max_faces, row*max_faces+col+1)
            img = cv2.cvtColor(face['img'], cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100,100))
            plt.imshow(img)
            plt.title(face['name'])
            plt.axis('off')
    plt.show()

def reduceDimensionAndDisplayScatterChart(face_db):
    vec=[]
    name=[]
    for person in face_db:
        for face in person:
            vec.append(np.array(face['feature'], dtype=np.float32))
            name.append(face['name'])

    tsne = TSNE(n_components=2, init='random', random_state=0, 
            n_iter=10000, n_iter_without_progress=200,
            learning_rate=50, perplexity=4)
    reduced = tsne.fit_transform(vec)

    fig, ax = plt.subplots(figsize=(5,5))
    cmap=plt.get_cmap('Dark2')
    col_idx = 0
    prev_name=os.path.dirname(name[0])
    for i in range(reduced.shape[0]):
        if os.path.dirname(name[i]) != prev_name:
            prev_name=os.path.dirname(name[i])
            col_idx+=1
        cval = cmap(col_idx)
        ax.scatter(reduced[i][0], reduced[i][1], marker='.', color=cval)
        ax.annotate(name[i], xy=(reduced[i][0], reduced[i][1]), color=cval)
    plt.show()


def main():
    ie = IECore()

    # Prep for face detection
    net_det  = ie.read_network(model_det+'.xml', model_det+'.bin')
    exec_net_det    = ie.load_network(net_det, 'CPU')

    # Preparation for face re-identification
    net_reid = ie.read_network(model_reid+".xml", model_reid+".bin")
    exec_net_reid    = ie.load_network(net_reid, 'CPU')

    db = createDB('face-db', exec_net_det, net_det, exec_net_reid, net_reid)
    displayFacesInDB(db)
    reduceDimensionAndDisplayScatterChart(db)

if __name__ == '__main__':
        sys.exit(main() or 0)
