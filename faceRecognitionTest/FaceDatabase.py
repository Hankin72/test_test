import os.path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from FaceUtils import *

DEFAULT_FILENAME_DB = 'my_face_database.npy'


class FaceDatabase:
    def __init__(self):
        """
        初始化人脸数据库，数据库以字典形式存储姓名和对应的特征向量。
        """
        self.known_faces_db = {}

    def add_face(self, name, embedding):
        if name in self.known_faces_db:
            self.known_faces_db[name].append(embedding)  # 如果该人名已经存在，则添加新特征
        else:
            self.known_faces_db[name] = [embedding]  # 否则新建一个列表
        print(f">>>>> Add face to db: {name}")

    def save_to_file(self, filename=DEFAULT_FILENAME_DB):
        """
        将人脸数据库保存到文件
        - filename: 保存文件的路径 (默认为 {DEFAULT_FILENAME_DB})
        """
        np.save(filename, self.known_faces_db)
        print(f">>>> face db saved to file: {filename}")

    def load_from_file(self, filename=DEFAULT_FILENAME_DB):
        """
        加载人脸数据库
        - filename: 文件路径
        """
        self.known_faces_db = np.load(filename, allow_pickle=True).item()
        print(f"已从文件 {filename} 加载人脸数据库")
        return self.known_faces_db


class FaceDetector:
    def __init__(self, model_name=USE_DEFAULT_MODEL):
        self.app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition', 'landmark_2d_106'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))  # 使用 CPU

    def detect_and_extract(self, image_path):
        """
        从照片中检测人脸并提取特征向量
        参数:
        - image_path: 照片路径
        返回:
        - faces: 检测到的人脸对象列表 (包含 bounding box, embedding 等)
        """
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if len(faces) == 0:
            print("没有检测到人脸")
        return faces


def get_image_paths(directory):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths

def save_face_database(image_paths=[], person_name="", filename=DEFAULT_FILENAME_DB):
    """
    将多个图片的 embedding 存入人脸数据库。
    参数:
    - image_paths: 图片路径列表
    - person_name: 对应的用户姓名
    """
    face_db = FaceDatabase()

    if os.path.exists(filename):
        face_db.load_from_file()
    else:
        print(f"{filename} not exists, create a face-db")
    detector = FaceDetector()

    for image_path in image_paths:
        print(">>>>> detector image_path: " + image_path)
        faces = detector.detect_and_extract(image_path)
        if faces:
            for face in faces:
                # 将每张图片中的人脸特征存入数据库
                embedding = face.embedding
                face_db.add_face(person_name, embedding)

    face_db.save_to_file()


def load_face_database(filename=""):
    if os.path.exists(filename):
        face_db = FaceDatabase()
        return face_db.load_from_file(filename=filename)
    return None


if __name__ == '__main__':
    # path = "./sample/U"
    image_paths = get_image_paths("./sample/haojin")

    save_face_database(image_paths=image_paths, person_name="haojin", filename=DEFAULT_FILENAME_DB)
