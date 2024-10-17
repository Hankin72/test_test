import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort
from FaceUtils import *
from FaceDatabase import load_face_database, DEFAULT_FILENAME_DB

FACE_DATABASE = load_face_database(filename=DEFAULT_FILENAME_DB)


class FaceRecognizer:
    def __init__(self,
                 model_name='buffalo_s',
                 allowed_modules=ALLOW_MODULES,
                 providers=None,
                 ctx_id=0,
                 det_size=(640, 640)):

        self.providers = providers if providers else ort.get_available_providers()
        print('Available providers:', self.providers)
        self.app = FaceAnalysis(name=model_name, allowed_modules=allowed_modules, providers=self.providers, root="./local_model")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.threshold = .50

    @staticmethod
    def cosine_similarity(a, b):
        """
        计算两个向量之间的余弦相似度
        返回:
        - similarity: 余弦相似度 (介于 -1 和 1 之间)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def recognize_face(self, embedding):
        if embedding is None:
            print("Error: 提取的 embedding 为 None")
            return 'Unknown'

        best_match = None
        highest_similarity = -1
        for name, embeddings in FACE_DATABASE.items():
            for known_embedding in embeddings:
                if known_embedding is None:
                    print(f"Warning: {name} 的 embedding 为 None，跳过")
                    continue
                similarity = self.cosine_similarity(embedding, known_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name
        if highest_similarity >= self.threshold:
            return best_match, highest_similarity
        else:
            return 'Unknown', highest_similarity

    def find_someone(self, frame, target_name):
        """
        在给定的帧中搜索目标姓名。
        如果找到，返回人脸坐标和图像尺寸。
        """

        faces = self.app.get(frame)
        image_height, image_width = frame.shape[:2]

        for face in faces:
            embedding = face.embedding
            if embedding is None:
                continue

            name, similarity = self.recognize_face(embedding)
            if name == target_name:
                bbox = face.bbox.astype(int)
                face_left, face_top, face_right, face_bottom = bbox
                return [face_left, face_top, face_right, face_bottom, image_width, image_height]
        return None  # 如果未找到指定的人，返回 None


class FaceRecognitionApp:
    def __init__(self,
                 model_name='buffalo_s',
                 allowed_modules=ALLOW_MODULES,
                 providers=None,
                 ctx_id=0,
                 det_size=(640, 640),
                 camera_index=0,
                 flip=True):
        """
        初始化人脸识别应用程序。
        """
        self.flip = flip
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")

        self.face_recognizer = FaceRecognizer(
            model_name=model_name,
            allowed_modules=allowed_modules,
            providers=providers,
            ctx_id=ctx_id,
            det_size=det_size
        )

    def run(self, target_name):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法接收帧（流结束？）。正在退出...")
                break

            if self.flip:
                frame = cv2.flip(frame, 1)

            result = self.face_recognizer.find_someone(frame, target_name)
            if result:

                face_left, face_top, face_right, face_bottom, image_width, image_height = result
                print(face_left, face_top, face_right, face_bottom, image_width, image_height)
                # 在找到的人脸周围绘制边框
                cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), COLOR_RED, 2)
                cv2.putText(frame, target_name, (face_left, face_top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)


            cv2.imshow('hello ', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    target_person_name = "haojin"  # 替换为你要寻找的人的名字
    app = FaceRecognitionApp(
        model_name='buffalo_s',
        allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
        ctx_id=-1,  # 使用 CPU
        det_size=(320, 320),
        camera_index=0,
        flip=True
    )
    app.run(target_person_name)
