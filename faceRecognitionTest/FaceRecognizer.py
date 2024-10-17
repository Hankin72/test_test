import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort
from picamera2 import Picamera2

from FaceUtils import *
from FaceDatabase import load_face_database, DEFAULT_FILENAME_DB

FACE_DATABASE = load_face_database(filename=DEFAULT_FILENAME_DB)
DRAW_LANDMARKS = False


class FaceRecognizer:
    def __init__(self,
                 model_name='buffalo_s',
                 allowed_modules=ALLOW_MODULES,
                 providers=None,
                 ctx_id=0,
                 det_size=(640, 640),
                 threshold=0.5):

        self.providers = providers if providers else ort.get_available_providers()
        print('Available providers:', self.providers)
        self.app = FaceAnalysis(name=model_name, allowed_modules=allowed_modules, providers=self.providers,
                                root=LOCAL_MODELS_PATH)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.threshold = threshold

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

            if DRAW_LANDMARKS:
                lmk = face.landmark_2d_106
                lmk = np.round(lmk).astype(int)

                draw_colored_landmarks(frame, lmk)

            name, similarity = self.recognize_face(embedding)
            bbox = face.bbox.astype(int)

            face_left, face_top, face_right, face_bottom = bbox
            bbox_res = [face_left, face_top, face_right, face_bottom, image_width, image_height]
            return name, bbox_res

        return None, None  # 如果未找到指定的人，返回 None


class FaceRecognitionApp:
    def __init__(self,
                 model_name='buffalo_s',
                 allowed_modules=ALLOW_MODULES,
                 providers=None,
                 ctx_id=0,
                 det_size=(640, 640),
                 camera_index=0,
                 flip=True,
                 isCsiCam=False,
                 threshold=0.50):
        """
        初始化人脸识别应用程序。
        """
        self.flip = flip
        self.isCsiCam = isCsiCam
        self.camera_index = camera_index

        if self.isCsiCam:
            pass
        else:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise IOError("无法打开摄像头")

        self.face_recognizer = FaceRecognizer(
            model_name=model_name,
            allowed_modules=allowed_modules,
            providers=providers,
            ctx_id=ctx_id,
            det_size=det_size,
            threshold=threshold
        )

    def run_usb_cam(self, target_name):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise IOError("无法打开USB摄像头, 请检查video index")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法接收帧。正在退出...")
                break

            if self.flip:
                frame = cv2.flip(frame, 1)

            self.find_and_draw_target_face(frame, target_name)

            cv2.imshow('usb CAM detect', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        self.release()

    def find_and_draw_target_face(self, frame, target_name):
        ret_name, result = self.face_recognizer.find_someone(frame, target_name)

        if ret_name and result:
            face_left, face_top, face_right, face_bottom, image_width, image_height = result

            if ret_name == target_name:
                print(target_name, "--->", result)
                # 在找到的人脸周围绘制边框
                cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), COLOR_RED, 2)
                cv2.putText(frame, target_name, (face_left, face_top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
            else:
                cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), COLOR_GREEN, 2)
                cv2.putText(frame, ret_name, (face_left, face_top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)

    def run_csi_cam(self, target_name):
        picam2 = Picamera2()
        # main={"size": (640, 480)}
        picam2.configure(picam2.create_preview_configuration())
        picam2.start()

        if not picam2.is_open:
            raise IOError("无法打开CSI摄像头")

        while True:
            frame = picam2.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.flip:
                frame_rgb = cv2.flip(frame_rgb, 1)

            self.find_and_draw_target_face(frame_rgb, target_name)

            cv2.imshow('Csi detect', frame_rgb)

            if cv2.waitKey(1) == ord('q'):
                break
        picam2.stop()
        self.release()

    def release(self):
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()


if __name__ == '__main__':
    target_person_name = "haojin"  # 替换为你要寻找的人的名字
    app = FaceRecognitionApp(
        model_name='buffalo_s',
        allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
        ctx_id=-1,  # 使用 CPU
        det_size=(320, 320),
        camera_index=0,
        flip=True,
        isCsiCam=False,
        threshold=0.5
    )
    app.run_usb_cam(target_person_name)
