import cv2


def hhhh():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("无法接收帧，请退出")
            break

        # 显示帧
        cv2.imshow('CSI Camera', frame)

        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    hhhh()
