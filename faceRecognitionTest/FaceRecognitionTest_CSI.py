import time, libcamera
from picamera2 import Picamera2, Preview
import cv2


SIZE = (640, 640)


picam = Picamera2()

camera_config = picam.create_preview_configuration(main={"size": SIZE, "format": 'RGB888'})
picam.configure(camera_config)

picam.start()
frame =  picam.capture_array()

cv2.imwrite("csi_test.jpg", frame)
# while True:
#     frame =  picam.capture_array()
#     cv2.imwrite("csi_test.jpg", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

picam.stop()
cv2.destroyAllWindows()