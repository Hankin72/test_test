from faceRecognitionTest.Pcb_car import Pcb_car2
import time
import threading

# 初始化舵机控制
car = Pcb_car2.Pcb_Car2()


def case01():
    car.Ctrl_Servo(1, 90)  # S1 舵机转动到90度
    time.sleep(0.5)

    car.Ctrl_Servo(2, 90)  # S2 舵机转动到90度
    time.sleep(0.5)

    car.Ctrl_Servo(1, 0)  # S1 舵机转动到0度
    time.sleep(0.5)

    car.Ctrl_Servo(2, 0)  # S2 舵机转动到0度
    time.sleep(0.5)

    car.Ctrl_Servo(1, 180)  # S1 舵机转动到180度
    time.sleep(0.5)

    car.Ctrl_Servo(2, 180)  # S2 舵机转动到180度
    time.sleep(0.5)

    car.Ctrl_Servo(1, 90)  # S1 舵机转动到90度
    time.sleep(0.5)

    car.Ctrl_Servo(2, 90)  # S2 舵机转动到90度
    time.sleep(0.5)

    car.Ctrl_Servo(1, 0)  # S1 舵机转动到0度
    time.sleep(0.5)

    car.Ctrl_Servo(2, 0)  # S2 舵机转动到0度
    time.sleep(0.5)

# 定义舵机旋转函数，指定舵机号和旋转范围
def rotate_servo_smoothly(servo_id, start_angle, end_angle, step_angle=1, delay=0.1):  # 增加延迟
    if start_angle < end_angle:
        for angle in range(start_angle, end_angle + 1, step_angle):
            car.Ctrl_Servo(servo_id, angle)
            time.sleep(delay)
    else:
        for angle in range(start_angle, end_angle - 1, -step_angle):
            car.Ctrl_Servo(servo_id, angle)
            time.sleep(delay)

    # 缓慢匀速回到90度位置
    current_angle = end_angle if end_angle <= 180 else 180
    return_to_initial(servo_id, current_angle, 90, step_angle, delay)


# 定义复位函数
def return_to_initial(servo_id, current_angle, target_angle, step_angle=1, delay=0.1):  # 增加延迟
    if current_angle < target_angle:
        for angle in range(current_angle, target_angle + 1, step_angle):
            car.Ctrl_Servo(servo_id, angle)
            time.sleep(delay)
    else:
        for angle in range(current_angle, target_angle - 1, -step_angle):
            car.Ctrl_Servo(servo_id, angle)
            time.sleep(delay)


# 控制两个舵机分别在不同的线程中执行旋转
def demo_two_servos_with_threads():
    servo1_thread = threading.Thread(target=rotate_servo_smoothly, args=(1, 0, 180, 1, 0.1))  # 减少步进角度
    servo2_thread = threading.Thread(target=rotate_servo_smoothly, args=(2, 0, 180, 1, 0.1))  # 减少步进角度

    servo1_thread.start()
    servo2_thread.start()

    servo1_thread.join()
    servo2_thread.join()


if __name__ == '__main__':
    case01()
    # 执行演示
    demo_two_servos_with_threads()