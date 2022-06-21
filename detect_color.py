import cv2
import numpy as np

def detect_color(file_name, red1_range, red2_range, green_range, save_path_dir):
    img = cv2.imread("./images/" + file_name)

    # 2_19のみ画像を切り取る
    # (146, 96) (344, 366)
    # img = img[96:366, 146:344]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # red
    lower_color = np.array(red1_range[0])
    upper_color = np.array(red1_range[1])
    frame_mask1 = cv2.inRange(hsv, lower_color, upper_color)

    lower_color = np.array(red2_range[0])
    upper_color = np.array(red2_range[1])
    frame_mask2 = cv2.inRange(hsv, lower_color, upper_color)

    # green
    lower_color = np.array(green_range[0])
    upper_color = np.array(green_range[1])
    frame_mask3 = cv2.inRange(hsv, lower_color, upper_color)

    frame_mask = frame_mask1 + frame_mask2 + frame_mask3

    # 割合検出
    red_ratio = np.sum(frame_mask1+frame_mask2)
    green_ratio = np.sum(frame_mask3)
    if red_ratio > green_ratio:
        print("Red")
    elif green_ratio > red_ratio:
        print("Green")
    else:
        print("Doesn't detect color")

    # 論理演算で色検出
    dst = cv2.bitwise_and(img,img, mask=frame_mask)
    # cv2.imshow("img",frame_mask)
    cv2.imshow("img", dst)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

    cv2.imwrite(save_path_dir + file_name, dst)


file_names = [
    # 青
    "ped_lights_image_2_79.jpg",
    "ped_lights_image_2_19.jpg",
    "ped_lights_image_2_33.jpg",
    
    # 赤
    "ped_lights_image_2_71.jpg",
    "ped_lights_image_2_45.jpg",
    "ped_lights_image_2_20.jpg",

]

if __name__ == "__main__":
    # np.array([h色相, s彩度, v明度])
    red1_range = [[0, 150, 150], [30, 255, 255]]
    red2_range = [[150, 150, 150], [179, 255, 255]]
    green_range = [[30, 150, 150], [90, 255, 255]]
    save_path_dir = "./output3/"
    
    file_name = file_names[4]

    detect_color(
        file_name = file_name,
        red1_range = red1_range,
        red2_range = red2_range,
        green_range = green_range,
        save_path_dir = save_path_dir,
    )
