# 用語
マスク：対象の特定の部位を処理や加工から保護する多いの役割を果たすものをマスクということが多い

グラフィック処理のマスク：
背景画像の上に別の画像を重ねる際に、背景を透過させたい領域が塗りつぶされないように保護する役割を果たす.
背景画像に対してマスクと画像を連続でビット演算させることにより、両者が合成された画像を得ることができる


# 検出の仕方
> 参考：OpenCVで色認識,色検出
> 
> https://data-analysis-stats.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/opencv-python%E3%81%A7%E3%81%AE%E8%89%B2%E8%AA%8D%E8%AD%98%E3%83%BB%E8%89%B2%E6%A4%9C%E5%87%BA/


InRange
```python
InRange(src, lower, upper, dst)
src: 入力画像
lower: 下界(その値を含む)を表す配列
upper: 上界(その値を含まない)を表す配列
dst: 出力画像
```
例：
```python
from IPython.display import Image
import cv2
import numpy as np

Image(filename="./img/ball.jpg")
img = cv2.imread("./img/ball.jpg")
# 色基準で2値化
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 色範囲指定
lower_color = np.array([20, 80, 10])
upper_color = np.array([50, 255, 255])
mask = cv2.inRange(hsv, lower_color, upper_color)
output = cv2.bitwise_and(hsv, hsv, mask = mask)

cv2.imwrite("./img/mask_img.jpg", output)
Image(filename="./img/mask_img.jpg")
```

# 粒子解析, 色を検出し数を数える
```python
image_file = "./img/img.jpg"
Image(filename = image_file)

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

img = cv2.imread(image_file)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

red = cv2.inRange(hsv, np.array([145, 70, 0]), np.array([180, 255, 255]))
yellow = cv2.inRange(hsv, np.array([10, 80, 0]), np.array([50, 255, 255]))
blue = cv2.inRange(hsv, np.array([108, 121, 0]), np.array([120, 255, 255]))

# 白だけゴミがあるので収縮演算
kernel = cv2.getStructuringElemtnt(cv2.MORPH_RECT, (3,3))

bin_imgs = {"red": red, "yellow": yellow, "blud": blue,}

fig, axes_list = plt.subplots(3,1,figsize=(10, 18))

for ax, (label, bin_img) in zip(axes_list.ravel(), bin_imgs.items()):
    ax.axis("off")
    ax.set_title(label)
    ax.imshow(bin_img, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(figsize=(12,10))
ax.axis("off")
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 輪郭検出し、数を求める
for label, bin_img in bin_imgs.items():
    _, contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE 
    )
    contours = list(filter(lambda cnt: len(cnt) > 1, contours))
    count = len(contours)

    print("color = {} conunt: {}".format(label, count))

    # 描画
    for cnt in contours:
        cnt = np.squeeze(cnt, axis=1)
        ax.add_patch(Polygon(cnt, fill=None, lw=2., color=label))
plt.show()
```


# 特定の色を検出する
> Python + OpenCV 特定の色を検出するプログラム
> 
> https://craft-gogo.com/python-opencv-color-detection/

```python
import cv2 as cv
import numpy as np

#画像データの読み込み
img = cv.imread("sample.jpg")

#BGR色空間からHSV色空間への変換
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#色検出しきい値の設定
lower = np.array([90,64,0])
upper = np.array([150,255,255])

#色検出しきい値範囲内の色を抽出するマスクを作成
frame_mask = cv.inRange(hsv, lower, upper)

#論理演算で色検出
dst = cv.bitwise_and(img, img, mask=frame_mask)

cv.imshow("img", dst)

if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()
```

# 解説
cv.imread() はBGR色空間で読み込まれる
HSV空間の方が色検出閾値を指定しやすいのでBGRからHSVに変換する
```python
img = cv.imread("sample.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```
hsv色空間：
色相(Hue), 彩度(Saturation), 明度(Value・Brightness)

cv.inRange()で検出する色範囲を設定し、マスクをかける. inRange()は指定したい範囲の色を255, それ以外の範囲の色を0として二値化する. 

cv.bitwise_and()でマスクをかける

### 〇 赤色検出には工夫が必要
赤は0-30(0-60度), 150-180(300-360度), 色範囲がに条件となる
```python
#色検出しきい値の設定
lower = np.array([0,64,0])
upper = np.array([30,255,255])

#色検出しきい値範囲内の色を抽出するマスクを作成
frame_mask1 = cv.inRange(hsv, lower, upper)

#色検出しきい値の設定
lower = np.array([150,64,0])
upper = np.array([180,255,255])

#色検出しきい値範囲内の色を抽出するマスクを作成
frame_mask2 = cv.inRange(hsv, lower, upper)

frame_mask = frame_mask1 + frame_mask2
```


# S(彩度), V(明度)の値の調整, 応用例
> 参考：赤・緑・青色の色検出, HSV色空間
> 
> https://algorithm.joho.info/programming/python/opencv-color-detection/

対象とする色以外も抽出してしまった場合は、S,Vの値を調整することで改善できる.

Sは彩度を示す値, 下限を下げるほど薄い色も検出してしまう

またVは明るさを示す値で、これも下限を下げると黒っぽい色も検出しやすくなる
```python

```

