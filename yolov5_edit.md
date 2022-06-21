# YoloV5 編集箇所

detect.py
166行目のnames[c]にlabelが格納されている

アノテーションについて
四角形の描画についてopencvかPillowを使うかで処理が分かれている

とりあえずPillowを想定
ImageDraw.Draw(image).rectangle()について
```python
from PIL import ImageDraw
self.draw = ImageDraw.Draw(self.im)

self.draw.rectangle(box, width=self.lw, outline=color)

# 座標指定
rectangle((左上のx座標, 左上のy座標), (右下のx座標, 右下のy座標))
rectangle((左上のx座標, 左上のy座標, 右下のx座標, 右下のy座標))

# detect.py, xyxyに座標がありそう ⇒ labelの座標だった
annotator.box_label(xyxy, label, color=colors(c, True))

# xyxyの中身
[tensor(147.), tensor(96.), tensor(343.), tensor(366.)]

# torch.Tensor
# pyTorchが用意している特殊な型, Tensor型
# Tensor型はGPUを使用して演算等が可能, 多数の機能を持つ

# annotationしている箇所
# plots.py 221行目
bs, _, h, w = images.shape
for i, im in enumerate(images):
    if i == max_subplots:  # if last batch has fewer images than we expect
        break
    x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
scale = max_size / ns / max(h, w)
if scale < 1:
    h = math.ceil(scale * h)
    w = math.ceil(scale * w)

# 座標はx, y, x+w, y+h
# rectangle((左上のx座標, 左上のy座標, 右下のx座標, 右下のy座標))
annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders

def rectangle(self, xy, fill=None, outline=None, width=1):
    # Add rectangle to image (PIL-only)
    self.draw.rectangle(xy, fill, outline, width)

# mosaic が画像
```



# cv2でのrectangle
自分の処理ではpillowではなくopencvを使っている

opencv rectangle 使い方
```python
cv2.rectangle(
    img,
    pt1=(10, 150),
    pt2=(125, 250),
    color=(0, 255, 0),
    thickness=3,
    lineType=cv2.LINE_4,
    shift=0
)
"""
pt1は左上の座標 (x, y)
pt2は右下の座標 (x, y)
"""
```

yoloでの処理
```python
cv2.rectangle(
    self.im, 
    p1, 
    p2, 
    color, 
    thickness=self.lw, 
    lineType=cv2.LINE_AA
)

"""
self.imはnumpy.ndarray形式
"""
```

つまり
```python
p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

# ここで色判定
# ROI抽出, [縦の範囲, 横の範囲]
# p1 = (x, y), 左上
# p2 = (x, y), 右下
detect_color_img = self.im[p1[1]:p2[1], p1[0]:p2[0]]

hsv = cv2.cvtColor(detect_color_img, cv2.COLOR_BGR2HSV)
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

# 論理演算で色検出
dst = cv2.bitwise_and(detect_color_img,detect_color_img, mask=frame_mask)

cv2.imshow("img", dst)
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()

# 描画処理
cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
```