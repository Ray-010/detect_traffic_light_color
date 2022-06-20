# 方針

BGR画像をHSV色空間に変換して色を検出する
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

H: Hue 色相, 0-360

S: Saturation 彩度, 0-255

V: Value 明度, 0-255

赤信号
- H, 0-8, 
- S, 144-194
- V, 153-252

青信号
- H, 84-104
- S, 163-203
- V, 108-155

赤信号と青信号の2値画像から安定して、信号機の赤と青の領域を抽出するために膨張と収縮の処理によってノイズを除去.



> 参考：
> https://ohta-lab.inf.gunma-u.ac.jp/ailwiki/index.php?%E3%81%A4%E3%81%8F%E3%81%B0%E3%83%81%E3%83%A3%E3%83%AC%E3%83%B3%E3%82%B8%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E4%BF%A1%E5%8F%B7%E8%AA%8D%E8%AD%98%E6%89%8B%E6%B3%95%E3%81%AE%E7%A0%94%E7%A9%B6



# HSV 色空間 試し
実験1
赤色
```python
lower_color = np.array([0, 230, 100])
upper_color = np.array([30, 255, 255])

frame_mask1 = cv2.inRange(hsv, lower_color, upper_color)

lower_color = np.array([150, 230, 100])
upper_color = np.array([180, 255, 255])
frame_mask2 = cv2.inRange(hsv, lower_color, upper_color)

frame_mask = frame_mask1 + frame_mask2
```

実験2
```python
# red
lower_color = np.array([0, 230, 230])
upper_color = np.array([30, 255, 255])
frame_mask1 = cv2.inRange(hsv, lower_color, upper_color)

lower_color = np.array([150, 230, 230])
upper_color = np.array([179, 255, 255])
frame_mask2 = cv2.inRange(hsv, lower_color, upper_color)

# green
lower_color = np.array([30, 150, 150])
upper_color = np.array([90, 255, 255])
frame_mask3 = cv2.inRange(hsv, lower_color, upper_color)
```
