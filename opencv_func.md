# 画像処理まとめ

## cvtColor()について
> 参考：
> https://kuroro.blog/python/7IFCPLA4DzV8nUTchKsb/

cvtColorは画像の色を変換するもの

RGBで全ての色を表すとは限らない. また基準となる色はコンピュータやプログラムによって異なる場合もある.
```python
cvtColor(画像ファイル, 変換の指定)

# 色空間, 指定の種類
cv2.COLOR_BGR2GRAY
cv2.COLOR_BGR2YCrCb
cv2.COLOR_BGR2HSV
cv2.COLOR_BGR2XYZ
```

