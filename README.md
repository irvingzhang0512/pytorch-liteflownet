# pytorch-liteflownet Video Demo

基于 [sniklaus/pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet) 的 Demo 二次开发。原始 README 可以参考[这里](origin_README.md)。

+ 要运行本库，需要安装：

```
cupy>=5.0.0
numpy>=1.15.0
Pillow>=5.0.0
torch>=1.6.0
```

+ 其中，安装 cupy 可以通过 `pip install cupy-cuda110` 这个格式，根据cuda版本改变后面的参数。
+ `run.py` 支持
  + 输入视频 `input_video_path`
  + 结果输出到 `output_video_path`
  + 通过 `show` 控制是否输出到 `cv2.imshow`
  + 可视化结果是原始图片与光流图片的concat结果（高concat）。
