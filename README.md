## CVDL-Homework-1

### Objective

1. Camera calibration

   - Find chessboard corners
   - Find the intrinsic, extrinsic and distortion parameters of the camera
   - Use the parameters to undistort the image

2. Augmented reality

   - Transform real world coordinates to image coordinates
   - Draw given word on the image

3. Stereo Disparity Map

   - calculate the disparity map of the given stereo image pair
   - draw corresponding points on the image pair according to the disparity map

4. SIFT

   - Find the keypoints and descriptors of the given image
   - Match the keypoints of the given image pair

5. Train VGG19_bn on CIFAR10

### Setup

1. Install dependencies

   ```shell
   pip install -r requirements.txt --no-cache-dir
   ```

   Windows 要裝有 CUDA 的 Pytorch 再跑以下指令

   ```shell
   pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Check whether torch is using GPU

   ```shell
   python test_gpu.py
   ```

3. Put pretrained weights `vgg_16_bn_final.pth` in `./weights/`

### Run

```
python app.py
```

### Train

After training, the weights will be saved in `./weights/`
and the result chart will be saved in `./result.png`

```shell
python train.py
```
