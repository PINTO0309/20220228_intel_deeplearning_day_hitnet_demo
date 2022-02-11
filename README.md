# 20210228_intel_deeplearning_day_hitnet_demo
## 1. Overview / 概要
This is a demonstration of the steps to convert and infer HITNet, a stereo depth estimation model, using a custom build of OpenVINO.  
OpenVINOをカスタムビルドしてステレオ深度推定モデルのHITNetを変換し、推論するまでの手順のデモです。
![132152654-fd689269-537f-4ab1-87fc-b08169311cc7](https://user-images.githubusercontent.com/33194443/153523403-afd059d7-6c5b-496e-a418-4b432e2e0f58.gif)

## 2. Environment / 環境
- Ubuntu 20.04
- Docker 20.10.12, build e91ed57
- OpenVINO commit hash: e89db1c6de8eb551949330114d476a2a4be499ed 
- ONNX
## 3. Overall flow / 全体の流れ
In order to optimize the process as much as possible, the following processing flow is adopted.  
TensorFlow **`pb`** -> TensorFlow **`saved_model`** -> TensorFlow Lite **`tflite`** -> ONNX **`onnx`** -> OpenVINO IR **`xml/bin`**  

- [4-1. Procurement of original model .pb / .pb オリジナルモデル.pbの調達](#4-1-procurement-of-original-model--pb-オリジナルモデルpbの調達)
- [4-2. Convert .pb to saved_model / .pbをsaved_modelに変換](#4-2-convert-pb-to-saved_model--pbをsaved_modelに変換)
- [4-3. Convert saved_model to ONNX / saved_modelをONNXに変換](#4-3-convert-saved_model-to-onnx--saved_modelをonnxに変換)
- [4-4. Building OpenVINO / OpenVINOのビルド](#4-4-building-openvino--openvinoのビルド)
- [4-5. Convert ONNX to OpenVINO IR / ONNXをOpenVINO IRへ変換](#4-5-convert-onnx-to-openvino-ir--onnxをopenvino-irへ変換)
- [4-6. HITNet's OpenVINO demo / HITNetのOpenVINOデモ](#4-6-hitnets-openvino-demo--hitnetのopenvinoデモ)
- [4-7. HITNet's ONNX demo / HITNetのONNXデモ](#4-7-hitnets-onnx-demo--hitnetのonnxデモ)

## 4. Procedure / 手順
### 4-1. Procurement of original model / .pb オリジナルモデル.pbの調達
Download the official HITNet model published by Google Research [here](https://github.com/google-research/google-research/tree/master/hitnet). The file to be downloaded is a Protocol Buffers format file.  
[こちら](https://github.com/google-research/google-research/tree/master/hitnet)のGoogle Researchが公開しているHITNet公式モデルをダウンロードします。ダウンロードするファイルはProtocol Buffers形式のファイルです。
```bash
$ git clone https://github.com/PINTO0309/20210228_intel_deeplearning_day_hitnet_demo
$ cd 20210228_intel_deeplearning_day_hitnet_demo

# [1, ?, ?, 2], Grayscale image x2
$ wget https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/eth3d.pb
or
# [1, ?, ?, 6], RGB image x2
$ wget https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/flyingthings_finalpass_xl.pb
or
# [1, ?, ?, 6], RGB image x2
$ wget https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/middlebury_d400.pb
```
Use [Netron](https://netron.app/) to check the structure of the model. In the case of eth3d, two grayscale images of one channel are used as input. The name of the input is **`input`**.  
モデルの構造を確認するには、[Netron](https://netron.app/)を使用します。eth3dの場合、1チャンネルのグレースケール画像2枚を入力として使用します。入力の名前は **`input`** です。  
![image](https://user-images.githubusercontent.com/33194443/153540670-354a575c-2c0a-4f1f-b350-767bfb2b1e5d.png)  
The name of the output is **`reference_output_disparity`**.  
出力の名前は **`reference_output_disparity`** です。  
![image](https://user-images.githubusercontent.com/33194443/153558437-da09fa51-aa84-4ecc-b42c-4197b8c06281.png)  

For non-eth3d, the input is two 3-channel RGB images.  
eth3d以外のモデルの場合、入力は3チャンネルのRGB画像2枚です。  
![image](https://user-images.githubusercontent.com/33194443/153541985-7e3e580d-b659-4532-b0e3-28bc2fea0957.png)  
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-2. Convert .pb to saved_model / .pbをsaved_modelに変換
Start a Docker container with all the latest versions of the various major frameworks such as OpenVINO, TensorFlow, PyTorch, ONNX, etc. Note that the Docker Image is quite large, 24GB, since all the huge frameworks such as CUDA and TensorRT are also installed. Also, in order to launch the demo with GUI from within the Docker container, many launch options are specified, such as **`xhost`**, **`--gpus`**, **`-v`**, **`-e`**, **`--net`**, **`--privileged`**, etc., but they do not need to be specified if you do not want to use the GUI. If you want to know what kind of framework is implemented in a Docker container, please click [here](https://github.com/PINTO0309/openvino2tensorflow#1-environment).  
OpenVINOやTensorFlowやPyTorchやONNXなどの各種主要フレームワークの最新バージョンが全て導入されたDockerコンテナを起動します。CUDAやTensorRTなどの巨大なフレームワークも全てインストールされているため、Docker Imageは24GBとかなり大きいことに注意してください。また、Dockerコンテナの中からGUIを使用したデモを起動するため、**`xhost`**, **`--gpus`**, **`-v`**, **`-e`**, **`--net`**, **`--privileged`** などの多くの起動オプションを指定していますが、GUIを使用しない場合は指定不要です。どのようなフレームワークが導入されたDockerコンテナかを知りたい場合は [こちら](https://github.com/PINTO0309/openvino2tensorflow#1-environment) をご覧ください。
```bash
$ xhost +local: && \
docker run --gpus all -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ghcr.io/pinto0309/openvino2tensorflow:latest
```
```bash
$ MODEL=eth3d
or
$ MODEL=flyingthings_finalpass_xl
or
$ MODEL=middlebury_d400

$ pb_to_saved_model \
--pb_file_path ${MODEL}.pb \
--inputs input:0 \
--outputs reference_output_disparity:0 \
--model_output_path ${MODEL}/saved_model
```
A sample without GUI is shown below.  
GUIを使用しない場合のサンプルは下記のとおりです。
```bash
$ docker run -it --rm \
-v `pwd`:/home/user/workdir \
ghcr.io/pinto0309/openvino2tensorflow:latest
```
```bash
$ MODEL=eth3d
or
$ MODEL=flyingthings_finalpass_xl
or
$ MODEL=middlebury_d400

$ pb_to_saved_model \
--pb_file_path ${MODEL}.pb \
--inputs input:0 \
--outputs reference_output_disparity:0 \
--model_output_path ${MODEL}/saved_model
```
Let's check the shape of the generated **`saved_model`**, using the standard TensorFlow tool **`saved_model_cli`**.Of the input NHWC shape **`batch,height,width,channel`**, the height and width are undefined **`-1`**.  
生成された **`saved_model`** の形状を確認してみます。TensorFlowの標準ツール **`saved_model_cli`** を使用します。入力のNHWC形状 **`バッチ,高さ,幅,チャンネル`** のうち、高さと幅が未定義の **`-1`** となっています。  
```bash
$ saved_model_cli show --dir flyingthings_finalpass_xl/saved_model/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, 6)
        name: input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['reference_output_disparity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, -1)
        name: reference_output_disparity:0
  Method name is: tensorflow/serving/predict
```
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-3. Convert saved_model to ONNX / saved_modelをONNXに変換
The tool **`saved_model_to_tflite`** introduced in the Dokcer container is used to generate **`tflite`** from **`saved_model`**. The tool **`tensorflow-onnx`** can be used to generate **`onnx`** from **`saved_model`** immediately, but I will convert it once to **`tflite`** to make it as optimized as possible. The **`--input_shapes`** option can be used to fix undefined input shapes to a specified size.  
Dokcerコンテナに導入されている **`saved_model_to_tflite`** というツールを使用して **`saved_model`** から **`tflite`** を生成します。 公式の **`tensorflow-onnx`** というツールを使用すると **`saved_model`** から即座に **`onnx`** を生成することが可能ですが、なるべく最適化を行うためにあえて一度 **`tflite`** へ変換します。**`--input_shapes`** オプションを使用することで未定義の入力形状を指定のサイズへ固定することができます。  
```bash
$ H=480
$ W=640
$ saved_model_to_tflite \
--saved_model_dir_path ${MODEL}/saved_model \
--input_shapes [1,${H},${W},6] \
--model_output_dir_path ${MODEL}/saved_model_${H}x${W} \
--output_no_quant_float32_tflite
```
Check the input and output structure of the generated TFLite. At this point, TensorFlowLite's optimizer has already removed a large number of unnecessary operations or merged multiple operations into a clean and simple structure.  
生成されたTFLiteの入力と出力の構造を確認します。この時点ですでにTensorFlowLiteのオプティマイザによって不要なオペレーションが大量に削除されたり、あるいは複数のオペレーションが融合して綺麗でシンプルな構造に変換されています。  
![image](https://user-images.githubusercontent.com/33194443/153568491-4165d552-fa2e-4fa8-ae3e-aa288b7998cc.png)  
Next, convert **`tflite`** to **`onnx`**. I will use **`tensorflow-onnx`** here. **`--inputs-as-nchw input`** is an option to convert the shape of the input from **`NHWC`** to **`NCHW`**. Note that the onnx opset to be generated must be **`12`**.  
次に、**`tflite`** を **`onnx`** へ変換します。ここで **`tensorflow-onnx`** を使用します。**`--inputs-as-nchw input`** は入力の形状を **`NHWC`** から **`NCHW`** へ変換するためのオプションです。なお、生成するonnxのopsetは **`12`** を指定する必要があります。  
```bash
$ python -m tf2onnx.convert \
--opset 12 \
--inputs-as-nchw input \
--tflite ${MODEL}/saved_model_${H}x${W}/model_float32.tflite \
--output ${MODEL}/saved_model_${H}x${W}/model_float32.onnx
```
Redundant onnx files are generated with insufficient optimization and undefined input/output information for each operation.  
最適化が不十分で、なおかつ各オペレーションの入出力情報が未定義の冗長なonnxファイルが生成されます。  
![image](https://user-images.githubusercontent.com/33194443/153574176-32dd914a-47e9-46b6-9d9a-a2ebaa2c52c8.png)  
Uses **`onnx-simplifier`** to further optimize onnx files.  
**`onnx-simplifier`** を使用してonnxファイルをさらに最適化します。  
```bash
$ python -m onnxsim \
${MODEL}/saved_model_${H}x${W}/model_float32.onnx \
${MODEL}/saved_model_${H}x${W}/model_float32.onnx
```
The file size will increase, but the structure of the model will be optimized and inference performance will not be affected.  
ファイルサイズが肥大化しますが、モデルの構造は最適化されおり推論パフォーマンスに影響はありません。  
![image](https://user-images.githubusercontent.com/33194443/153575177-3b9c5b06-080b-45fc-bfbb-c4814b5ac00d.png)  
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-4. Building OpenVINO / OpenVINOのビルド
Since there are some issues with the current latest version of the OpenVINO model optimizer, we will build OpenVINO itself from the source code of the commits that have already resolved the [issues](github.com/openvinotoolkit/openvino/issues/7379).  
OpenVINOモデルオプティマイザの現行最新バージョンには一部問題があるため、問題箇所を解消済みのコミットのソースコードからOpenVINOそのものをビルドします。Intelのエンジニアとやりとりして解消いただいた問題点の内容が気になる方は [こちら](https://github.com/openvinotoolkit/openvino/issues/7379) をご覧ください。  
```bash
$ git clone https://github.com/openvinotoolkit/openvino \
&& cd openvino \
&& git checkout e89db1c6de8eb551949330114d476a2a4be499ed \
&& git submodule update --init --recursive

$ pip install pip --upgrade \
&& pip install Cython numpy setuptools wheel \
&& chmod +x scripts/submodule_update_with_gitee.sh \
&& ./scripts/submodule_update_with_gitee.sh \
&& chmod +x install_build_dependencies.sh \
&& ./install_build_dependencies.sh \
&& mkdir build \
&& cd build \
&& cmake \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_PYTHON=ON \
-DPYTHON_EXECUTABLE=`which python3` \
-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
-DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
-DENABLE_CLDNN=ON \
-DENABLE_WHEEL=ON .. \
&& make -j$(nproc)

cp wheels/*.whl ..
```
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-5. Convert ONNX to OpenVINO IR / ONNXをOpenVINO IRへ変換
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-6. HITNet's OpenVINO demo / HITNetのOpenVINOデモ
[↥ Back to top](#3-overall-flow--全体の流れ)
### 4-7. HITNet's ONNX demo / HITNetのONNXデモ
[↥ Back to top](#3-overall-flow--全体の流れ)
## 5. Acknowledgements / 謝辞
- **Intel Team**: https://github.com/openvinotoolkit/openvino/issues/7379
- **[NobuoTsukamoto](https://github.com/NobuoTsukamoto)**: https://github.com/NobuoTsukamoto/benchmarks
- **[ibaiGorordo](https://github.com/ibaiGorordo)**: https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation
