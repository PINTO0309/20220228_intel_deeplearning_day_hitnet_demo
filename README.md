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
1. [Procurement of original model .pb / .pb オリジナルモデル.pbの調達](#4-1-procurement-of-original-model--pb-オリジナルモデルpbの調達)
2. [Convert .pb to saved_model / .pbをsaved_modelに変換](#4-2-convert-pb-to-saved_model--pbをsaved_modelに変換)
3. [Convert saved_model to ONNX / saved_modelをONNXに変換](#4-3-convert-saved_model-to-onnx--saved_modelをonnxに変換)
4. [Building OpenVINO / OpenVINOのビルド](#4-4-building-openvino--openvinoのビルド)
5. [Convert ONNX to OpenVINO IR / ONNXをOpenVINO IRへ変換](#4-5-convert-onnx-to-openvino-ir--onnxをopenvino-irへ変換)
6. [HITNet's OpenVINO demo / HITNetのOpenVINOデモ](#4-6-hitnets-openvino-demo--hitnetのopenvinoデモ)
7. [HITNet's ONNX demo / HITNetのONNXデモ](#4-7-hitnets-onnx-demo--hitnetのonnxデモ)

## 4. Procedure / 手順
### 4-1. Procurement of original model / .pb オリジナルモデル.pbの調達
[↥ Back to top](#4-procedure--手順)
### 4-2. Convert .pb to saved_model / .pbをsaved_modelに変換
[↥ Back to top](#4-procedure--手順)
### 4-3. Convert saved_model to ONNX / saved_modelをONNXに変換
[↥ Back to top](#4-procedure--手順)
### 4-4. Building OpenVINO / OpenVINOのビルド
[↥ Back to top](#4-procedure--手順)
### 4-5. Convert ONNX to OpenVINO IR / ONNXをOpenVINO IRへ変換
[↥ Back to top](#4-procedure--手順)
### 4-6. HITNet's OpenVINO demo / HITNetのOpenVINOデモ
[↥ Back to top](#4-procedure--手順)
### 4-7. HITNet's ONNX demo / HITNetのONNXデモ
[↥ Back to top](#4-procedure--手順)
## 5. Acknowledgements / 謝辞
- Intel Team: https://github.com/openvinotoolkit/openvino/issues/7379
- **[ibaiGorordo](https://github.com/ibaiGorordo)**: https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation
