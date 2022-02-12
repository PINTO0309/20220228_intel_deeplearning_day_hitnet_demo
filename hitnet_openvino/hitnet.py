import time
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
from hitnet_openvino.utils_hitnet import *

drivingStereo_config = CameraConfig(0.546, 1000)

class HitNet():

    def __init__(self, model_path, model_type=ModelType.eth3d, camera_config=drivingStereo_config):
        self.fps = 0
        self.timeLastPrediction = time.time()
        self.frameCounter = 0
        self.model_type = model_type
        self.camera_config = camera_config
        # Initialize model
        self.initialize_model(model_path)

    def __call__(self, left_img, right_img):
        return self.estimate_disparity(left_img, right_img)

    def initialize_model(self, model_path):
        log.info('Configuring input and output blobs')
        self.ie = IECore()
        self.net = self.ie.read_network(
            model=model_path
        )
        # Get model info
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        log.info(f'self.net.input_info[self.input_blob].input_data.shape: {self.net.input_info[self.input_blob].input_data.shape}')
        log.info(f'self.net.outputs[self.out_blob].shape: {self.net.outputs[self.out_blob].shape}')
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape
        self.channes = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        log.info('Loading the model to the plugin')
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU',)
        log.info('Model loading complete')

    def estimate_disparity(self, left_img, right_img):
        # Update fps calculator
        self.updateFps()
        input_tensor = self.prepare_input(left_img, right_img)
        # Perform inference on the image
        self.disparity_map = self.inference(input_tensor)
        return self.disparity_map

    def get_depth(self):
        return self.camera_config.f*self.camera_config.baseline/self.disparity_map

    def prepare_input(self, left_img, right_img):
        left_img = cv2.resize(left_img, (self.input_width, self.input_height))
        right_img = cv2.resize(right_img, (self.input_width, self.input_height))

        if (self.model_type == ModelType.eth3d):
            # Shape (1, None, None, 2)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            left_img = np.expand_dims(left_img,2)
            right_img = np.expand_dims(right_img,2)
            combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
        else:
            # Shape (1, None, None, 6)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0

        combined_img = combined_img.transpose(2, 0, 1)
        return np.expand_dims(combined_img, 0).astype(np.float32)

    def inference(self, input_tensor):
        left_disparity = self.exec_net.infer({self.input_blob: input_tensor})
        return np.squeeze(left_disparity['reference_output_disparity'])

    def updateFps(self):
        updateRate = 1
        self.frameCounter += 1
        # Every updateRate frames calculate the fps based on the ellapsed time
        if self.frameCounter == updateRate:
            timeNow = time.time()
            ellapsedTime = timeNow - self.timeLastPrediction
            self.fps = int(updateRate/ellapsedTime)
            self.frameCounter = 0
            self.timeLastPrediction = timeNow