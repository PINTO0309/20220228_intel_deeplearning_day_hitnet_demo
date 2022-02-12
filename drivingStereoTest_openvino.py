import sys
import cv2
import numpy as np
import glob
import argparse
import logging as log

from hitnet_openvino import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig

def main():
	log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model_type',
		type=str,
		default='middlebury_d400',
		choices=['middlebury_d400', 'flyingthings_finalpass_xl', 'eth3d'],
		help='model type'
	)
	parser.add_argument(
		'--dataset_path',
		type=str,
		default='DrivingStereo images',
		help='dataset folder path'
	)
	parser.add_argument(
		'--input_size',
		type=str,
		default='480x640',
		help='input resolution'
	)
	args = parser.parse_args()

	# Get image list
	left_images = glob.glob(f'{args.dataset_path}/left/*.jpg')
	left_images.sort()
	right_images = glob.glob(f'{args.dataset_path}/right/*.jpg')
	right_images.sort()
	depth_images = glob.glob(f'{args.dataset_path}/depth/*.png')
	depth_images.sort()

	# Select model type
	model_type = ModelType(args.model_type)
	# Get the input size
	H=args.input_size.split('x')[0]
	W=args.input_size.split('x')[1]

	if model_type == ModelType.middlebury_d400:
		model_path = f"{args.model_type}/saved_model_{H}x{W}/openvino/FP32/{args.model_type}_{H}x{W}.xml"
	elif model_type == ModelType.flyingthings_finalpass_xl:
		model_path = f"{args.model_type}/saved_model_{H}x{W}/openvino/FP32/{args.model_type}_{H}x{W}.xml"
	elif model_type == ModelType.eth3d:
		model_path = f"{args.model_type}/saved_model_{H}x{W}/openvino/FP32/{args.model_type}_{H}x{W}.xml"

	input_width = int(args.input_size.split('x')[1])
	camera_config = CameraConfig(0.546, 2000/1920*input_width)
	max_distance = 80

	# Initialize model
	hitnet_depth = HitNet(model_path, model_type, camera_config)

	cv2.namedWindow("Estimated depth", cv2.WINDOW_AUTOSIZE)
	for left_path, right_path, depth_path in zip(left_images[700:], right_images[700:], depth_images[700:]):

		# Read frame from the video
		left_img = cv2.imread(left_path)
		right_img = cv2.imread(right_path)
		depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/256
		# Estimate the depth
		disparity_map = hitnet_depth(left_img, right_img)
		depth_map = hitnet_depth.get_depth()

		color_disparity = draw_disparity(disparity_map)
		color_depth = draw_depth(depth_map, max_distance)
		color_real_depth = draw_depth(depth_img, max_distance)
		color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
		cobined_image = np.hstack((left_img, color_depth))

		cv2.imshow("Estimated depth", cobined_image)

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()