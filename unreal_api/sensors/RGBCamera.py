from .sensor import Sensor
import numpy as np
import math
import cv2
import os


np.set_printoptions(threshold=np.inf)


class RGBCamera(Sensor):

	def __init__(self, width, height, channels, FOV=None, focal=None, show=False, image_folder_path=None, video_path=None):
		super().__init__()
		if [FOV, focal].count(None) != 1:
			raise TypeError("Exactly 1 between the FOV and the focal must be specified.")

		self.num_channels = 3
		self.image_folder_path = None
		self.idx = 0
		self.video_path = None
		self.video = None

		self.__command = 'OBS_{}'.format(self.__class__.__name__)

		self.__settings = {
							'width'			: None,
							'height'		: None,
							'channels'		: None,
							'FOV'			: None,
							'focal'			: None,
							'cx'			: None,
							'cy'			: None,
							'camera_matrix'	: None
						}

		self.change_settings(width, height, channels, FOV, focal, show, image_folder_path, video_path)

		self.__init_command = 'INIT_{}_{}_{}_{}'.format(self.__class__.__name__, self.__settings['width'], self.__settings['height'], self.__settings['FOV'])

		id_r = np.random.randint(0, 1000, size=1)
		pid = os.getpid()
		self.__identification = 'PID:'+str(pid)+str(id_r)

	@property
	def init_command(self):
		return self.__init_command

	@property
	def change_command(self):
		return self.__change_command

	@property
	def command(self):
		return self.__command

	@property
	def settings(self):
		return self.__settings

	def change_settings(self, width=None, height=None, channels=None, FOV=None, focal=None, show=None, image_folder_path=None, video_path=None):
		if [FOV, focal].count(None) == 0:
			raise TypeError("Exactly 1 between the FOV and the focal must be specified.")

		if width is not None:
			self.__settings['width'] = width
			self.__settings['cx'] = width / 2
		if height is not None:
			self.__settings['height'] = height
			self.__settings['cy'] = height / 2
		if channels is not None:
			self.__settings['channels'] = channels
			channel_dict = {'R': 0, 'G': 1, 'B': 2, 'D': 3}
			self.channel_list = []
			for c in self.__settings['channels']:
				c_n = channel_dict[c]
				self.channel_list.append(c_n)
		if FOV is not None:
			self.__settings['FOV'] = FOV
			self.__settings['focal'] = 0.5 * self.__settings['width'] / math.tan(0.5 * math.radians(self.__settings['FOV']))
		elif focal is not None:
			self.__settings['focal'] = focal
			self.__settings['FOV'] = math.degrees(2 * math.atan(self.__settings['width'] / (2 * self.__settings['focal'])))
		camera_matrix = [[self.__settings['focal'], 0, self.__settings['cx']], [0, self.__settings['focal'], self.__settings['cy']], [0, 0, 1]]
		self.__settings['camera_matrix'] = np.array(camera_matrix)
		if show is not None:
			self.show = show
		if image_folder_path is not None:
			if self.image_folder_path != image_folder_path:
				self.idx = 0
			self.image_folder_path = image_folder_path
		if video_path is not None:
			self.video_path = video_path
			self.video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (self.__settings['width'], self.__settings['height']))

		self.__change_command = 'CHANGE_{}_{}_{}_{}'.format(self.__class__.__name__, self.__settings['width'], self.__settings['height'], self.__settings['FOV'])

	def get_observation(self, data):
		image = np.frombuffer(data, dtype=np.uint8).reshape(self.__settings['height'], self.__settings['width'], self.num_channels).astype(np.float32) / 255
		RGB_image = image[:, :, :3]

		if self.show or self.image_folder_path or self.video_path:
			RGB_image_formatted = self.__format_image(RGB_image)

			if self.show:
				self.__show_image_on_screen(RGB_image_formatted)
			if self.image_folder_path is not None:
				self.__save_image(RGB_image_formatted)
			if self.video_path is not None:
				self.__save_video(RGB_image_formatted)

		return RGB_image[:, :, self.channel_list]

	def __format_image(self, RGB_image):
		rgb_list = list(set(self.channel_list) & {0, 1, 2})
		if rgb_list:
			RGB_image_to_show = np.zeros_like(RGB_image)
			for c_n in rgb_list:
				RGB_image_to_show[:, :, c_n] = RGB_image[:, :, c_n]
			return RGB_image_to_show[..., ::-1]

	def __show_image_on_screen(self, RGB_image_formatted):
		cv2.imshow('{}_{}_{}'.format(self.__class__.__name__, ''.join(sorted(set('RGB') & set(self.__settings['channels']), key='RGB'.index)), self.__identification), RGB_image_formatted)
		cv2.waitKey(1)

	def __save_image(self, RGB_image_formatted):
		cv2.imwrite('{}{}.png'.format(self.image_folder_path, str(self.idx).zfill(6)), (RGB_image_formatted * 255).astype(np.uint8))
		self.idx += 1

	def __save_video(self, RGB_image_formatted):
		self.video.write((RGB_image_formatted * 255).astype(np.uint8))

	def close(self):
		if self.video is not None:
			self.video.release()
		cv2.destroyAllWindows()
