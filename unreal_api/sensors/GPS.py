from .sensor import Sensor
import numpy as np


class GPS(Sensor):

	def __init__(self):
		super().__init__()
		self.__command = 'OBS_{}'.format(self.__class__.__name__)
		self.__settings = {}

		self.change_settings()

		self.__init_command = 'INIT_{}'.format(self.__class__.__name__)

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

	def get_observation(self, data):
		GPS_info = np.frombuffer(data, dtype=np.float32)
		GPS_info[1] *= -1
		GPS_info[4] *= -1
		GPS_info[7] *= -1
		GPS_info[10] *= -1
		return GPS_info

	def change_settings(self):
		self.__change_command = 'CHANGE_{}'.format(self.__class__.__name__)

	def close(self):
		pass
