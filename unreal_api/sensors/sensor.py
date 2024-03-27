from abc import ABCMeta, abstractmethod


class Sensor(metaclass=ABCMeta):

	@property
	@abstractmethod
	def init_command(self):
		pass

	@property
	@abstractmethod
	def change_command(self):
		pass

	@property
	@abstractmethod
	def command(self):
		pass

	@property
	@abstractmethod
	def settings(self):
		pass

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def change_settings(self):
		pass

	@abstractmethod
	def get_observation(self, data):
		pass

	@abstractmethod
	def close(self):
		pass
