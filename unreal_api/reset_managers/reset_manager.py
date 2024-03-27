from abc import ABCMeta, abstractmethod


class ResetManager(metaclass=ABCMeta):

	@property
	@abstractmethod
	def set_command(self):
		pass

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def perform_reset(self, settings=None):
		pass
