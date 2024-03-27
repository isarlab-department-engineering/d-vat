from abc import ABCMeta, abstractmethod


class ActionManager(metaclass=ABCMeta):

	@property
	@abstractmethod
	def set_command(self):
		pass

	@property
	@abstractmethod
	def action_set(self):
		pass

	@property
	@abstractmethod
	def settings(self):
		pass

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def get_number_of_actions(self):
		pass

	@abstractmethod
	def set_action_set(self, command_dict):
		pass

	@abstractmethod
	def perform_action(self, val):
		pass
