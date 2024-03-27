from .action_manager import ActionManager


class CoordinateActionManager(ActionManager):

	def __init__(self, command_dict, settings):
		super().__init__()
		self.__action_set = {
								'MOVETO'		: None
							}

		self.set_action_set(command_dict)

		self.__settings = settings

		self.__set_command = 'SETACTIONMAN_{}'.format(self.__class__.__name__)

	@property
	def set_command(self):
		return self.__set_command

	@property
	def action_set(self):
		return self.__action_set

	@property
	def settings(self):
		return self.__settings

	def get_number_of_actions(self):
		n = 0
		for _, val in self.__action_set.items():
			if val is not None:
				n += 1
		return n

	def set_action_set(self, command_dict):
		for action_name, _ in self.__action_set.items():
			self.__action_set[action_name] = None
		for action_name, val in command_dict.items():
			if action_name in self.__action_set:
				self.__action_set[action_name] = val
			else:
				raise Exception('Invalid action set.')

	def perform_action(self, val_list):
		for key, value in self.__action_set.items():
			if value is not None:
				command = 'ACTION_{};'.format(key)
				for val in val_list:
					command += '{:.4f};'.format(val)
				command = command[:-1] + '_'
		return command[:-1]
