from .isar_socket import IsarSocket
import numpy as np
import importlib
import time


class Environment:

	def __init__(self, port, sensor_settings, action_manager_settings, reset_manager_settings=None, address='localhost', render=False, fps=False, observations_step=None, observations_reset=None):
		self.__port = port
		self.__sock = IsarSocket(port, address)

		self.__action_manager_set = self.__get_action_managers_set_from_UE4()

		self.__reset_manager_set = self.__get_reset_managers_set_from_UE4()

		self.__sensor_set = self.__get_sensor_set_from_UE4()

		if reset_manager_settings is not None:
			self.__reset_manager = self.__set_reset_manager(reset_manager_settings)
		else:
			self.__reset_manager = None

		self.__use_sensor(sensor_settings)
		self.__action_manager = self.__set_action_manager(action_manager_settings)

		if render:
			self.switch_rendering()

		if fps:
			self.enable_fps_counter()

		self.__observations_step = observations_step
		self.__observations_reset = observations_reset

	@property
	def port(self):
		return self.__port

	@property
	def action_manager_set(self):
		return self.__action_manager_set

	@property
	def action_set(self):
		return self.__action_manager.action_set

	@property
	def sensor_set(self):
		return self.__sensor_set

	@property
	def observations_step(self):
		return self.__observations_step

	@property
	def observations_reset(self):
		return self.__observations_reset

	def __get_sensor_set_from_UE4(self):
		self.__sock.send_command('SENSORS')
		sensor_list = self.__sock.rec_bytes(1)[0].decode("utf-8").split(' ')[:-1]
		time.sleep(0.1)
		return {sensor: None for sensor in sensor_list}

	def __get_action_managers_set_from_UE4(self):
		self.__sock.send_command('ACTIONS')
		action_manager_list = self.__sock.rec_bytes(1)[0].decode("utf-8").split(' ')[:-1]
		time.sleep(0.1)
		return {action_manager: None for action_manager in action_manager_list}

	def __get_reset_managers_set_from_UE4(self):
		self.__sock.send_command('RESETS')
		reset_manager_list = self.__sock.rec_bytes(1)[0].decode("utf-8").split(' ')[:-1]
		time.sleep(0.1)
		return {reset_manager: None for reset_manager in reset_manager_list}

	def __get_sensor(self, sensor_name):
		try:
			sensor = self.__sensor_set[sensor_name]
		except KeyError:
			self.close_connection()
			raise Exception('Invalid sensor.')
		if sensor is None:
			self.close_connection()
			raise Exception('{} has not been initialized.'.format(sensor_name))
		else:
			return sensor

	def __set_action_manager(self, action_manager_settings):
		action_manager_name = list(action_manager_settings.keys())[0]
		settings = action_manager_settings[action_manager_name]
		module = importlib.import_module('unreal_api.action_managers.{}'.format(action_manager_name))
		class_ = getattr(module, action_manager_name)
		instance = class_(**settings)
		self.__action_manager_set[action_manager_name] = instance
		self.__sock.send_command(instance.set_command)
		time.sleep(0.1)
		return instance

	def __set_reset_manager(self, reset_manager_settings):
		reset_manager_name = list(reset_manager_settings.keys())[0]
		settings = list(reset_manager_settings.values())[0]
		module = importlib.import_module('unreal_api.reset_managers.{}'.format(reset_manager_name))
		class_ = getattr(module, reset_manager_name)
		instance = class_(**settings)
		self.__reset_manager_set[reset_manager_name] = instance
		self.__sock.send_command(instance.set_command)
		time.sleep(0.1)
		return instance

	def __set_sensor(self, sensor_name, settings):
		module = importlib.import_module('unreal_api.sensors.{}'.format(sensor_name))
		class_ = getattr(module, sensor_name)
		instance = class_(**settings)
		self.__sensor_set[sensor_name] = instance
		return instance

	def __use_sensor(self, sensor_settings):
		sensor_init_command_list = []
		for sensor_name, settings in sensor_settings.items():
			sensor = self.__set_sensor(sensor_name, settings)
			sensor_init_command_list.append(sensor.init_command)
		self.__sock.send_command(' '.join(sensor_init_command_list))
		time.sleep(0.1)

	def switch_rendering(self):
		self.__sock.send_command('RENDER')
		time.sleep(0.1)

	def enable_fps_counter(self):
		self.__sock.send_command('FPS')
		time.sleep(0.1)

	def change_sensor_settings(self, sensor_settings):
		sensor_change_command_list = []
		for sensor_name, settings in sensor_settings.items():
			sensor = self.__get_sensor(sensor_name)
			sensor.change_settings(**settings)
			sensor_change_command_list.append(sensor.change_command)
		self.__sock.send_command(' '.join(sensor_change_command_list))

	def change_reset_manager(self, reset_manager_settings):
		self.__set_reset_manager(reset_manager_settings)
		
	def change_action_manager(self, action_manager_settings):
		self.__set_action_manager(action_manager_settings)

	def get_sensor_settings(self, sensor_name_list):
		sensor_settings_list = []
		for sensor_name in sensor_name_list:
			sensor = self.__get_sensor(sensor_name)
			sensor_settings_list.append(sensor.settings)
		return sensor_settings_list

	def perform_action(self, action):
		command = self.__action_manager.perform_action(action)
		self.__sock.send_command(command)
		data = self.__sock.rec_bytes(1)[0]
		hit = np.frombuffer(data, dtype=np.uint8)
		return hit

	def reset(self, sensor_name_list=None, reset_settings=None):
		if sensor_name_list is None:
			sensor_name_list = self.__observations_reset
		if type(sensor_name_list) is not list:
			raise Exception("Sensor Name List must be a list")

		if self.__reset_manager is not None:
			command = self.__reset_manager.perform_reset(reset_settings)

			sensor_list = []
			sensor_command_list = []

			for sensor_name in sensor_name_list:
				sensor = self.__get_sensor(sensor_name)
				sensor_command_list.append(sensor.command)
				sensor_list.append(sensor)

			self.__sock.send_command('{} {}'.format(command, ' '.join(sensor_command_list)))

			data = self.__sock.rec_bytes(len(sensor_name_list))
			obs_list = []
			for i in range(len(sensor_list)):
				obs_list.append(sensor_list[i].get_observation(data[i]))

			return obs_list
		else:
			raise Exception("Reset Manager has not been configured")

	def get_obs(self, sensor_name_list):
		sensor_list = []
		sensor_command_list = []
		for sensor_name in sensor_name_list:
			sensor = self.__get_sensor(sensor_name)
			sensor_command_list.append(sensor.command)
			sensor_list.append(sensor)
		self.__sock.send_command(' '.join(sensor_command_list))
		data = self.__sock.rec_bytes(len(sensor_name_list))
		obs_list = []
		for i in range(len(sensor_list)):
			obs_list.append(sensor_list[i].get_observation(data[i]))
		return obs_list

	def env_step(self, action, sensor_name_list=None):
		if sensor_name_list is None:
			sensor_name_list = self.__observations_step
		if type(sensor_name_list) is not list:
			raise Exception("Sensor Name List must be a list")

		command = self.__action_manager.perform_action(action)
		sensor_list = []
		sensor_command_list = []
		for sensor_name in sensor_name_list:
			sensor = self.__get_sensor(sensor_name)
			sensor_command_list.append(sensor.command)
			sensor_list.append(sensor)
		self.__sock.send_command('{} {}'.format(command, ' '.join(sensor_command_list)))
		data = self.__sock.rec_bytes(len(sensor_name_list) + 1)
		data_hit, data_obs = data[0], data[1:]
		hit = np.frombuffer(data_hit, dtype=np.uint8)
		obs_list = []
		for i in range(len(sensor_list)):
			obs_list.append(sensor_list[i].get_observation(data_obs[i]))

		return hit, obs_list

	def close_connection(self):
		self.__sock.send_command('CLOSE')
		self.__sock.close()
		for _, sensor in self.__sensor_set.items():
			if sensor is not None:
				sensor.close()
