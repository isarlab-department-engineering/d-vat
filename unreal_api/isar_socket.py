import numpy as np
import socket
import sys


def highestPowerof2(n):
	if n < 1:
		return 0
	res = 1
	for i in range(8 * sys.getsizeof(n)):
		curr = 1 << i
		if curr > n:
			break
		res = curr
	return res


class IsarSocket:

	def __init__(self, port, address):
		self.__MAX_COMMAND_LEN = 1000
		server_address = (address, port)
		check = False
		while not check:
			try:
				self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				self.sock.connect(server_address)
				check = True
			except:
				pass

	def send_command(self, command):
		command = command.ljust(self.__MAX_COMMAND_LEN)[:self.__MAX_COMMAND_LEN]
		self.sock.sendall(bytes(command, 'utf8'))

	def rec_bytes(self, n):
		received_list = []
		for _ in range(n):
			data = bytearray()
			n_bytes = np.frombuffer(self.sock.recv(4), dtype=np.int32)[0]
			rec_size = highestPowerof2(max(n_bytes, 4096))
			while len(data) < n_bytes:
				packet = self.sock.recv(min(rec_size, n_bytes - len(data)))
				data.extend(packet)
			received_list.append(data)
		return received_list

	def close(self):
		self.sock.close()
