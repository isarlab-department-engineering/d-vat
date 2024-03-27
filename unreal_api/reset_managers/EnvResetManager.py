from .reset_manager import ResetManager


class EnvResetManager(ResetManager):

    def __init__(self):
        super().__init__()
        self.__set_command = 'SETRESETMAN_{}'.format(self.__class__.__name__)

    @property
    def set_command(self):
        return self.__set_command

    def perform_reset(self, settings):
        return "RESET_{}".format(settings['scale'])
