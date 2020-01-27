"""
LSTM Base Model
"""

from .model import Model

class LSTMBase(Model):
    """
    LSTM Base Model Class
    """
    def __init__(self, config):
        """
        Initialization of params needed to create/use an alpha model
        :param config: config file path or dictionary
        :return: n/a
        """
        cfg = Model.parse_config(config)

        # Parse required model params
        self.blah = 'X'

        Model.__init__(cfg)
