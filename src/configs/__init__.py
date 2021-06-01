from src.configs.cfg_bisenetv1 import cfg as bisenetv1_cfg
from src.configs.cfg_bisenetv2 import cfg as bisenetv2_cfg


class ConfigDict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    bisenetv1=ConfigDict(bisenetv1_cfg),
    bisenetv2=ConfigDict(bisenetv2_cfg),
)
