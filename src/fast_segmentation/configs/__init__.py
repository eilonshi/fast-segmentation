from src.fast_segmentation.configs.cfg_bisenetv2 import cfg as bisenetv2_cfg


class ConfigDict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    bisenetv2=ConfigDict(bisenetv2_cfg),
)
