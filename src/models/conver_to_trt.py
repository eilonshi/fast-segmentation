import argparse
import torch
from torch2trt import torch2trt

from src.lib.architectures import model_factory
from src.configs import cfg_factory
from src.models.consts import NUM_CLASSES

torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='deeplab_cityscapes', )
parse.add_argument('--weight-path', dest='weight_pth', type=str,
                   default='model_final.pth')
parse.add_argument('--outpath', dest='out_pth', type=str,
                   default='model.onnx')
args = parse.parse_args()

cfg = cfg_factory[args.config]
if cfg.use_sync_bn: cfg.use_sync_bn = False

net = model_factory[cfg.model_type](NUM_CLASSES, output_aux=False).cuda()
#  net.load_state_dict(torch.load(args.weight_pth))
net.eval()

#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
dummy_input = torch.randn(1, 3, 1024, 2048).cuda()
input_names = ['input_image']
output_names = ['preds']

trt_model = torch2trt(net, [dummy_input, ])
#  torch.onnx.export(net, dummy_input, args.out_pth,
#      input_names=input_names, output_names=output_names,
#      verbose=False, opset_version=11)
#
#
#  import onnx
#  import onnxruntime as ort
#
#  print('checking {}'.format(args.out_pth))
#  onnx_obj = onnx.load(args.out_pth)
#  print('model loaded')
#  onnx.checker.check_model(onnx_obj)
#
#  sess = ort.InferenceSession(args.out_pth, None)
#  print(sess.get_outputs()[0].name)
#  print(sess.get_outputs()[0].shape)
#  print(len(sess.get_outputs()))
