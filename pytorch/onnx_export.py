from __future__ import absolute_import, division, print_function

import os
import argparse
import sys
import torch
import torch.nn as nn

from efficient_depth import efficient_depth_module

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='BTS PyTorch modified.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def export(args):
    """Export torch models to onnx models."""
    args.mode = 'test'
    model = efficient_depth_module(args)
    model.encoder.set_swish(memory_efficient=False)

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    input_shape = (3, args.input_height, args.input_width)
    x = torch.randn(1, *input_shape)
    torch.onnx.export(model,
                    x,
                    args.export_name,
                    opset_version=12,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})
    print("finished")
    exit()

if __name__ == '__main__':
    export(args)


