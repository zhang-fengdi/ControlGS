#
# This file is part of ControlGS.
# 
# ControlGS is a derivative work based on 3D Gaussian Splatting (3DGS), originally developed by 
# the GRAPHDECO research group at Inria.
#
# Copyright (C) 2025, Fengdi Zhang
# Portions copyright (C) 2023, Inria - GRAPHDECO research group
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact: zhangfd22@mails.tsinghua.edu.cn
# Original authors contact: george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self.masks = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cpu"
        self.eval = False
        self.preview_masked_images = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.001
        self.exposure_lr_final = 0.0001
        self.exposure_lr_delay_steps = 5000
        self.exposure_lr_delay_mult = 0.001
        self.lambda_dssim = 0.2
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.depth_l1_max_steps = 30_000
        self.random_background = False
        self.optimizer_type = "default"

        # ControlGS Hyperparameters:
        self.lambda_opacity = 3e-7  # Opacity regularization strength; ↑ = compact, ↓ = quality (range: [1e-7, 1e-6])
        self.max_densification = 6  # Max number of densification steps
        self.densification_batch_size = 100_000  # Batch size used for densification
        self.prune_change_threshold = 2_000  # Min Gaussian count change to trigger pruning
        self.opacity_threshold = 0.005  # Opacity threshold below which pruning occurs
        self.post_densification_filter_delay = 100  # Delay (iterations) before applying opacity filtering after densification
        self.is_plot_enabled = False  # Enable real-time plotting with matplotlib during training
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
