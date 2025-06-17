<!-- Badges -->
[![Project Page](https://img.shields.io/badge/Project%20Page-ControlGS-blue?style=flat-square)](https://zhang-fengdi.github.io/ControlGS/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.10473-B31B1B.svg?style=flat-square)](https://arxiv.org/abs/2505.10473)

# ControlGS

To reduce storage and computational costs, 3D Gaussian Splatting (3DGS) seeks to minimize the number of Gaussians used while preserving high rendering quality, introducing an inherent trade-off between Gaussian quantity and rendering quality. ControlGS extends 3DGS with *semantically meaningful*, cross-scene consistent quantity–quality control. Through a single training run using a fixed setup and a user-specified hyperparameter reflecting quantity–quality preference, ControlGS can automatically find desirable quantity–quality trade-off points across diverse scenes, from compact objects to large outdoor scenes. It also outperforms baselines by achieving higher rendering quality with fewer Gaussians, and supports a broad adjustment range with stepless control over the trade-off.

For more details, please see the paper: 
["*Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting*".](https://arxiv.org/abs/2505.10473)

## Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM

## Environment Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/zhang-fengdi/ControlGS.git --recursive 
   cd ControlGS
   ```

2. **Prepare the Conda environment**
   ```bash
   conda env create --file environment.yml
   conda activate controlgs
   ```

## Running

To run the optimizer, simply use

```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Key Common Arguments

#### --source_path / -s  
Path to the source directory containing a COLMAP or Synthetic NeRF data set.

#### --lambda_opacity  
Opacity regularization weight; higher → fewer Gaussians (more compact) with lower fidelity, lower → more Gaussians (higher fidelity). Typical range: `1e-7` to `1e-6`.

### Optional Advanced Common Arguments

<details>
<summary>Show advanced options (defaults are sufficient for most use cases)</summary>

#### --eval  
Use a MipNeRF360-style training/test split for evaluation.

#### --masks  
Path to directory containing binary mask images; each mask corresponds to an input image and is used to ignore background or unwanted regions during training. In each mask, `1` marks regions to be masked out, and `0` marks regions to be kept.

#### --is_plot_enabled  
Enable real-time plotting of loss and PSNR curves.

#### --model_path / -m  
Path where the trained model should be stored (e.g. `output/<random>`).

#### --images / -i  
Alternative subdirectory for COLMAP images.

#### --resolution / -r  
Resolution of loaded images before training.  
- If `1, 2, 4, 8`, uses original, ½, ¼ or ⅛ resolution  
- Otherwise, rescales width to the given value (preserving aspect)  
- If unset and input width > 1600px, images auto-rescale to 1600px

#### --data_device  
Device to load source image data onto (`cuda` or `cpu`). It is recommended to use `cpu`.

#### --white_background / -w  
Use white background instead of black (e.g. for NeRF Synthetic evaluation).

#### --sh_degree  
Order of spherical harmonics (max 3).

#### --convert_SHs_python  
Compute SH forward/backward in PyTorch instead of the optimized implementation.

#### --convert_cov3D_python  
Compute 3D covariance forward/backward in PyTorch instead of the optimized implementation.

#### --debug  
Enable debug mode and dump failed rasterizer output for issue reporting.

#### --debug_from  
Iteration (from 0) after which debug mode becomes active.

#### --iterations  
Total number of training iterations.

#### --ip  
IP address for the GUI server.

#### --port  
Port for the GUI server.

#### --test_iterations  
Iterations at which to compute L1 and PSNR on the test set.

#### --checkpoint_iterations  
Iterations at which to save a checkpoint in the model directory.

#### --start_checkpoint  
Path to a checkpoint file to resume training.

#### --quiet  
Suppress console output.

#### --feature_lr  
Learning rate for spherical harmonics features.

#### --opacity_lr  
Learning rate for opacity.

#### --scaling_lr  
Learning rate for scaling parameters.

#### --rotation_lr  
Learning rate for rotations.

#### --position_lr_max_steps  
Steps over which position LR interpolates from initial to final.

#### --position_lr_init  
Initial learning rate for 3D positions.

#### --position_lr_final  
Final learning rate for 3D positions.

#### --position_lr_delay_mult  
Multiplier on the position LR schedule (see *Plenoxels* for reference).

#### --lambda_dssim  
Weight of the SSIM term in the total loss (0–1).

#### --max_densification  
Maximum number of densification steps.

#### --densification_batch_size  
Number of Gaussians in each densification batch.

#### --prune_change_threshold  
Minimum change in the number of Gaussians to trigger pruning.

#### --opacity_threshold  
Opacity value below which Gaussians will be pruned.

#### --post_densification_filter_delay  
Iterations to wait after each densification before filtering.

</details>

## Additional Tools & Resources

ControlGS reuses the same utilities and pipelines from the original 3DGS codebase—see the [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting) for full details:

* **Evaluation & Rendering:** `render.py`, `metrics.py`
* **Interactive Viewers:** Remote and real-time SIBR viewers
* **Dataset Conversion:** `convert.py` pipeline and expected folder layout

## License

This software is free for non-commercial, research and evaluation use under the terms of the [LICENSE.md](LICENSE.md) file.


## Citation

If you find this code useful, please consider giving it a ⭐ star and citing our work:

```bibtex
@article{zhang2025consistent,
   title={Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting},
   author={Zhang, Fengdi and Cao, Hongkun and Huang, Ruqi},
   journal={arXiv preprint arXiv:2505.10473},
   year={2025}
}
```
