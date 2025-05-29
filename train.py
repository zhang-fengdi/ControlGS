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

# Standard library imports
import os
import sys
import time
import uuid
from random import randint
from argparse import ArgumentParser, Namespace

# Third-party imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lpipsPyTorch import lpips

# Conditional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

# Project-specific imports
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

def training(dataset, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from):

    # Save original lambda_opacity for training report
    original_lambda_opacity = opt.lambda_opacity

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.depth_l1_max_steps)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.is_plot_enabled:
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(10, 5))  # Initialize the plot
        ax2 = ax.twinx()  # Create a secondary Y-axis

        # Initialize empty lists to store plotting data
        losses = []
        psnr_train_list = []
        psnr_test_list = []
        psnr_train_iterations = []
        psnr_test_iterations = []

        # Initialize line objects for dynamic updating
        loss_line, = ax.plot([], [], label="Loss", color='blue')
        psnr_train_line, = ax2.plot([], [], label="Train PSNR", color='green')
        psnr_test_line, = ax2.plot([], [], label="Test PSNR", color='red')

        # Set labels and title
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss and PSNR Over Iterations")
        ax.grid(True)
        ax2.set_ylabel("PSNR")
        ax2.yaxis.set_label_position('right')

        # Draw the initial legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        # Modify the test frequency
        testing_iterations = [1] + list(range(200, opt.iterations + 1, 200))

    # Variable initialization
    before_pruned_gaussian_count = gaussians.get_xyz.shape[0]  # Number of Gaussian points remaining after the last pruning
    num_removed = float('inf')
    no_filtering_until = opt.post_densification_filter_delay  # Initialize with the delay interval for consistent handling

    
    train_start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 iterations, increase the SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, radii = render_pkg["render"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.is_masked
        if mask is not None:
            mask = mask.cuda()
            gt_image[mask] = image.detach()[mask]

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        opacity = gaussians.get_opacity
        opacity_l1 = opacity.sum()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opt.lambda_opacity * opacity_l1

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            if mask is not None:
                depth_mask = depth_mask * ~mask

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Gaussian Number": f"{gaussians.get_xyz.shape[0]:.{0}f}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Gaussian Number": f"{gaussians.get_xyz.shape[0]:.{0}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Prune
            if iteration > no_filtering_until and iteration % 100 == 0:
                gaussians.prune_by_opacity(opt.opacity_threshold, scene.cameras_extent)
                after_pruned_gaussian_count = gaussians.get_xyz.shape[0]
                num_removed = before_pruned_gaussian_count - after_pruned_gaussian_count
                before_pruned_gaussian_count = after_pruned_gaussian_count
                tqdm.write("[ITER {}] Number of Gaussians after pruning: {} (Removed: {})".format(iteration, gaussians.get_xyz.shape[0], num_removed))

            # dynamic pruning threshold
            opt.prune_change_threshold = 2_000 if (gaussians.get_split_count == gaussians.get_min_split_count).all() else float('inf')

            # calculate the number of Gaussians after pruning
            if gaussians.get_min_split_count < opt.max_densification and num_removed < opt.prune_change_threshold:                
                # densify the Gaussians
                gaussians.octree_densify(opt.densification_batch_size)

                # calsulate and display the number of Gaussians after densification
                num_gaussians_after_densification = gaussians.get_xyz.shape[0]
                tqdm.write("Number of Gaussians after densification: {}".format(num_gaussians_after_densification))

                # Display densification progress
                if (gaussians.get_split_count == gaussians.get_min_split_count).all():
                    tqdm.write("[ITER {}] Completed {} rounds of densification.".format(iteration, gaussians.get_min_split_count))
                else:
                    tqdm.write("Remaining {} Gaussians to densify in round {}.".format((gaussians.get_split_count == gaussians.get_min_split_count).sum(), gaussians.get_min_split_count + 1))
                
                # Reset parameters
                before_pruned_gaussian_count = gaussians.get_xyz.shape[0]
                num_removed = float('inf')

                # After densification, skip opacity filtering for 'no_filtering_until' iterations
                no_filtering_until = iteration + opt.post_densification_filter_delay    

            # If the number of removed Gaussians is below the threshold after max densification rounds, set opacity regularization to zero
            if gaussians.get_min_split_count == opt.max_densification and num_removed < opt.prune_change_threshold:        
                opt.lambda_opacity = 0              

            # Log and save
            kwargs = {"ax": ax, "ax2": ax2, "losses": losses, 
                "psnr_train_iterations": psnr_train_iterations, "psnr_train_list": psnr_train_list,
                "psnr_test_iterations": psnr_test_iterations, "psnr_test_list": psnr_test_list,
                "loss_line": loss_line, "psnr_train_line": psnr_train_line, "psnr_test_line": psnr_test_line,
                "gaussian_number": gaussians.get_xyz.shape[0]} if opt.is_plot_enabled else {
                    "original_lambda_opacity": original_lambda_opacity,
                    "max_densification": opt.max_densification,
                    "train_start_time": train_start_time,
                    "gaussian_number": gaussians.get_xyz.shape[0]}
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, opt.is_plot_enabled, **kwargs)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
            if iteration == opt.iterations:
                tqdm.write("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

    # Save the training progress plot
    if opt.is_plot_enabled:
        fig.savefig(os.path.join(dataset.model_path,"loss_psnr_plot__lambda_{:1.1e}__max_dens_{}__max_iter_{}_time_{:.2f}min.png".format(
                original_lambda_opacity,
                opt.max_densification,
                iteration,
                (time.time() - train_start_time) / 60
            )
        ))
        print("Loss and PSNR plot saved to:", os.path.join(dataset.model_path, "loss_psnr_plot.png"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", f"{os.path.basename(os.path.normpath(args.source_path))}_{unique_str[0:10]}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, is_plot_enabled, **kwargs):
    
    if is_plot_enabled:
        # Parse arguments
        ax = kwargs.get("ax")
        ax2 = kwargs.get("ax2")
        losses = kwargs.get("losses")
        psnr_train_iterations = kwargs.get("psnr_train_iterations")
        psnr_train_list = kwargs.get("psnr_train_list")
        psnr_test_iterations = kwargs.get("psnr_test_iterations")
        psnr_test_list = kwargs.get("psnr_test_list")
        loss_line = kwargs.get("loss_line")
        psnr_train_line = kwargs.get("psnr_train_line")
        psnr_test_line = kwargs.get("psnr_test_line")
        gaussian_number = kwargs.get("gaussian_number")

        # Append the current loss to the losses list
        losses.append(loss.item())

        # Initialize best PSNR and iteration tracking variables
        if not hasattr(training_report, "best_train_psnr"):
            training_report.best_train_psnr = 0
            training_report.best_train_iter = 0
            training_report.best_test_psnr = 0
            training_report.best_test_iter = 0
            training_report.best_gaussian_number = 0

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # Save Gaussians
        if not is_plot_enabled:
            tqdm.write("[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        metrics_dict = {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask = viewpoint.is_masked
                    if mask is not None:
                        mask = mask.cuda()
                        gt_image[mask] = image.detach()[mask] # Apply mask
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if not is_plot_enabled:
                        lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if not is_plot_enabled:
                    lpips_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])

                    tqdm.write("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    tqdm.write("[ITER {}] Evaluating {}: LPIPS {} SSIM {}".format(iteration, config['name'], lpips_test, ssim_test))

                    metrics_dict[config['name']] = {
                        "L1": float(l1_test),
                        "PSNR": float(psnr_test),
                        "LPIPS": float(lpips_test),
                        "SSIM": float(ssim_test)
                    }
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                # Append the PSNR value to the corresponding list and record the iteration number
                if is_plot_enabled:
                    if config['name'] == 'train':
                        psnr_train_list.append(psnr_test.item())
                        psnr_train_iterations.append(iteration)
                        if psnr_test > training_report.best_train_psnr:
                            training_report.best_train_psnr = psnr_test.item()
                            training_report.best_train_iter = iteration
                            training_report.best_gaussian_number = gaussian_number
                    elif config['name'] == 'test':
                        psnr_test_list.append(psnr_test.item())
                        psnr_test_iterations.append(iteration)
                        if psnr_test > training_report.best_test_psnr:
                            training_report.best_test_psnr = psnr_test.item()
                            training_report.best_test_iter = iteration

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

        # Save metrics to a TXT file
        if not is_plot_enabled:
            orig_lambda = kwargs.get("original_lambda_opacity")
            max_dens = kwargs.get("max_densification")
            train_start = kwargs.get("train_start_time")
            gau_num = kwargs.get("gaussian_number")
            elapsed_total = (time.time() - train_start) / 60
            file_name = os.path.join(scene.model_path, f"metrics_lambda_{orig_lambda:.1e}_maxdens_{max_dens}_iter_{iteration}_time_{elapsed_total:.2f}min.txt")
            with open(file_name, "w") as f:
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Elapsed training time (min): {elapsed_total:.2f}\n")
                f.write(f"Number of Gaussians: {gau_num}\n")
                for key, m in metrics_dict.items():
                    f.write(f"=== {key.upper()} metrics ===\n")
                    f.write(f"L1: {m['L1']}\n")
                    f.write(f"PSNR: {m['PSNR']}\n")
                    f.write(f"SSIM: {m['SSIM']}\n")
                    f.write(f"LPIPS: {m['LPIPS']}\n")
            tqdm.write(f"[ITER {iteration}] Metrics saved to {file_name}")

    # Dynamically update loss and PSNR plot
    if is_plot_enabled:
        if iteration % 200 == 0:
            # Update loss line data
            loss_line.set_data(range(len(losses)), losses)
            ax.relim()
            ax.autoscale_view()

            # Update PSNR lines if data exists
            if psnr_train_list:
                psnr_train_line.set_data(psnr_train_iterations, psnr_train_list)
            if psnr_test_list:
                psnr_test_line.set_data(psnr_test_iterations, psnr_test_list)
            ax2.relim()
            ax2.autoscale_view()

            # Update the title
            title = (
                f"Training Loss and PSNR Over Iterations\n"
                f"Best Train PSNR: {training_report.best_train_psnr:.2f} (Iter {training_report.best_train_iter})"
                )
            if psnr_test_list:  # Check if test PSNR exists
                title += f", Best Test PSNR: {training_report.best_test_psnr:.2f} (Iter {training_report.best_test_iter})"
            title += f", Best Gaussian Number: {training_report.best_gaussian_number}"
            ax.set_title(title)

            plt.draw()
            plt.pause(1)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(10_000, 100_000, 10_000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # additional code to keep the plot open
    # plt.ioff()
    # plt.show()

    # All done
    print("\nTraining complete.")
