import os
import time
import torch
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util import util
from util.metric import AverageMeter
from util.stage_schedule import apply_stage, parse_stages
from util.text_logger import TextLogger
from util.visualizer import Visualizer

def remove_existing_optimizer(model, optimizer_name):
    optimizer = getattr(model, optimizer_name, None)
    if optimizer is None:
        return
    model.optimizers = [item for item in model.optimizers if item is not optimizer]
    delattr(model, optimizer_name)

def reset_optimizer_lrs(model, lr):
    for optimizer in model.optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def apply_model_stage(model, opt):
    netG = getattr(model, "netG", None)
    if netG is None:
        return
    for name in ("res_out_i", "res_in_j"):
        if hasattr(opt, name) and hasattr(netG, name):
            setattr(netG, name, getattr(opt, name))

def data_dependent_initialize(model, data, style_start):
    remove_existing_optimizer(model, "optimizer_F")
    if hasattr(model, "netD"):
        model.set_requires_grad(model.netD, True)
    if style_start is None:
        model.data_dependent_initialize(data)
        return
    try:
        model.data_dependent_initialize(data, style_start)
    except TypeError:
        model.data_dependent_initialize(data)

def train_stage(opt, model, visualizer, logger, total_iters, stage_idx, stage):
    apply_stage(opt, stage)
    apply_model_stage(model, opt)
    reset_optimizer_lrs(model, opt.lr)
    prefix = f"stage{stage_idx}" if stage is not None else "train"
    if stage is not None:
        print(f"\n{'=' * 20} Start Stage {stage_idx + 1} {'=' * 20}")
        print(f"Config: {stage}")
        logger.log(
            {"event": "stage_config", "stage": stage_idx, "config": stage},
            prefix=prefix,
        )
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print("The number of training images = %d" % dataset_size)
    optimize_time = 0.1
    initialized = False
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        dataset.set_epoch(epoch)
        loss_meters = {name: AverageMeter() for name in model.loss_names}
        for _, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if not initialized:
                data_dependent_initialize(model, data, opt.style_start)
                setup_opt = util.copyconf(
                    opt, continue_train=opt.continue_train and stage_idx == 0
                )
                model.setup(setup_opt)
                model.parallelize()
                initialized = True
            model.set_input(data)
            model.optimize_parameters()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (
                time.time() - optimize_start_time
            ) / batch_size * 0.005 + 0.995 * optimize_time
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, optimize_time, t_data
                )
                logger.log(
                    {"epoch": epoch, "iter": total_iters, "losses": losses},
                    step=total_iters,
                    prefix=f"{prefix}/iter",
                )
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )
            else:
                losses = model.get_current_losses()
            for name, value in losses.items():
                loss_meters[name].update(value, n=batch_size)
            if total_iters % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                print(opt.name)
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)
            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            latest_name = f"stage{stage_idx}_latest" if stage is not None else "latest"
            epoch_name = (
                f"stage{stage_idx}_epoch{epoch}" if stage is not None else epoch
            )
            model.save_networks(latest_name)
            model.save_networks(epoch_name)
        epoch_log_dict = {
            f"epoch/avg_loss_{name}": meter.avg for name, meter in loss_meters.items()
        }
        logger.log(epoch_log_dict, step=total_iters, prefix=f"{prefix}/epoch")
        print(f"epoch {epoch}", epoch_log_dict)
        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()
    return total_iters

def main():
    opt = TrainOptions().parse()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    logger = TextLogger(
        os.path.join(opt.checkpoints_dir, opt.name), filename="run_log.txt"
    )
    logger.log({"event": "config", "options": vars(opt)})
    total_iters = 0
    for stage_idx, stage in enumerate(parse_stages(opt.train_stages)):
        total_iters = train_stage(
            opt, model, visualizer, logger, total_iters, stage_idx, stage
        )

if __name__ == "__main__":
    main()
