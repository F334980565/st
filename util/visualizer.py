import numpy as np
import os
import sys
import ntpath
import time
import torch
from . import util, html_
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        return image_numpy.astype(imtype)
    else:
        return input_image

def save_images(webpage, visuals, image_paths, aspect_ratio=1.0, width=256, batch=True):
    image_dir = webpage.get_image_dir()
    for i, image_path in enumerate(image_paths):
        short_path = ntpath.basename(image_path)
        name = os.path.splitext(short_path)[0]
        webpage.add_header(name)
        ims, txts, links = [], [], []
        for label, im_datas in visuals.items():
            im_data = im_datas[i].unsqueeze(0)
            im = util.tensor2im(im_data)
            image_name = "%s/%s.png" % (label, name)
            os.makedirs(os.path.join(image_dir, label), exist_ok=True)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=width)

class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10
        else:
            self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:
            import visdom

            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(
                    server=opt.display_server,
                    port=opt.display_port,
                    env=opt.display_env,
                )
            else:
                self.vis = visdom.Visdom(
                    port=2004, base_url=os.environ["tensorboard_base_url"] + "/visdom"
                )
            if not self.vis.check_connection():
                self.create_visdom_connections()
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            print("create web directory %s..." % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        cmd = sys.executable + " -m visdom.server -p %d &>/dev/null &" % self.port
        print("\n\nCould not connect to Visdom server. \n Trying to start a server....")
        print("Command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ""
                label_html_row = ""
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += "<td>%s</td>" % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += "<tr>%s</tr>" % label_html_row
                        label_html_row = ""
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += "<td></td>"
                    idx += 1
                if label_html_row != "":
                    label_html += "<tr>%s</tr>" % label_html_row
                try:
                    self.vis.images(
                        images,
                        ncols,
                        2,
                        self.display_id + 1,
                        None,
                        dict(title=title + " images"),
                    )
                    label_html = "<table>%s</table>" % label_html
                    self.vis.text(
                        table_css + label_html,
                        win=self.display_id + 2,
                        opts=dict(title=title + " labels"),
                    )
                except VisdomExceptionBase:
                    self.create_visdom_connections()
            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(
                            image_numpy.transpose([2, 0, 1]),
                            self.display_id + idx,
                            None,
                            dict(title=label),
                        )
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(
                    self.img_dir, "epoch%.3d_%s.png" % (epoch, label)
                )
                util.save_image(image_numpy, img_path)
            webpage = html_.HTML(
                self.web_dir, "Experiment name = %s" % self.name, refresh=0
            )
            for n in range(epoch, 0, -1):
                webpage.add_header("epoch [%d]" % n)
                ims, txts, links = [], [], []
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = "epoch%.3d_%s.png" % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        if len(losses) == 0:
            return
        plot_name = "_".join(list(losses.keys()))
        if plot_name not in self.plot_data:
            self.plot_data[plot_name] = {
                "X": [],
                "Y": [],
                "legend": list(losses.keys()),
            }
        plot_data = self.plot_data[plot_name]
        plot_id = list(self.plot_data.keys()).index(plot_name)
        plot_data["X"].append(epoch + counter_ratio)
        plot_data["Y"].append([losses[k] for k in plot_data["legend"]])
        try:
            self.vis.line(
                X=np.stack([np.array(plot_data["X"])] * len(plot_data["legend"]), 1),
                Y=np.array(plot_data["Y"]),
                opts={
                    "title": self.name,
                    "legend": plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss",
                },
                win=self.display_id - plot_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) " % (
            epoch,
            iters,
            t_comp,
            t_data,
        )
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)
