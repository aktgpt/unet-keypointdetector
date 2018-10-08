import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR
from torch.optim import Adam
from dataset import get_data_loaders
import argparse
import json
from unet import UNet, UNetCrossEntropyLoss, FocalLoss
from utility import save_mask_and_pred
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision
from inference import PrecisionRecall

argparser = argparse.ArgumentParser(
    description='Train U-net dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')



class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.learning_rate = config['train']['learning_rate']
        self.learning_decay_type = config['train']['decay_type']
        self.learning_rate_decay = config['train']['learning_rate_decay']
        self.epoch_decay = config['train']['epoch_decay']
        self.epochs = config['train']['epochs']
        self.batch_size = config['train']['batch_size']
        self.val_split_ratio = config['train']['val_split_ratio']

        self.model_save_path = config['train']['model_save_path']
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Model = model.to(self.device)
        self.logs_save_dir = config['train']['log_dir']
        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)

    def learning_rate_scheduler(self):
        if self.learning_decay_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size= self.epoch_decay, gamma=self.learning_rate_decay)


    def train_ignite(self):
        train_loader, validation_loader = get_data_loaders(self.config)
        writer = create_summary_writer(self.Model, train_loader, self.logs_save_dir)

        self.optimizer = Adam(self.Model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        self.learning_rate_scheduler()
        loss = UNetCrossEntropyLoss().cuda()

        trainer = create_trainer(model=self.Model, optimizer=self.optimizer, criterion=loss, device=self.device)
        evaluator = create_evaluator(self.Model, metrics={'CrossEntropy':Loss(loss),
                                    'PrecisionRecall':PrecisionRecall()}, device=self.device)

        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader),
            desc=desc.format(0)
        )
        log_interval = 2

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1

            if iter % log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(log_interval)
            writer.add_scalar("training/logs", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.scheduler.step()
            pbar.refresh()
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            cross_entropy_loss = metrics['CrossEntropy']
            tqdm.write(
                "Current Learning Rate:{:.10f}: Training Results - Epoch: {}  Cross Entropy Loss: {:.2f}"
                    .format(self.optimizer.param_groups[0]['lr'], engine.state.epoch, cross_entropy_loss)
            )
            writer.add_scalar("training/cross_entropy_loss", cross_entropy_loss, engine.state.epoch)


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            pbar.refresh()
            evaluator.run(validation_loader)
            metrics = evaluator.state.metrics
            cross_entropy_loss = metrics['CrossEntropy']
            precision_recall_loss = metrics['PrecisionRecall']
            tqdm.write(
                "Validation Results - Epoch: {}  Cross Entropy Loss: {:.2f} \n Precision: {:.4f},"
                " Recall: {:.4f}, Mean Euclidean Distance: {:.2f}"
                .format(engine.state.epoch, cross_entropy_loss, precision_recall_loss['precision']
                        , precision_recall_loss['recall'], precision_recall_loss['mean_euclidean_dist'])
                    )
            pbar.n = pbar.last_print_n = 0

            input = evaluator.state.batch['image']

            output = evaluator.state.output
            pred = output[0]
            mask = output[1]

            input_grid = torchvision.utils.make_grid(torch.stack([img.cpu() for img in input], dim=0), normalize=True)
            pred_grid = torchvision.utils.make_grid(torch.stack([img.cpu() for img in pred]))
            mask_grid = torchvision.utils.make_grid(torch.stack([img.cpu() for img in mask]))
            # torchvision.utils.save_image(pred_grid, "pred/pred_grid_" + str(engine.state.epoch) + ".png")
            # torchvision.utils.save_image(mask_grid, "pred/mask_grid_" + str(engine.state.epoch) + ".png")
            writer.add_image("Input", input_grid, engine.state.epoch)
            writer.add_image("Result", pred_grid, engine.state.epoch)
            writer.add_image("Ground Truth", mask_grid, engine.state.epoch)
            writer.add_scalar("validation/precision", precision_recall_loss['precision'], engine.state.epoch)
            writer.add_scalar("validation/recall", precision_recall_loss['recall'], engine.state.epoch)
            writer.add_scalar("validation/mean_euclidean_dist", precision_recall_loss['mean_euclidean_dist'], engine.state.epoch)
            writer.add_scalar("validation/cross_entropy_loss", cross_entropy_loss, engine.state.epoch)

        checkpointer = ModelCheckpoint(self.model_save_path, 'unet_v_1_', save_interval=1, n_saved=20, require_empty=False,
                                        save_as_state_dict=True)
        # early_stopping = EarlyStopping(patience=5, score_function=self.score_function, trainer=trainer)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'epoch': self.Model})
        # trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        # evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        trainer.run(train_loader, max_epochs=self.epochs)
        pbar.close()
        writer.close()

    @staticmethod
    def score_function(engine):
        cross_entropy_loss = engine.state.metrics['CrossEntropy']
        return cross_entropy_loss

def create_trainer(model, optimizer, criterion, device=None):
    softmx = nn.Softmax(dim=0)

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        data = batch['image']
        target = batch['mask']

        output = model(data)
        output = softmx(output)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_evaluator(model, metrics={}, device=None):
    softmx = nn.Softmax(dim=0)

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data = batch['image']
            target = batch['mask']

            output = model(data)
            output = softmx(output)
            return output, target

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    sample = next(data_loader_iter)
    x, y = sample['image'], sample['mask']
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


if __name__ == '__main__':
    args = argparser.parse_args(['-c', 'configs/config_2.json'])
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    unet_model = UNet(config)
    print(unet_model)

    trainer = Trainer(unet_model, config)
    trainer.train_ignite()
