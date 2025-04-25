import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from ..utils.general_utils import CfgNode as CN
from torch.cuda.amp import autocast, GradScaler
import time


class Trainer():

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # training parameters
        C.epochs = 10
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # lr scheduler parameters
        C.use_lr_scheduler = True
        C.lr_scheduler_patience = 2
        C.lr_scheduler_factor = 0.5
        C.lr_scheduler_min_lr = 1e-6
        
        return C
    
    def __init__(self, config, model, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.optimizer = model.configure_optimizers(config)

        # Initialize the learning rate scheduler if enabled optimizer =
        if config.use_lr_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience, 
                verbose=True, 
                min_lr=config.lr_scheduler_min_lr
            )
        else:
            self.scheduler = None
            
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.device.startswith('cuda'))
        print("Train", self.device)

        self.n_examples = 0
        self.n_iter = 0
        self.n_epoch = 0
        self.best_loss = np.inf
        self.loss_improved = False

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
            

    def run(self):
        model = self.model
        config = self.config

        model.train()

        for epoch in range(config.epochs):

            for batch, encodings in enumerate(self.dataloader):
                if batch % 1000 == 0:
                    if self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    start = time.perf_counter()

                self.optimizer.zero_grad()

                for k, v in encodings.items():
                    encodings[k] = v.to(self.device)
                
                # self.loss, logits, *_ = self.model(**encodings)
                with autocast(enabled=self.device.startswith('cuda')):
                    self.loss, logits, *_ = self.model(**encodings)

                model.zero_grad(set_to_none=True)
                self.scaler.scale(self.loss).backward()
                self.scaler.unscale_(self.optimizer)  # because we need unscaled for grad clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               config.grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.n_examples += self.dataloader.batch_size

                self.trigger_callbacks('on_batch_end')
                if (batch + 1) % 1000 == 0:
                    if self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    avg_ms = elapsed
                    print(f"epoch {epoch + 1:02d} | "
                          f"batch {batch + 1:05d} | "
                          f"loss {self.loss.item():.4f} | "
                          f"{elapsed:.2f} for 1000 batches "
                          f"({avg_ms:.1f} s/1k batch)")
                
                self.n_iter += 1

            self.n_epoch += 1
            
            # Step the learning rate scheduler at the end of each epoch
            if self.scheduler is not None:
                # Get average loss for this epoch (could also track separately)
                self.scheduler.step(self.loss.item())
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {self.n_epoch} completed. Current learning rate: {current_lr:.2e}')


