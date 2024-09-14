from typing import OrderedDict, Optional

import copy
import os
import sys
import timeit
import random

import torch
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from accelerate import Accelerator
from safetensors.torch import load_model

from diffusion.gaussian_diffusion import GaussianDiffusion


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def sample_timestep(timesteps):
    return random.choice(timesteps)



class TrainLoop:
    def __init__(
            self,
            *,
            model: torch.nn.Module,
            diffusion: GaussianDiffusion,
            data: DataLoader,
            loss_fn: torch.nn.Module,
            controlnet: Optional[torch.nn.Module] = None,
            num_epochs: int = 10000,
            lr: float = 1e-4,
            ema_rate: float = 0.9999,
            save_dir: str = '',
            log_interval: int = 100,
            save_interval: int = 100,
            mixed_precision: str = 'fp16',
            resume_checkpoint: Optional[str] = None,
            weight_decay: float = 0.0,
    ):
        # If ControlNet is present, train ControlNet instead
        opt = AdamW(model.parameters() if controlnet is None else controlnet.parameters(), lr=lr, weight_decay=weight_decay)
        accelerator = Accelerator(project_dir=save_dir, mixed_precision=mixed_precision)
        if not controlnet is None:
            model, controlnet, opt, data = accelerator.prepare(model, controlnet, opt, data)
        else:
            model, opt, data = accelerator.prepare(model, opt, data)
        
        self.device = accelerator.device
        self.model = model.to(device=self.device)
        self.loss_fn = loss_fn.to(device=self.device)
        self.diffusion = diffusion
        self.controlnet = controlnet
        self.data = data
        self.opt = opt
        self.accelerator = accelerator
        self.lr = lr
        self.ema_rate = ema_rate
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.epochs = num_epochs
        self.weight_decay = weight_decay
        
        self.ema_model = copy.deepcopy(model).to(device=self.device)
        accelerator.register_for_checkpointing(self.ema_model)

        if not self.resume_checkpoint is None and os.path.exists(self.resume_checkpoint):
            # Resuming model checkpoints
            print(f'Resuming checkpoint from {self.resume_checkpoint}')
            load_model(self.model, os.path.join(self.resume_checkpoint, 'model.safetensors'))
            if not self.controlnet is None:
                if os.path.exists(os.path.join(self.resume_checkpoint, 'model_1.safetensors')):
                    # Resuming ControlNet checkpoints if present
                    load_model(self.controlnet, os.path.join(self.resume_checkpoint, 'model_1.safetensors'))
                else:
                    # Otherwise, initialize the weights from pretrained model
                    self.controlnet.from_dit(self.model)
                self.controlnet.set_trainable()
                self.controlnet = self.controlnet.to(device=self.device)
            try:
                self.opt.load_state_dict(torch.load(os.path.join(self.resume_checkpoint, 'optimizer.bin')))
            except Exception as e:
                print(f'Resuming optimizer state failed: {e}\nThis should be OK, but double check you are indeed training ControlNet')

    

    def run_loop(self):
        if not self.controlnet is None:
            self.model.eval()
            self.model.requires_grad_(False)
        else:
            self.model.train()
        self.ema_model.eval()
        for epoch in range(self.epochs):
            start = timeit.default_timer()
            acc_loss = []

            for batch, cond in self.data:
                self.opt.zero_grad()
                batch = batch.to(device=self.device)
                cond = {
                    k: v.to(self.device)
                    for k, v in cond.items()
                }
                t = torch.randint(0, self.diffusion.num_timesteps, (batch.shape[0],), device=self.device)
                
                with self.accelerator.autocast():
                    loss = self.diffusion.training_losses(
                        self.model,
                        self.loss_fn,
                        batch,
                        t,
                        model_kwargs=cond,
                        controlnet=self.controlnet,
                    )['loss'].mean()

                    acc_loss.append(loss.detach().cpu().item())

                self.accelerator.backward(loss)
                self.opt.step()
                update_ema(self.ema_model, self.model, decay=self.ema_rate)
            
            end = timeit.default_timer()
            print(f'Epoch {epoch}/{self.epochs}\tLoss: {sum(acc_loss) / len(acc_loss):>8.5f}\tTime elapsed: {end - start:>6.3f}s', file=sys.stderr, flush=True)

            if epoch % self.save_interval == 0:
                self.accelerator.save_state(self.save_dir)
