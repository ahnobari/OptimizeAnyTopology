import os
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from torch.utils.data import Dataset
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup
)
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
import pickle
import shutil

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 lr: Optional[float] = 1e-4,
                 weight_decay: Optional[float] = 0.0,
                 schedule_type: Optional[str] = "cosine_with_warmup",
                 lr_final: Optional[float] = 1e-6,
                 max_steps: Optional[int] = 1000,
                 warmup_steps: Optional[int] = 100,
                 checkpoint_path: Optional[str] = None,
                 optimizer_type: Optional[str] = "AdamW",
                 gradient_checkpointing: Optional[bool] = False,
                 max_grad_norm: Optional[float] = 1.0,
                 gradient_accumulation_steps: Optional[int] = 1,
                 custom_loss: Optional[Callable] = None,
                 scale_scheduler_by_porc_count: Optional[bool] = False,
                 dataloader_seed: Optional[int] = 0):
        
        self.dataloader_seed = dataloader_seed
        self.optimizer_type = optimizer_type
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        self.custom_loss = custom_loss
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize accelerator with gradient accumulation config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        
        if scale_scheduler_by_porc_count:
            np = self.accelerator.state.num_processes
            max_steps = (max_steps + np - 1) // np
        
        # Setup model
        self.model = model
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            
        # Set up optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.setup_optimizer()
        
        # Set up learning rate scheduler
        self.schedule_type = schedule_type
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.setup_scheduler()
        
        self.current_epoch = 0
        self.global_step = 0
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint_on_train = True
            self.checkpoint_path = checkpoint_path
        else:
            self.load_checkpoint_on_train = False
            self.checkpoint_path = None
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
    def setup_optimizer(self):
        """Set up the optimizer for training."""
        param_list = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "Lion":
            try:
                import lion_pytorch
                self.optimizer = lion_pytorch.Lion(param_list, lr=self.lr, weight_decay=self.weight_decay)
            except ImportError:
                print("Lion optimizer not available. Falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        else:
            # Default to AdamW
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
    
    def setup_scheduler(self):
        """Set up the learning rate scheduler."""
        if self.schedule_type == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps * self.accelerator.num_processes,
                num_training_steps=self.max_steps
            )
        elif self.schedule_type == "cosine_with_warmup" and self.lr_final > 0.0:
            self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps * self.accelerator.num_processes,
                num_training_steps=self.max_steps,
                min_lr=self.lr_final
            )
        elif self.schedule_type == "cosine_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps * self.accelerator.num_processes,
                num_training_steps=self.max_steps
            )
        elif self.schedule_type == "constant_with_warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps * self.accelerator.num_processes
            )
        else:
            self.scheduler = None
            
    def save_checkpoint(self, path: str):
        """Save the current model and optimizer state to a checkpoint."""
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir=path)
        
        if self.accelerator.is_main_process:
            trainer_state = {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step
            }

            with open(os.path.join(path, "trainer_state.pkl"), "wb") as f:
                pickle.dump(trainer_state, f)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(path)
       
        trainer_state_path = os.path.join(path, "trainer_state.pkl")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "rb") as f:
                trainer_state = pickle.load(f)
                self.current_epoch = trainer_state.get('current_epoch', 0)
                self.global_step = trainer_state.get('global_step', 0)
        else:
            print("No trainer state found in checkpoint.")

    def train(self,
              dataset: Dataset, train_indices: Optional[Union[np.ndarray, list]] = None, 
              batch_size: Optional[int] = 32, epochs: Optional[int] = 40, continue_loop: Optional[bool] = True,
              verbose: Optional[bool] = True, checkpoint_steps: Optional[int] = None, 
              collator: Optional[Callable] = None, checkpoint_dir: Optional[str] = 'checkpoints', 
              dataloader_num_workers: Optional[int] = 4, remove_old_checkpoints: Optional[bool] = True, 
              n_total_checkpoints_to_keep: Optional[int] = 10,
              pass_batch_as_dict: Optional[bool] = True):

        if not continue_loop:
            self.model.train()
            self.current_epoch = 0
            self.global_step = 0
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up indices for training
        if train_indices is None:
            train_indices = np.array(list(range(len(dataset))))
            rng = np.random.default_rng(seed=self.dataloader_seed)
            train_indices = rng.permutation(train_indices).tolist()
        
        local_dataset_subset = torch.utils.data.Subset(dataset, train_indices)
            
        dataloader = torch.utils.data.DataLoader(
            local_dataset_subset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            collate_fn=collator,
            pin_memory=True,
            shuffle=True
        )
        
        # Prepare model, optimizer, scheduler, and dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, dataloader
        )
        
        if self.load_checkpoint_on_train:
            self.load_checkpoint(self.checkpoint_path)
            self.load_checkpoint_on_train = False
            
        epoch = self.current_epoch
        total_steps_done = self.global_step
        steps_per_epoch = (len(dataloader) + self.gradient_accumulation_steps - 1) // self.gradient_accumulation_steps
        
        if verbose and self.accelerator.is_main_process:
            print(f"Training with {len(train_indices)} samples in {steps_per_epoch} steps per epoch")
            print(f"Training for {epochs} epochs ({epochs * steps_per_epoch} steps)")
        
        # find current dataloader iteration
        if hasattr(dataloader._iterator, '_next_index'):
            start_iter = dataloader._iterator._next_index()
        else:
            start_iter = 0
        if self.accelerator.is_main_process and verbose:
            print(f"Starting from dataloader iteration {start_iter} for epoch {epoch + 1}")
        
        while epoch < epochs:
            
            if verbose and self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                prog = tqdm(dataloader, total=len(dataloader), initial=start_iter)
            else:
                prog = dataloader
                
            accumulated_loss = 0.0
            step_loss = 0.0
            reported_step_loss = 0.0
            self.model.train()
            
            for step, batch in enumerate(prog):
                batch = batch.to(self.accelerator.device)
                if self.custom_loss is not None:
                    if pass_batch_as_dict:
                        loss_dict = self.custom_loss(self.model, batch)
                    else:
                        loss_dict = self.custom_loss(self.model, **batch)
                else:
                    if pass_batch_as_dict:
                        loss_dict = self.model(batch, compute_loss=True)[1]
                    else:
                        loss_dict = self.model(**batch, compute_loss=True)[1]
                
                l = loss_dict['loss']
                l = self.accelerator.gather(l).mean()
                accumulated_loss += l.item()
                substep_loss = l.item()
                
                loss = loss_dict['loss']/self.gradient_accumulation_steps
                step_loss += self.accelerator.gather(loss).mean().item()

                self.accelerator.backward(loss)
                
                if (step+1) % self.gradient_accumulation_steps == 0:
                    
                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Step scheduler if we're using it
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    total_steps_done += 1
                    
                    if verbose and self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        avg_loss = accumulated_loss / (step+1)
                        reported_step_loss = step_loss
                        
                        if self.gradient_accumulation_steps > 1:
                            display_dict = {
                                'epoch loss': f"{avg_loss:.5f}",
                                'step loss': f"{reported_step_loss:.5f}",
                                'lr': f"{current_lr:.7f}",
                                'step': total_steps_done,
                                'substep loss': f"{substep_loss:.5f}"
                            }
                        else:
                            display_dict = {
                                'epoch loss': f"{avg_loss:.5f}",
                                'lr': f"{current_lr:.7f}",
                                'step': total_steps_done,
                                'loss': f"{substep_loss:.5f}"
                            }
                        
                        for key, value in loss_dict.items():
                            if key != 'loss':
                                display_dict[key] = f"{value.item():.5f}"
                        
                        prog.set_postfix(display_dict)
                        
                    step_loss = 0.0
                    
                    # Save checkpoint if needed
                    if checkpoint_steps is not None and total_steps_done % checkpoint_steps == 0:
                        self.save_checkpoint(os.path.join(checkpoint_dir, f"step_{total_steps_done}"))
                        if verbose and self.accelerator.is_main_process:
                            print(f"Checkpoint saved at step {total_steps_done}")
                        
                        if remove_old_checkpoints and self.accelerator.is_main_process:
                            # Remove old checkpoints if needed
                            checkpoint_to_remove = total_steps_done - checkpoint_steps * n_total_checkpoints_to_keep
                            if checkpoint_to_remove > 0:
                                checkpoint_path = os.path.join(checkpoint_dir, f"step_{checkpoint_to_remove}")
                                if os.path.exists(checkpoint_path):
                                    shutil.rmtree(checkpoint_path)
                        self.accelerator.wait_for_everyone()
                else:
                    if verbose and self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        avg_loss = accumulated_loss / (step+1)

                        if self.gradient_accumulation_steps > 1:
                            display_dict = {
                                'epoch loss': f"{avg_loss:.5f}",
                                'step loss': f"{reported_step_loss:.5f}",
                                'lr': f"{current_lr:.7f}",
                                'step': total_steps_done,
                                'substep loss': f"{substep_loss:.5f}"
                            }
                        else:
                            display_dict = {
                                'epoch loss': f"{avg_loss:.5f}",
                                'lr': f"{current_lr:.7f}",
                                'step': total_steps_done,
                                'loss': f"{substep_loss:.5f}"
                            }
                        
                        for key, value in loss_dict.items():
                            if key != 'loss':
                                display_dict[key] = f"{value.item():.5f}"
                        
                        prog.set_postfix(display_dict)
                    
            epoch += 1
            self.current_epoch = epoch
            
            # Print epoch summary
            if verbose and self.accelerator.is_main_process:
                print(f"Epoch {epoch} completed. Average loss: {accumulated_loss / (step+1):.5f}")
            
            # save epoch checkpoint
            self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}"))
            
            start_iter = 0