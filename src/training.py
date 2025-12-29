import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, List, Callable
import wandb
import time
from tqdm import tqdm

from src.models import BaseModel


def train_model(
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        loss_func: nn.Module,
        input_fn: Callable[[Any], Tuple[torch.Tensor, ...]],
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        run: Optional[wandb.Run],
        save_path: Optional[str],
        target_idx: int = -1,
        log_predictions: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
    ) -> Tuple[List[float], List[float], float]:
    """
    Train loop for a given model with logging to wandb.

    Args:
        model (BaseModel): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loss_func (nn.Module): The loss function to use.
        input_fn (Any): A function that takes a batch and returns the model inputs.
        epochs (int): The number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        valid_dataloader (torch.utils.data.DataLoader): The validation data loader.
        run (wandb.Run): The wandb run object.
        save_path (str): The path to save the best model.
        target_idx (int): The index of the target variable in the batch.
        log_predictions (bool): Whether to log sample predictions to wandb.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        
        
    Returns:
        Tuple[List[float], List[float], float]: Training losses, validation losses, total training time with logging.
    """    
    # Log model and training configuration using BaseModel methods
    run.config.update({
        "batch_size": train_dataloader.batch_size,
        "model_architecture": str(model),
        "num_parameters": model.get_number_of_parameters(),
        "optimizer": optimizer.__class__.__name__,
        "loss_function": loss_func.__class__.__name__,
        "epochs": epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "embedding_size": model.get_embedding_size(),
        "fusion_strategy": model.get_fusion_strategy(),
    }, allow_val_change=True)

    # To calculate the total training time, we need to ignore the WandB logging.
    total_time = 0
    
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        model.train()
        batch_losses = []

        # Train loop with updating weights
        for batch in train_dataloader:
            target = batch[target_idx]
            optimizer.zero_grad()
            outputs = model(*input_fn(batch))

            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        model.eval()
        batch_losses = []
        # Validation loop without updating weights
        for i, batch in enumerate(valid_dataloader):
            target = batch[target_idx]
            outputs = model(*input_fn(batch))
            batch_losses.append(loss_func(outputs, target).item())
            # Log sample predictions for the first batch
            if i == 0 and log_predictions:
                # We assume binary classification             
                probs = torch.sigmoid(outputs).squeeze()
                # Get predicted class (0 or 1)
                preds = (probs >= 0.5).long()
                run.log({"val_sample_prediction": preds.cpu().numpy()})

        valid_loss = np.mean(batch_losses)
        valid_losses.append(valid_loss)

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Track time before logging because of network latency
        elapsed = time.time() - start_time
        total_time += elapsed

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)

        # Log metrics of current epoch to wandb
        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

    return train_losses, valid_losses, total_time