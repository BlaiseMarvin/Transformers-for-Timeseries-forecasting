import numpy as np 
import torch 
import os 

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, checkpoint_path="resnet_cpt.pt", restore_best_weights=True, verbose=True):
        """
            Early stopping - stops training when validation loss doesn't improve -saves the model + optimizer dicts for resumption later on
            Args:
             - patience (int) => Epochs to wait after last improvement
             - min_delta (float) => minimum change to count as improvement
             - checkpoint_path (str) => file to save the best model
             - restore_best_weights (bool) => reload best weights when stopping
             - verbose (bool) => whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_metric = np.inf
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, metric, model, optimizer, epoch):
        """Call this after evaluating the metric of interest"""
        """Taking an assumption here that we are looking to minimize the metric of interest"""
        if metric < self.best_metric - self.min_delta:
            # improvement detected at this point
            self.best_metric = metric
            self.counter = 0

            # save full checkpoint (model + optimizer + epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': metric 
            }, self.checkpoint_path)

            if self.verbose:
                print(f"Metric improved to {metric:.4f}. Checkpoint saved at epoch {epoch}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epoch(s)")
            
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.patience} epochs of no improvement")

                if self.restore_best_weights and os.path.exists(self.checkpoint_path):
                    checkpoint = torch.load(self.checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if self.verbose:
                        print("Restored best model and optimizer states from checkpoint")

