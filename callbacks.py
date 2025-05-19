
import torch
import reciprocalspaceship as rs
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
import wandb

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt

class LossLogging(Callback):
    
    def __init__(self):
        super().__init__()
        self.loss_per_epoch: float = 0
        self.count_batches: int = 0

    def on_train_epoch_end(self, trainer, pl_module):
        avg = trainer.callback_metrics["train/loss_step_epoch"]
        pl_module.log("train/loss_epoch", avg, on_step=False, on_epoch=True)
    
class Plotting(Callback):

    def __init__(self):
        super().__init__()
        self.fixed_batch = None

    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        _raw_batch = next(iter(pl_module.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=10)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, photon_rate_output, hkl_batch, dials_reference = pl_module(self.fixed_batch, log_images=True)
            dials_id = dials_reference[:,-1]
            photon_rate, profile, samples_surrogate_posterior, samples_predicted_structure_factor = photon_rate_output
            batch_size = shoeboxes_batch.size(0)

            # Set font sizes
            plt.rcParams.update({
                'font.size': 35,
                'axes.titlesize': 35,
                'axes.labelsize': 35,
            })

            fig, axes = plt.subplots(3, batch_size, figsize=(5*batch_size, 15))
            
            for b in range(batch_size):
                im1 = axes[0,b].imshow(shoeboxes_batch[b,:, -1].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                cbar1 = plt.colorbar(im1, ax=axes[0,b], ticks=[im1.get_array().min(), im1.get_array().max()])
                cbar1.ax.tick_params(labelsize=12)
                axes[0,b].set_title(f'Batch {int(dials_id[b])} \n Raw Shoebox', pad=10, fontsize=16)
                axes[0,b].axis('off')
                
                im2 = axes[1,b].imshow(profile[b,0,:].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                cbar2 = plt.colorbar(im2, ax=axes[1,b], ticks=[im2.get_array().min(), im2.get_array().max()])
                cbar2.ax.tick_params(labelsize=12)
                axes[1,b].set_title(f'Profile')
                axes[1,b].axis('off')
                
                im3 = axes[2,b].imshow(photon_rate[b,0,:].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                cbar3 = plt.colorbar(im3, ax=axes[2,b], ticks=[im3.get_array().min(), im3.get_array().max()])
                cbar3.ax.tick_params(labelsize=12)
                axes[2,b].set_title(f'Photon Rate')
                axes[2,b].axis('off')
            
            axes[0,0].set_ylabel('Raw Shoebox', size=16, labelpad=20)
            axes[1,0].set_ylabel('Profile', size=16, labelpad=20)
            axes[2,0].set_ylabel('Rate', size=16, labelpad=20)
            
            plt.tight_layout(h_pad=0.8, w_pad=0.5)
            
            pl_module.logger.experiment.log({
                "Profiles": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
            
            plt.close()
            
            fig, axes = plt.subplots()




class CorrelationPlotting(Callback):
    
    def __init__(self):
        super().__init__()
        self.dials_reference = None
        self.fixed_batch = None
    
    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        self.dials_reference = rs.read_mtz(pl_module.settings.merged_mtz_file_path)#.to(device)
        # self.dials_reference = _dials_reference.columns.tolist()
        print("Read mtz successfully: ", self.dials_reference["F"].shape)
        print("head hkl", self.dials_reference.get_hkls().shape)

        _raw_batch = next(iter(pl_module.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=100)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

    def on_train_epoch_end(self, trainer, pl_module):
        shoeboxes_batch, (photon_rate, profile, samples_surrogate_posterior, samples_predicted_structure_factor), hkl_batch, _dials_reference = pl_module(self.fixed_batch, log_images=True)
        print("full smaples", samples_surrogate_posterior.shape) 
        print("gathered ", samples_predicted_structure_factor.shape)
        rasu_ids = pl_module.surrogate_posterior.rac.rasu_ids[0]
        ordered_samples = pl_module.surrogate_posterior.rac.gather(
            source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
        ).T
        print("get_hkl from dials", self.dials_reference.get_hkls().shape)
        print("dials ref", self.dials_reference["F"].shape)
        print("ordered samples", ordered_samples.shape)

        fig, axes = plt.subplots()
        axes.scatter(self.dials_reference["F"].squeeze().to_numpy(), ordered_samples[0,:].squeeze().cpu().detach().numpy(), alpha=0.7)
        axes.set_xlabel('Modeled (True) Value')
        axes.set_ylabel('Predicted Value')
        pl_module.logger.experiment.log({
                "Correlation": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
            
        plt.close()
        print("logged images")
