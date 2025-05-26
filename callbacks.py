import torch
import reciprocalspaceship as rs
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
import wandb

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt
import numpy as np

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

class CorrelationPlotting(Callback):
    
    def __init__(self):
        super().__init__()
        self.dials_reference = None
        self.fixed_batch = None
        self.fixed_hkl_indices_for_surrogate_posterior = None
    
    
    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        self.dials_reference = rs.read_mtz(pl_module.settings.merged_mtz_file_path)
        print("Read mtz successfully: ", self.dials_reference["F"].shape)
        print("head hkl", self.dials_reference.get_hkls().shape)

        _raw_batch = next(iter(pl_module.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=100)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)
        self.fixed_hkl_indices_for_surrogate_posterior = torch.randint(0, len(self.dials_reference.get_hkls()), (5,))

    def on_train_epoch_end(self, trainer, pl_module):
        shoeboxes_batch, (photon_rate, profile, samples_surrogate_posterior, samples_predicted_structure_factor), hkl_batch, _dials_reference = pl_module(self.fixed_batch, log_images=True)
        
        samples_surrogate_posterior = pl_module.surrogate_posterior.mean
        print("full samples", samples_surrogate_posterior.shape) 
        print("gathered ", samples_predicted_structure_factor.shape)
        rasu_ids = pl_module.surrogate_posterior.rac.rasu_ids[0]
        print("rasu id (all teh same as the first one?):",pl_module.surrogate_posterior.rac.rasu_ids)
        # extract only the surrogate mean that matces the F available in the mtz

        print("observed attribute", pl_module.surrogate_posterior.observed.sum())
        mask_observed_indexed_as_dials_reference = pl_module.surrogate_posterior.rac.gather(
            source=pl_module.surrogate_posterior.observed, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
        )
        print("mask_observed_indexed_as_dials_reference sum", mask_observed_indexed_as_dials_reference)

        ordered_samples = pl_module.surrogate_posterior.rac.gather(  # gather wants source to be (rac_size, )
            source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
        ) #(reflections_from_mtz, number_of_samples)

        print("get_hkl from dials", self.dials_reference.get_hkls().shape)
        print("dials ref", self.dials_reference["F"].shape)
        print("ordered samples", ordered_samples.shape)

        # num_hkl, num_samples = ordered_samples.shape
        # random_indices = torch.randint(0, num_samples, (num_hkl,))
        # random_sampled = ordered_samples[torch.arange(num_hkl), random_indices]

        # mean_sampled = ordered_samples.mean(dim=1)

        # mask out nonobserved reflections
        print("sum of (reindexed) observed reflections", mask_observed_indexed_as_dials_reference.sum())
        _masked_dials_reference = self.dials_reference[mask_observed_indexed_as_dials_reference.cpu().detach().numpy()]
        print("masked dials reference", _masked_dials_reference["F"].shape)
        _masked_ordered_samples = ordered_samples[mask_observed_indexed_as_dials_reference]
        print("masked ordered samples", _masked_ordered_samples.shape)

        fig, axes = plt.subplots()
        axes.scatter(_masked_dials_reference["F"].squeeze().to_numpy(), _masked_ordered_samples.squeeze().cpu().detach().numpy(), marker='x', s=10, alpha=0.5)

        pl_module.logger.experiment.log({
                "Correlation": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
        plt.close()

        _hkl_plot = self.fixed_hkl_indices_for_surrogate_posterior
        pl_module.logger.experiment.log({
            f"structure_factor_evolution/{self.dials_reference.get_hkls()[_hkl_plot[0]]}": ordered_samples[0],
            f"structure_factor_evolution/{self.dials_reference.get_hkls()[_hkl_plot[1]]}": ordered_samples[1],
            f"structure_factor_evolution/{self.dials_reference.get_hkls()[_hkl_plot[2]]}": ordered_samples[2],
            f"structure_factor_evolution/{self.dials_reference.get_hkls()[_hkl_plot[3]]}": ordered_samples[3],
            f"structure_factor_evolution/{self.dials_reference.get_hkls()[_hkl_plot[4]]}": ordered_samples[4]
        })

        # Plot histograms of posterior samples with true values
        fig, axes = plt.subplots(1, len(self.fixed_hkl_indices_for_surrogate_posterior), figsize=(5*len(self.fixed_hkl_indices_for_surrogate_posterior), 5))
        
        # Convert F values to numpy array for easier indexing
        true_values = self.dials_reference["F"].to_numpy()
        
        for ax_index, hkl_index in enumerate(self.fixed_hkl_indices_for_surrogate_posterior):
            reflection_samples = ordered_samples[hkl_index].cpu().detach().numpy()
            true_value = true_values[hkl_index]
            
            axes[ax_index].axvline(reflection_samples, alpha=0.7, label='Posterior Samples')
            axes[ax_index].axvline(true_value, color='r', linestyle='--', label='True Value')
            axes[ax_index].set_title(f'HKL {self.dials_reference.get_hkls()[hkl_index]} \n {reflection_samples.shape} samples')
            axes[ax_index].set_xlabel('Amplitude')
            if ax_index == 0:  # Only show y-label for the first plot
                axes[ax_index].set_ylabel('Count')
            # axes[i].legend()
        
        plt.tight_layout()
        
        pl_module.logger.experiment.log({
            "Posterior_Samples_with_True_Values": wandb.Image(plt),
            "epoch": trainer.current_epoch
        })
        
        plt.close()
        print("logged images")

class ScalePlotting(Callback):
    
    def __init__(self):
        super().__init__()
        self.fixed_batch = None
        self.scale_history = {
            'loc': [],
            'scale': []
        }
    
    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        _raw_batch = next(iter(pl_module.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=100)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch, dials_reference_batch = self.fixed_batch
            
            standardized_counts = shoeboxes_batch[:,:,-1].reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21)
            shoebox_representation = pl_module.shoebox_encoder(standardized_counts.reshape(shoeboxes_batch.shape[0], 1, 3, 21, 21), mask=dead_pixel_mask_batch)
            metadata_representation = pl_module.metadata_encoder(pl_module._cut_metadata(metadata_batch).float())
            joined_shoebox_representation = shoebox_representation + metadata_representation
            image_representation = torch.max(joined_shoebox_representation, dim=0, keepdim=True)[0]

            scale_distribution = pl_module.compute_scale(image_representation=image_representation, metadata_representation=metadata_representation)
            
            loc = scale_distribution.loc.cpu().numpy()
            scale = scale_distribution.scale.cpu().numpy()
            
            print("Shape of loc:", loc.shape)
            print("Shape of scale:", scale.shape)
            print("Values of loc:", loc)
            print("Values of scale:", scale)
            
            self.scale_history['loc'].append(loc)
            self.scale_history['scale'].append(scale)
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 5))
            
            x = np.linspace(min(loc - 3*scale), max(loc + 3*scale), 1000)
            
            # Plot scale normal distribution
            for i in range(len(loc)):
                y = np.exp(-0.5 * ((x - loc[i])/scale[i])**2) / (scale[i] * np.sqrt(2*np.pi))
                axes[0].plot(x, y, alpha=0.3)
            axes[0].set_title('Scale Distribution Location', fontsize=12, pad=10)
            axes[0].set_xlabel('Value', fontsize=10)
            axes[0].set_ylabel('Probability Density', fontsize=10)
            axes[0].tick_params(axis='both', which='major', labelsize=8)
            
            # Plot scale parameters
            axes[1].hist(loc, bins='auto', alpha=0.7)
            axes[1].set_title("Histogram of Location Parameters", fontsize=10)

            axes[2].hist(scale, bins='auto', alpha=0.7)
            axes[2].set_title("Histogram of Scale Parameters", fontsize=10)
            
            plt.tight_layout(pad=2.0)
            
            pl_module.logger.experiment.log({
                "Scale_Distribution_Parameters": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
            
            plt.close()

            pl_module.logger.experiment.log({
                "Scale_loc_evolution/predicted": loc,
                "Scale_scale_evolution/predicted": scale,
            })
            
            background_samples = pl_module.background_distribution(shoebox_representation).mean
            background_samples_variance = (pl_module.background_distribution(shoebox_representation).scale** 2 * (1 - 2 / torch.pi)).mean()
            dials_background = dials_reference_batch[:,10]
            print("dials reference shape", dials_reference_batch.shape, "background", dials_background.shape)
            fig, axes = plt.subplots(1,2)
            axes[0].scatter(range(len(background_samples)), background_samples.cpu().numpy())
            axes[0].set_title(f"samples bkg, var mean={background_samples_variance.cpu().numpy()}")
            axes[1].scatter(range(len(dials_background)), dials_background.cpu().numpy(),marker="x") 
            # axes[1].set_title(f"dials bkg, var mean={dials_background[-1].cpu().numpy().mean()}")
            # axes[1].scatter(range(len(dials_background[0])), dials_background[0].cpu().numpy(),marker="x") 
            # axes[1].scatter(range(len(dials_background[1])), dials_background[1].cpu().numpy()) 
            # axes[1].scatter(range(len(dials_background[2])), dials_background[2].cpu().numpy())       

            plt.tight_layout()
            
            pl_module.logger.experiment.log({
                "Background samples vs dials": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
            
            plt.close()

    def on_train_end(self, trainer, pl_module):
        unmerged_mtz = rs.read_mtz(pl_module.settings.unmerged_mtz_file_path)
        scale_dials = unmerged_mtz["SCALEUSED"]
        pl_module.logger.experiment.log({
                "scale/mean_dials": scale_dials.mean(),
                "scale/max_dials": scale_dials.max(),
                "scale/min_dials": scale_dials.min()
            })

