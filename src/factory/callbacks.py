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

    def __init__(self, dataloader):
        super().__init__()
        self.fixed_batch = None
        self.dataloader = dataloader

    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=10)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, photon_rate_output, hkl_batch = pl_module(self.fixed_batch, log_images=True)
            # dials_id = dials_reference[:,-1]
            photon_rate, profile, samples_surrogate_posterior, samples_predicted_structure_factor = photon_rate_output
            batch_size = shoeboxes_batch.size(0)

            fig, axes = plt.subplots(3, batch_size, figsize=(5*batch_size, 15))
            
            for b in range(batch_size):
                im1 = axes[0,b].imshow(shoeboxes_batch[b,:, -1].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                cbar1 = plt.colorbar(im1, ax=axes[0,b], ticks=[im1.get_array().min(), im1.get_array().max()])
                cbar1.ax.tick_params(labelsize=12)
                axes[0,b].set_title(f'Batch  \n Raw Shoebox', pad=10, fontsize=16)
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
    
    def __init__(self, dataloader):
        super().__init__()
        self.dials_reference = None
        self.fixed_batch = None
        self.fixed_hkl_indices_for_surrogate_posterior = None
        self.dataloader = dataloader
    
    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        self.dials_reference = rs.read_mtz(pl_module.settings.merged_mtz_file_path)
        print("Read mtz successfully: ", self.dials_reference["F"].shape)
        print("head hkl", self.dials_reference.get_hkls().shape)

        _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=100)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)
        self.fixed_hkl_indices_for_surrogate_posterior = torch.randint(0, len(self.dials_reference.get_hkls()), (5,))

    def on_train_epoch_end(self, trainer, pl_module):
        shoeboxes_batch, (photon_rate, profile, samples_surrogate_posterior, samples_predicted_structure_factor), hkl_batch = pl_module(self.fixed_batch, log_images=True)
        
        samples_surrogate_posterior = pl_module.surrogate_posterior.mean
        print("full samples", samples_surrogate_posterior.shape) 
        print("gathered ", samples_predicted_structure_factor.shape)
        rasu_ids = pl_module.surrogate_posterior.rac.rasu_ids[0]
        print("rasu id (all teh same as the first one?):",pl_module.surrogate_posterior.rac.rasu_ids)
        
        if hasattr(pl_module.surrogate_posterior, 'reliable_observations_mask'):
            reliable_mask = pl_module.surrogate_posterior.reliable_observations_mask(min_observations=5)
            
        else:
            reliable_mask = pl_module.surrogate_posterior.observed

        print("observed attribute", pl_module.surrogate_posterior.observed.sum())
        mask_observed_indexed_as_dials_reference = pl_module.surrogate_posterior.rac.gather(
            source=pl_module.surrogate_posterior.observed & reliable_mask, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
        )

        ordered_samples = pl_module.surrogate_posterior.rac.gather(  # gather wants source to be (rac_size, )
            source=samples_surrogate_posterior.T, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
        ) #(reflections_from_mtz, number_of_samples)

        print("get_hkl from dials", self.dials_reference.get_hkls().shape)
        print("dials ref", self.dials_reference["F"].shape)
        print("ordered samples", ordered_samples.shape)

        # mask out nonobserved reflections
        print("sum of (reindexed) observed reflections", mask_observed_indexed_as_dials_reference.sum())
        _masked_dials_reference = self.dials_reference[mask_observed_indexed_as_dials_reference.cpu().detach().numpy()]
        print("masked dials reference", _masked_dials_reference["F"].shape)
        _masked_ordered_samples = ordered_samples[mask_observed_indexed_as_dials_reference]
        print("masked ordered samples", _masked_ordered_samples.shape)

        if hasattr(pl_module.surrogate_posterior, 'reliable_observations_mask'):
            observations_count = pl_module.surrogate_posterior.observation_count

            ordered_observations_count = pl_module.surrogate_posterior.rac.gather(
                source=observations_count.T, rasu_id=rasu_ids, H=self.dials_reference.get_hkls()
            )
            _masked_ordered_observations_count = ordered_observations_count[mask_observed_indexed_as_dials_reference]

            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            ax_hist.hist(_masked_ordered_observations_count.cpu().detach().numpy(), 
                        bins=30, 
                        alpha=0.5, 
                        label=['# observations'])
            ax_hist.set_xlabel('Number of Observations')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title('Distribution of Observation Counts')
            ax_hist.legend()

            wandb.log({"Correlation Plotting: Observation Counts Histogram": wandb.Image(fig_hist)})
            plt.close(fig_hist)

        fig, axes = plt.subplots()
        axes.scatter(_masked_dials_reference["F"].squeeze().to_numpy(), _masked_ordered_samples.squeeze().cpu().detach().numpy(), marker='x', s=10, alpha=0.5)

        pl_module.logger.experiment.log({
                "Correlation": wandb.Image(plt),
                "epoch": trainer.current_epoch
            })
        plt.close()

class ScalePlotting(Callback):
    
    def __init__(self, dataloader):
        super().__init__()
        self.fixed_batch = None
        self.scale_history = {
            'loc': [],
            'scale': []
        }
        self.dataloader = dataloader
    
    def on_fit_start(self, trainer, pl_module):
        device = next(pl_module.parameters()).device
        _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=100)))
        _raw_batch = [_batch_item.to(device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, metadata_batch, dead_pixel_mask_batch, counts_batch, hkl_batch = self.fixed_batch
            
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

            pl_module.logger.experiment.log({
                "Scale_loc_evolution/predicted": loc,
                "Scale_scale_evolution/predicted": scale,
            })
            
            # background_samples = pl_module.background_distribution(shoebox_representation).mean
            # background_samples_variance = (pl_module.background_distribution(shoebox_representation).scale** 2 * (1 - 2 / torch.pi)).mean()
            # dials_background = dials_reference_batch[:,10]
            # print("dials reference shape", dials_reference_batch.shape, "background", dials_background.shape)
            # fig, axes = plt.subplots(1,2)
            # axes[0].scatter(range(len(background_samples)), background_samples.cpu().numpy())
            # axes[0].set_title(f"samples bkg, var mean={background_samples_variance.cpu().numpy()}")
            # axes[1].scatter(range(len(dials_background)), dials_background.cpu().numpy(),marker="x")      

            # plt.tight_layout()
            
            # pl_module.logger.experiment.log({
            #     "Background samples vs dials": wandb.Image(plt),
            #     "epoch": trainer.current_epoch
            # })
            
            # plt.close()

    def on_train_end(self, trainer, pl_module):
        unmerged_mtz = rs.read_mtz(pl_module.settings.unmerged_mtz_file_path)
        scale_dials = unmerged_mtz["SCALEUSED"]
        pl_module.logger.experiment.log({
                "scale/mean_dials": scale_dials.mean(),
                "scale/max_dials": scale_dials.max(),
                "scale/min_dials": scale_dials.min()
            })

