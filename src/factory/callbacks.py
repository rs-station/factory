import torch
import reciprocalspaceship as rs
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
import wandb
from phenix_callback import find_peaks_from_model

from torchvision.transforms import ToPILImage
from lightning.pytorch.utilities import grad_norm
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from scipy.optimize import minimize
import matplotlib.cm as cm
import pandas as pd

class PeakHeights(Callback):
    "Compute Anomalous Peak Heights during training."

    def __init__(self):
        super().__init__()
        self.residue_colors = {}  # Maps residue to color
        self.peak_history = {}    # Maps residue to list of (epoch, peak_height)
        self.colormap = cm.get_cmap('tab20') 
        # self.peakheights_log = open("peakheights.out", "a")

    # def __del__(self):
    #     self.peakheights_log.close()

    def _get_color(self, residue):
        # Assign a color to each residue, keep consistent
        if residue not in self.residue_colors:
            idx = len(self.residue_colors) % self.colormap.N
            self.residue_colors[residue] = self.colormap(idx)
        return self.residue_colors[residue]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch%2==0:
            try:
                r_work, r_free, path_to_peaks = find_peaks_from_model(model=pl_module, repo_dir=trainer.logger.experiment.dir)
                pl_module.logger.experiment.log({"r/r_work": r_work, "r/r_free": r_free})

                peaks_df = pd.read_csv(path_to_peaks)
                epoch = trainer.current_epoch

                # Update peak history
                print("organize peaks")
                for _, row in peaks_df.iterrows():
                    residue = row['residue']
                    print(residue)
                    peak_height = row['peakz']
                    print(peak_height)
                    if residue not in self.peak_history:
                        self.peak_history[residue] = []
                    self.peak_history[residue].append((epoch, peak_height))

                # Plot

                print("plot peaks")
                plt.figure(figsize=(10, 6))
                for residue, history in self.peak_history.items():
                    epochs, heights = zip(*history)
                    color = self._get_color(residue)
                    plt.plot(epochs, heights, label=str(residue), color=color)
                plt.xlabel('Epoch')
                plt.ylabel('Peak Height')
                plt.title('Peak Heights per Residue')
                plt.legend(title='Residue', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                # Log to wandb
                pl_module.logger.experiment.log({
                    "PeakHeights": wandb.Image(plt.gcf()),
                    "epoch": epoch
                })
                plt.close()
            except Exception as e:
                print(f"Failed for {epoch}: {e}")#, file=self.peakheights_log, flush=True)
        else:
            return




class LossLogging(Callback):
    
    def __init__(self):
        super().__init__()
        self.loss_per_epoch: float = 0
        self.count_batches: int = 0

    def on_train_epoch_end(self, trainer, pl_module):
        avg = trainer.callback_metrics["train/loss_step_epoch"]
        pl_module.log("train/loss_epoch", avg, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        
        train_loss = trainer.callback_metrics.get("train/loss_step_epoch")
        val_loss = trainer.callback_metrics.get("validation/loss_step_epoch")
        if train_loss is not None and val_loss is not None:
            difference_train_val_loss = train_loss - val_loss
            pl_module.log("loss/train-val", difference_train_val_loss, on_step=False, on_epoch=True)
        else:
            print("Skipping loss/train-val logging: train or validation loss not available this epoch.")
    
class Plotting(Callback):

    def __init__(self, dataloader):
        super().__init__()
        self.fixed_batch = None
        self.dataloader = dataloader
    
    def on_fit_start(self, trainer, pl_module):
        _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=8)))
        _raw_batch = [_batch_item.to(pl_module.device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)
        
        self.fixed_batch_intensity_sum_values = self.fixed_batch[1].to(pl_module.device)[:, 9]
        self.fixed_batch_intensity_sum_variance = self.fixed_batch[1].to(pl_module.device)[:, 10]
        self.fixed_batch_total_counts = self.fixed_batch[3].to(pl_module.device).sum(dim=1)
        print("self.fixed_batch_total_counts", self.fixed_batch_total_counts.shape)

        self.fixed_batch_background_mean = self.fixed_batch[1].to(pl_module.device)[:, 13]

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            shoeboxes_batch, photon_rate_output, hkl_batch, counts_batch = pl_module(self.fixed_batch, verbose_output=True)
            # Extract values from dictionary
            samples_photon_rate = photon_rate_output['photon_rate']
            samples_profile = photon_rate_output['samples_profile']
            samples_background = photon_rate_output['samples_background']
            samples_scale = photon_rate_output['samples_scale']
            samples_predicted_structure_factor = photon_rate_output["samples_predicted_structure_factor"]
            batch_size = shoeboxes_batch.size(0)

            model_intensity = samples_scale * torch.square(samples_predicted_structure_factor)
            model_intensity = torch.mean(model_intensity, dim=1)
            print("model_intensity", model_intensity.shape)

            photon_rate = torch.mean(samples_photon_rate, dim=1)

            print("photon rate", photon_rate.shape)
            print("profile", samples_profile.shape)
            print("background", samples_background.shape)
            print("scale",  samples_scale.shape)

            fig, axes = plt.subplots(
                2, batch_size, 
                figsize=(4*batch_size, 16),  # Keep wide, but reduce height to focus on images
                gridspec_kw={'hspace': 0.05, 'wspace': 0.3}  # Reduce vertical space between rows
            )
            
            im_handles = []

            for b in range(batch_size):
                if len(shoeboxes_batch.shape)>2:
                    im1 = axes[0,b].imshow(shoeboxes_batch[b,:, -1].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                    im2 = axes[1,b].imshow(samples_profile[b,0,:].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                else:
                    im1 = axes[0,b].imshow(shoeboxes_batch[b,:].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())
                    im2 = axes[1,b].imshow(samples_profile[b,0,:].reshape(3,21,21)[1:2].squeeze().cpu().detach().numpy())

                im_handles.append(im1)

                # Build title string with each element on its own row (3 rows)
                title_lines = [
                    f'Raw Shoebox {b}',
                    f'I = {self.fixed_batch_intensity_sum_values[b]:.2f} pm {self.fixed_batch_intensity_sum_variance[b]:.2f}',
                    f'Bkg (mean) = {self.fixed_batch_background_mean[b]:.2f}',
                    f'total counts = {self.fixed_batch_total_counts[b]:.2f}'
                ]
                axes[0,b].set_title('\n'.join(title_lines), pad=5, fontsize=16)
                axes[0,b].axis('off')
                
                title_prf_lines = [
                    f'Profile {b}',
                    f'photon_rate = {photon_rate[b].sum():.2f}', 
                    f'I(F) = {model_intensity[b].item():.2f}',
                    f'Scale = {torch.mean(samples_scale, dim=1)[b].item():.2f}',
                    f'F= {torch.mean(samples_predicted_structure_factor, dim=1)[b].item():.2f}',
                    f'Bkg (mean) = {torch.mean(samples_background, dim=1)[b].item():.2f}'
                ]
                axes[1,b].set_title('\n'.join(title_prf_lines), pad=5, fontsize=16)
                axes[1,b].axis('off')
                
           
            
            plt.tight_layout(h_pad=0.0, w_pad=0.3)  # Remove extra vertical padding
            
            cbar = fig.colorbar(im_handles[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert buffer to PIL Image
            pil_image = Image.open(buf)
            
            pl_module.logger.experiment.log({
                "Profiles": wandb.Image(pil_image),
                "epoch": trainer.current_epoch
            })
            
            plt.close()
            buf.close()

class CorrelationPlottingBinned(Callback):
    
    def __init__(self, dataloader):
        super().__init__()
        self.dials_reference = None
        self.fixed_batch = None
        # self.fixed_hkl_indices_for_surrogate_posterior = None
        self.dataloader = dataloader
    
    def on_fit_start(self, trainer, pl_module):
        print("on fit start callback")
        self.dials_reference = rs.read_mtz(pl_module.model_settings.merged_mtz_file_path)
        self.dials_reference, self.resolution_bins = self.dials_reference.assign_resolution_bins(bins=10)
        print("Read mtz successfully: ", self.dials_reference["F"].shape)
        print("head hkl", self.dials_reference.get_hkls().shape)

        # _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=10000)))
        # _raw_batch = [_batch_item.to(pl_module.device) for _batch_item in _raw_batch]
        # self.fixed_batch = tuple(_raw_batch)
        # self.fixed_hkl_indices_for_surrogate_posterior = torch.randint(0, len(self.dials_reference.get_hkls()), (5,))

    def on_train_epoch_end(self, trainer, pl_module):
        print("on train epoch end in callback")
        try:
            # shoeboxes_batch, photon_rate_output, hkl_batch = pl_module(self.fixed_batch, log_images=True)
        

            samples_surrogate_posterior = pl_module.surrogate_posterior.mean
            print("full samples", samples_surrogate_posterior.shape) 
            # print("gathered ", photon_rate_output['samples_predicted_structure_factor'].shape)
            rasu_ids = pl_module.surrogate_posterior.rac.rasu_ids[0]
            print("rasu id (all teh same as the first one?):",pl_module.surrogate_posterior.rac.rasu_ids)
            
            if hasattr(pl_module.surrogate_posterior, 'reliable_observations_mask'):
                reliable_mask = pl_module.surrogate_posterior.reliable_observations_mask(min_observations=50)
                print("take reliable mask for correlation plotting")
                
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

            print("check output of assign resolution bins:")
            print( self.dials_reference.shape)
            print( self.dials_reference.head())
            # mask out nonobserved reflections
            print("sum of (reindexed) observed reflections", mask_observed_indexed_as_dials_reference.sum())
            _masked_dials_reference = self.dials_reference[mask_observed_indexed_as_dials_reference.cpu().detach().numpy()]
            print("masked dials reference", _masked_dials_reference["F"].shape)
            _masked_ordered_samples = ordered_samples[mask_observed_indexed_as_dials_reference]
            print("masked ordered samples", _masked_ordered_samples.shape)

            print("self.resolution_bins", self.resolution_bins)

            # _masked_resolution_bins = self.resolution_bins[mask_observed_indexed_as_dials_reference.cpu().detach().numpy()]
            if _masked_ordered_samples.numel() == 0:
                print("Warning: No observed reflections after masking")
                return

            print("Creating correlation plot...")

            n_bins = len(self.resolution_bins)
            n_rows = 2
            n_cols = int(np.ceil(n_bins / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten() if n_bins > 1 else [axes]

            for i, bin_index in enumerate(self.resolution_bins):
                bin_mask = _masked_dials_reference["bin"] == i
                print("bin maks", sum(bin_mask))
                reference_to_plot = _masked_dials_reference[bin_mask]["F"].squeeze().to_numpy()
                model_to_plot = _masked_ordered_samples[np.array(bin_mask)].squeeze().cpu().detach().numpy()

            
                # Remove any NaN or Inf values
                valid_mask = ~(np.isnan(reference_to_plot) | np.isnan(model_to_plot) | 
                            np.isinf(reference_to_plot) | np.isinf(model_to_plot))

                if not np.any(valid_mask):
                    print("Warning: No valid data points after filtering")
                    continue
                
                reference_to_plot = reference_to_plot[valid_mask]
                model_to_plot = model_to_plot[valid_mask]
                
                print(f"After filtering - valid data points: {len(reference_to_plot)}")
            
                scatter = axes[i].scatter(reference_to_plot, 
                                 model_to_plot, 
                                 marker='x', s=10, alpha=0.5, c="black")
            
            # Fit a straight line through the origin using numpy polyfit
            # For line through origin, we fit y = mx (no intercept)
            # try:
            #     fitted_slope = np.polyfit(reference_to_plot, model_to_plot, 1, w=None, cov=False)[0]
            #     print(f"Polyfit successful, slope: {fitted_slope:.6f}")
            # except np.linalg.LinAlgError as e:
            #     print(f"Polyfit failed with LinAlgError: {e}")
            #     print("Falling back to manual calculation")
            #     # Fallback to manual calculation for line through origin
            #     fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
            #     print(f"Fallback slope: {fitted_slope:.6f}")
            # except Exception as e:
            #     print(f"Polyfit failed with other error: {e}")
            #     # Fallback to manual calculation
            #     fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
            #     print(f"Fallback slope: {fitted_slope:.6f}")
            
            # # Plot the fitted line
            # x_range = np.linspace(reference_to_plot.min(), reference_to_plot.max(), 100)
            # y_fitted = fitted_slope * x_range
            # axes.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted line: y = {fitted_slope:.4f}x')
            
                # Set log-log scale
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
            
                try:
                    correlation = np.corrcoef(reference_to_plot, 
                                        model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))[0,1]
                    print("reference_to_plot", reference_to_plot)
                    print("model to plot scaled", model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))
                except Exception as e:
                    print(f"skip correlation computataion {e}")
                # print(f"Fitted slope: {fitted_slope:.4f}")
                axes[i].set_xlabel('Dials Reference F')
                axes[i].set_ylabel('Predicted F')
                axes[i].set_title(f'bin {bin_index} (r={correlation:.3f}')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            

            pl_module.logger.experiment.log({
                "Correlation/structure_factors_binned": wandb.Image(fig),
                # "Correlation/correlation_coefficient": correlation,
                "epoch": trainer.current_epoch
            })
            plt.close(fig)
            print("logged image")
            
        except Exception as e:
            print(f"Error in correlation plotting: {str(e)}")
            import traceback
            traceback.print_exc()

class CorrelationPlotting(Callback):
    
    def __init__(self, dataloader):
        super().__init__()
        self.dials_reference = None
        self.fixed_batch = None
        self.dataloader = dataloader
    
    def on_fit_start(self, trainer, pl_module):
        print("on fit start callback")
        self.dials_reference = rs.read_mtz(pl_module.model_settings.merged_mtz_file_path)
        print("Read mtz successfully: ", self.dials_reference["F"].shape)
        print("head hkl", self.dials_reference.get_hkls().shape)



    def on_train_epoch_end(self, trainer, pl_module):
        print("on train epoch end in callback")
        try:
            # shoeboxes_batch, photon_rate_output, hkl_batch = pl_module(self.fixed_batch, log_images=True)
            
            samples_surrogate_posterior = pl_module.surrogate_posterior.mean
            print("full samples", samples_surrogate_posterior.shape) 
            # print("gathered ", photon_rate_output['samples_predicted_structure_factor'].shape)
            rasu_ids = pl_module.surrogate_posterior.rac.rasu_ids[0]
            print("rasu id (all teh same as the first one?):",pl_module.surrogate_posterior.rac.rasu_ids)
            
            if hasattr(pl_module.surrogate_posterior, 'reliable_observations_mask'):
                reliable_mask = pl_module.surrogate_posterior.reliable_observations_mask(min_observations=50)
                print("take reliable mask for correlation plotting")
                
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

            if _masked_ordered_samples.numel() == 0:
                print("Warning: No observed reflections after masking")
                return

            print("Creating correlation plot...")
            reference_to_plot = _masked_dials_reference["F"].squeeze().to_numpy()
            model_to_plot = _masked_ordered_samples.squeeze().cpu().detach().numpy()
            
            # Remove any NaN or Inf values
            valid_mask = ~(np.isnan(reference_to_plot) | np.isnan(model_to_plot) | 
                          np.isinf(reference_to_plot) | np.isinf(model_to_plot))
            
            if not np.any(valid_mask):
                print("Warning: No valid data points after filtering")
                return
                
            reference_to_plot = reference_to_plot[valid_mask]
            model_to_plot = model_to_plot[valid_mask]
            
            print(f"After filtering - valid data points: {len(reference_to_plot)}")
            
            fig, axes = plt.subplots(figsize=(10, 10))
            scatter = axes.scatter(reference_to_plot, 
                                 model_to_plot, 
                                 marker='x', s=10, alpha=0.5, c="black")
            
            # Fit a straight line through the origin using numpy polyfit
            # For line through origin, we fit y = mx (no intercept)
            try:
                fitted_slope = np.polyfit(reference_to_plot, model_to_plot, 1, w=None, cov=False)[0]
                print(f"Polyfit successful, slope: {fitted_slope:.6f}")
            except np.linalg.LinAlgError as e:
                print(f"Polyfit failed with LinAlgError: {e}")
                print("Falling back to manual calculation")
                # Fallback to manual calculation for line through origin
                fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                print(f"Fallback slope: {fitted_slope:.6f}")
            except Exception as e:
                print(f"Polyfit failed with other error: {e}")
                # Fallback to manual calculation
                fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                print(f"Fallback slope: {fitted_slope:.6f}")
            
            # Plot the fitted line
            x_range = np.linspace(reference_to_plot.min(), reference_to_plot.max(), 100)
            y_fitted = fitted_slope * x_range
            axes.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted line: y = {fitted_slope:.4f}x')
            
            # Set log-log scale
            axes.set_xscale('log')
            axes.set_yscale('log')
            
            correlation = np.corrcoef(reference_to_plot, 
                                    model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))[0,1]
            print("reference_to_plot", reference_to_plot)
            print("model to plot scaled", model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))
            print(f"Fitted slope: {fitted_slope:.4f}")
            axes.set_xlabel('Dials Reference F')
            axes.set_ylabel('Predicted F')
            axes.set_title(f'Correlation Plot (Log-Log) (r={correlation:.3f}, slope={fitted_slope:.4f})')
            axes.legend()
            
            # Add diagonal line
            # min_val = min(axes.get_xlim()[0], axes.get_ylim()[0])
            # max_val = max(axes.get_xlim()[1], axes.get_ylim()[1])
            # axes.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

            pl_module.logger.experiment.log({
                "Correlation/structure_factors": wandb.Image(fig),
                "Correlation/correlation_coefficient": correlation,
                "epoch": trainer.current_epoch
            })
            plt.close(fig)
            print("logged image")
            
        except Exception as e:
            print(f"Error in correlation plotting: {str(e)}")
            import traceback
            traceback.print_exc()

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
        _raw_batch = next(iter(self.dataloader.load_data_for_logging_during_training(number_of_shoeboxes_to_log=10000)))
        _raw_batch = [_batch_item.to(pl_module.device) for _batch_item in _raw_batch]
        self.fixed_batch = tuple(_raw_batch)

        self.fixed_batch_background_mean = self.fixed_batch[1].to(pl_module.device)[:, 13]
        self.fixed_batch_intensity_sum_values = self.fixed_batch[1].to(pl_module.device)[:, 9]
        self.fixed_batch_total_counts = self.fixed_batch[3].to(pl_module.device).sum(dim=1)


    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            try:
            
                shoebox_representation, metadata_representation, image_representation = pl_module._batch_to_representations(batch=self.fixed_batch)


                scale_per_reflection = pl_module.compute_scale(representations=[image_representation, metadata_representation])
                print(scale_per_reflection)
                scale_mean_per_reflection = scale_per_reflection.mean.cpu().detach().numpy()
                print(len(scale_mean_per_reflection) )               
                # loc = scale_distribution.loc.cpu().numpy()
                # scale = scale_distribution.scale.cpu().numpy()
                
                plt.figure(figsize=(8,5))
                plt.hist(scale_mean_per_reflection, bins=100, edgecolor="black", alpha=0.6, label="Scale_mean")
                plt.xlabel("mean scale per reflection")
                plt.ylabel("Number of Observations")
                plt.title("Histogram of Per‐Reflection Scale Factors-Model")
                plt.tight_layout()

                pl_module.logger.experiment.log({"scale_histogram": wandb.Image(plt.gcf())})
                plt.close()

                # # Plot histogram for scales > 20000
                # high_scale_values = scale_mean_per_reflection[scale_mean_per_reflection > 20000]
                # if len(high_scale_values) > 0:
                #     plt.figure(figsize=(8,5))
                #     plt.hist(high_scale_values, bins=100, edgecolor="black", alpha=0.6, label="Scale_mean > 20000")
                #     plt.xlabel("mean scale per reflection (> 20000)")
                #     plt.ylabel("Number of Observations")
                #     plt.title("Histogram of Per‐Reflection Scale Factors-Model (> 20000)")
                #     plt.tight_layout()
                #     pl_module.logger.experiment.log({"scale_histogram_high": wandb.Image(plt.gcf())})
                #     plt.close()
                
                background_mean_per_reflection = pl_module.compute_background_distribution(shoebox_representation=shoebox_representation).mean.cpu().detach().numpy()

                plt.figure(figsize=(8,5))
                plt.hist(background_mean_per_reflection, bins=100, edgecolor="black", alpha=0.6, label="Background_mean")
                plt.xlabel("mean background per reflection")
                plt.ylabel("Number of Observations")
                plt.title("Histogram of Per‐Reflection Background -Model")
                plt.tight_layout()

                pl_module.logger.experiment.log({"background_histogram": wandb.Image(plt.gcf())})
                plt.close()

                reference_to_plot = self.fixed_batch_background_mean.cpu().detach().numpy()
                model_to_plot = background_mean_per_reflection.squeeze()
                print("bkg", reference_to_plot.shape, model_to_plot.shape)
                fig, axes = plt.subplots(figsize=(10, 10))
                scatter = axes.scatter(reference_to_plot, 
                                 model_to_plot, 
                                 marker='x', s=10, alpha=0.5, c="black")

                try:
                    fitted_slope = np.polyfit(reference_to_plot, model_to_plot, 1, w=None, cov=False)[0]
                    print(f"Polyfit successful, slope: {fitted_slope:.6f}")
                except np.linalg.LinAlgError as e:
                    print(f"Polyfit failed with LinAlgError: {e}")
                    print("Falling back to manual calculation")
                    # Fallback to manual calculation for line through origin
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                except Exception as e:
                    print(f"Polyfit failed with other error: {e}")
                    # Fallback to manual calculation
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                
                # Plot the fitted line
                x_range = np.linspace(reference_to_plot.min(), reference_to_plot.max(), 100)
                y_fitted = fitted_slope * x_range
                axes.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted line: y = {fitted_slope:.4f}x')
                
                # Set log-log scale
                axes.set_xscale('log')
                axes.set_yscale('log')
                
                correlation = np.corrcoef(reference_to_plot, 
                                        model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))[0,1]
                print("reference_to_plot", reference_to_plot)
                print("model to plot scaled", model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))
                print(f"Fitted slope: {fitted_slope:.4f}")
                axes.set_xlabel('Dials Reference Bkg')
                axes.set_ylabel('Predicted Bkg')
                axes.set_title(f'Correlation Plot (Log-Log) (r={correlation:.3f}, slope={fitted_slope:.4f})')
                axes.legend()
                
                # Add diagonal line
                # min_val = min(axes.get_xlim()[0], axes.get_ylim()[0])
                # max_val = max(axes.get_xlim()[1], axes.get_ylim()[1])
                # axes.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                pl_module.logger.experiment.log({
                    "Correlation/Background": wandb.Image(fig),
                    "Correlation/background_correlation_coefficient": correlation,
                    "epoch": trainer.current_epoch
                })
                plt.close(fig)
                print("logged image")

            except Exception as e:
                print(f"failed ScalePlotting/Bkg with: {e}")


            ################# Plot Intensity Correlation ##############################
            shoeboxes_batch, photon_rate_output, hkl_batch, counts_batch = pl_module(self.fixed_batch, verbose_output=True)
            
            try:
                # Extract values from dictionary
                samples_scale = photon_rate_output['samples_scale']
                samples_predicted_structure_factor = photon_rate_output["samples_predicted_structure_factor"]

                model_intensity = samples_scale * torch.square(samples_predicted_structure_factor)
                model_intensity = torch.mean(model_intensity, dim=1).squeeze() #(batch_size, 1)
                print("model_intensity", model_intensity.shape)


                reference_to_plot = self.fixed_batch_intensity_sum_values.cpu().detach().numpy()
                model_to_plot = model_intensity.cpu().detach().numpy()
                fig, axes = plt.subplots(figsize=(10, 10))
                scatter = axes.scatter(reference_to_plot, 
                                    model_to_plot, 
                                    marker='x', s=10, alpha=0.5, c="black")

                try:
                    fitted_slope = np.polyfit(reference_to_plot, model_to_plot, 1, w=None, cov=False)[0]
                    print(f"Polyfit successful, slope: {fitted_slope:.6f}")
                except np.linalg.LinAlgError as e:
                    print(f"Polyfit failed with LinAlgError: {e}")
                    print("Falling back to manual calculation")
                    # Fallback to manual calculation for line through origin
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                except Exception as e:
                    print(f"Polyfit failed with other error: {e}")
                    # Fallback to manual calculation
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                
                # Plot the fitted line
                x_range = np.linspace(reference_to_plot.min(), reference_to_plot.max(), 100)
                y_fitted = fitted_slope * x_range
                axes.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted line: y = {fitted_slope:.4f}x')
                
                # Set log-log scale
                axes.set_xscale('log')
                axes.set_yscale('log')
                
                correlation = np.corrcoef(reference_to_plot, 
                                        model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))[0,1]
                print("reference_to_plot", reference_to_plot)
                print("model to plot scaled", model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))
                print(f"Fitted slope: {fitted_slope:.4f}")
                axes.set_xlabel('Dials Reference Intensity')
                axes.set_ylabel('Predicted Intensity')
                axes.set_title(f'Correlation Plot (Log-Log) (r={correlation:.3f}, slope={fitted_slope:.4f})')
                axes.legend()
                
                pl_module.logger.experiment.log({
                    "Correlation/Intensity": wandb.Image(fig),
                    "Correlation/intensity_correlation_coefficient": correlation,
                    "epoch": trainer.current_epoch
                })
                plt.close(fig)
                print("logged image")

            except Exception as e:
                print(f"failed ScalePlotting/Bkg with {e}")

            
            ################# Plot Count Correlation ##############################
            try:
                # Extract values from dictionary
                samples_photon_rate = photon_rate_output['photon_rate']

                print("samples_photon_rate (batch, mc, pix)", samples_photon_rate.shape)
                samples_photon_rate = samples_photon_rate.sum(dim=-1)
                print("samples_photon_rate (batch, mc)", samples_photon_rate.shape)
                
                photon_rate_per_sb = torch.mean(samples_photon_rate, dim=1).squeeze() #(batch_size, 1)
                print("photon_rate (batch)", photon_rate_per_sb.shape)


                reference_to_plot = self.fixed_batch_total_counts.cpu().detach().numpy()
                model_to_plot = photon_rate_per_sb.cpu().detach().numpy()
                fig, axes = plt.subplots(figsize=(10, 10))
                scatter = axes.scatter(reference_to_plot, 
                                    model_to_plot, 
                                    marker='x', s=10, alpha=0.5, c="black")

                try:
                    fitted_slope = np.polyfit(reference_to_plot, model_to_plot, 1, w=None, cov=False)[0]
                    print(f"Polyfit successful, slope: {fitted_slope:.6f}")
                except np.linalg.LinAlgError as e:
                    print(f"Polyfit failed with LinAlgError: {e}")
                    print("Falling back to manual calculation")
                    # Fallback to manual calculation for line through origin
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                except Exception as e:
                    print(f"Polyfit failed with other error: {e}")
                    # Fallback to manual calculation
                    fitted_slope = np.sum(reference_to_plot * model_to_plot) / np.sum(reference_to_plot**2)
                    print(f"Fallback slope: {fitted_slope:.6f}")
                
                # Plot the fitted line
                x_range = np.linspace(reference_to_plot.min(), reference_to_plot.max(), 100)
                y_fitted = fitted_slope * x_range
                axes.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted line: y = {fitted_slope:.4f}x')
                
                # Set log-log scale
                axes.set_xscale('log')
                axes.set_yscale('log')
                
                correlation = np.corrcoef(reference_to_plot, 
                                        model_to_plot*np.max(reference_to_plot)/np.max(model_to_plot))[0,1]
                axes.set_xlabel('Dials Reference Intensity')
                axes.set_ylabel('Predicted Intensity')
                axes.set_title(f'Correlation Plot (Log-Log) (r={correlation:.3f}, slope={fitted_slope:.4f})')
                axes.legend()
                
                pl_module.logger.experiment.log({
                    "Correlation/Counts": wandb.Image(fig),
                    "Correlation/counts_correlation_coefficient": correlation,
                    "epoch": trainer.current_epoch
                })
                plt.close(fig)
                print("logged image")

            except Exception as e:
                print(f"failed corr counts with {e}")

            

    def on_train_end(self, trainer, pl_module):
        unmerged_mtz = rs.read_mtz(pl_module.model_settings.unmerged_mtz_file_path)
        scale_dials = unmerged_mtz["SCALEUSED"]
        pl_module.logger.experiment.log({
                "scale/mean_dials": scale_dials.mean(),
                "scale/max_dials": scale_dials.max(),
                "scale/min_dials": scale_dials.min()
            })

