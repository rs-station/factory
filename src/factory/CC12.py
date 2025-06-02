import torch
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from model import *

wandb.init(project="CC12")
wandb_logger = WandbLogger(project="CC12", name="CC12-Training")
artifact = wandb.use_artifact("flaviagiehr-harvard-university/full-model/best_model:latest", type="model")
artifact_dir = artifact.download()

ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
assert len(ckpt_files) == 1, "Expected only one checkpoint file."
checkpoint_path = os.path.join(artifact_dir, ckpt_files[0])

checkpoint = torch.load(checkpoint_path, map_location="cuda")
state_dict = checkpoint['state_dict']

filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith("surrogate_posterior.")
}

settings = Settings()
loss_settings = LossSettings()
data_loader_settings = data_loader.DataLoaderSettings(data_directory=settings.data_directory,
                            data_file_names=settings.data_file_names,
                            validation_set_split=0.5, test_set_split=0
                            )
_dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
_dataloader.load_data_()

def _train_model(model_no, dataloader, settings, loss_settings):
    model = Model(settings=settings, loss_settings=loss_settings, dataloader=dataloader)
    print("model type:", type(model))

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("Missing keys (expected for surrogate posterior):", missing_keys)
    print("Unexpected keys (should be none):", unexpected_keys)

    for name, param in model.named_parameters():
        if not name.startswith("surrogate_posterior."):
            param.requires_grad = False
    print(list(state_dict.keys())[:10])

    trainer = Trainer(
        logger=wandb_logger, 
        max_epochs=20, 
        # log_every_n_steps=5, 
        val_check_interval=None,
        accelerator="auto", 
        enable_checkpointing=False, 
        default_root_dir="/tmp",
        # callbacks=[Plotting(), LossLogging(), CorrelationPlotting(), checkpoint_callback] # ScalePlotting()
    )
    if model_no == 1:
        trainer.fit(model, train_dataloaders=_dataloader.load_data_set_batched_by_image(
            data_set_to_load=_dataloader.train_data_set
        ))
    else:
        trainer.fit(model, train_dataloaders=_dataloader.load_data_set_batched_by_image(
            data_set_to_load=_dataloader.validation_data_set
        ))

    print("finished training for model", model_no)
    return model

model1 = _train_model(1, _dataloader, settings, loss_settings)
model2 = _train_model(2, _dataloader, settings, loss_settings)

model1.eval()
model2.eval()

outputs_1 = []
outputs_2 = []

device = next(model1.parameters()).device
print("start testing")
# with torch.no_grad():
#     for batch in _dataloader.load_data_set_batched_by_image(data_set_to_load=_dataloader.test_data_set):
#         batch = [batch_item.to(device) for batch_item in batch]

#         out1 = model1(batch)[5]
#         out2 = model2(batch)[5]

#         outputs_1.append(out1.cpu())
#         outputs_2.append(out2.cpu())

def get_reliable_mask(model):

    if hasattr(model.surrogate_posterior, 'reliable_observations_mask'):
        return model.surrogate_posterior.reliable_observations_mask(min_observations=7)
    else:
        return model.surrogate_posterior.observed

reliable_mask_1 = get_reliable_mask(model1)
reliable_mask_2 = get_reliable_mask(model2)

common_reliable_mask = reliable_mask_1 & reliable_mask_2

# Plot histogram of observation counts
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
observation_counts_1 = model1.surrogate_posterior.observation_count.detach().cpu().numpy()
observation_counts_2 = model2.surrogate_posterior.observation_count.detach().cpu().numpy()

ax_hist.hist([observation_counts_1, observation_counts_2], 
             bins=30, 
             alpha=0.5, 
             label=['Model 1', 'Model 2'])
ax_hist.set_xlabel('Number of Observations')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title('Distribution of Observation Counts')
ax_hist.legend()

wandb.log({"Observation Counts Histogram": wandb.Image(fig_hist)})
plt.close(fig_hist)

structure_factors_1 = torch.cat([model1.surrogate_posterior.mean], dim=0).flatten().detach().cpu().numpy()
structure_factors_2 = torch.cat([model2.surrogate_posterior.mean], dim=0).flatten().detach().cpu().numpy()

structure_factors_1 = structure_factors_1[common_reliable_mask.cpu().numpy()]
structure_factors_2 = structure_factors_2[common_reliable_mask.cpu().numpy()]

def compute_cc12(a, b):
    # a = a.detach().cpu().numpy()
    # b = b.detach().cpu().numpy()

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    numerator = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a)**2) * np.sum((b - mean_b)**2))

    return numerator / (denominator + 1e-8)

cc12 = compute_cc12(structure_factors_1, structure_factors_2)
print("CC12:", cc12)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(structure_factors_1, 
          structure_factors_2, 
          alpha=0.5)
ax.set_xlabel("Model 1 Structure Factors")
ax.set_ylabel("Model 2 Structure Factors")
ax.set_title(f"CC12 = {cc12:.3f}")

# min_val = min(structure_factors_1.min(), structure_factors_2.min())
# max_val = max(structure_factors_1.max(), structure_factors_2.max())
# ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

wandb.log({"CC12 Scatter Plot": wandb.Image(fig)})
wandb.log({"CC12": cc12})

plt.close()
print("finished successfully!")
