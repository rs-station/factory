
import torch
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from model import *
from pytorch_lightning import Trainer

wandb.init(project="CC12")
# wandb_logger = WandbLogger(project="CC12", name="CC12-Training")
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
                            validation_set_split=0.4, test_set_split=0.2
                            )
_dataloader = data_loader.CrystallographicDataLoader(settings=data_loader_settings)
_dataloader.load_data_()
dataloader1 = _dataloader.load_data_set_batched_by_image(data_set_to_load=_dataloader.train_data_set)
dataloader2 = _dataloader.load_data_set_batched_by_image(data_set_to_load=_dataloader.validation_data_set)

def _train_model(dataloader, settings, loss_settings):
    model = Model(settings=settings, loss_settings=loss_settings, dataloader=dataloader)
    print("model type:", type(model))

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("Missing keys (expected for netother):", missing_keys)
    print("Unexpected keys (should be empty):", unexpected_keys)

    for name, param in model.named_parameters():
        if not name.startswith("surrogate_posterior."):
            param.requires_grad = False
    print(list(state_dict.keys())[:10])

    trainer = Trainer(
        logger=wandb_logger, 
        max_epochs=100, 
        # log_every_n_steps=5, 
        # val_check_interval=3, 
        accelerator="auto", 
        enable_checkpointing=False, 
        default_root_dir="/tmp",
        # callbacks=[Plotting(), LossLogging(), CorrelationPlotting(), checkpoint_callback] # ScalePlotting()
    )
    trainer.fit(model, train_dataloaders=dataloader)
    return model

model1 = _train_model(dataloader1, settings, loss_settings)
model2 = _train_model(dataloader2, settings, loss_settings)

model1.eval()
model2.eval()

outputs_1 = []
outputs_2 = []

device = next(model1.parameters()).device

with torch.no_grad():
    for batch in _dataloader.load_data_set_batched_by_image(data_set_to_load=_dataloader.test_data_set):
        batch = [batch_item.to(device) for batch_item in batch]

        out1 = model1(batch)[5]
        out2 = model2(batch)[5]

        outputs_1.append(out1.cpu())
        outputs_2.append(out2.cpu())

outputs_1 = torch.cat(outputs_1, dim=0)
outputs_2 = torch.cat(outputs_2, dim=0)

def compute_cc12(a, b):
    a = a.numpy()
    b = b.numpy()

    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)

    numerator = np.sum((a - mean_a) * (b - mean_b), axis=0)
    denominator = np.sqrt(np.sum((a - mean_a)**2, axis=0) * np.sum((b - mean_b)**2, axis=0))

    correlation = numerator / (denominator + 1e-8)
    return correlation

cc12 = compute_cc12(outputs_1, outputs_2)
print("CC12:", cc12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(outputs_1.numpy(), outputs_2.numpy(), alpha=0.5)
ax.set_xlabel("Model 1 Outputs")
ax.set_ylabel("Model 2 Outputs")
wandb.logger.experiment.log({"CC12 Scatter Plot": wandb.Image(fig)})
wandb.log({"CC12": cc12})

