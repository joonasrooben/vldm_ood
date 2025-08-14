import torch
import sys
sys.path.append('..')
from vdm.src.unetfc2 import UNetFC2

from vdm.src.vdm import VariationalDiffusion
from vdm.src.schedule import LearnableSchedule, LinearSchedule, FixedLinearSchedule
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

class ActivationsDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (Tensor or array): Input data.
            targets (Tensor or array): Target data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

class ActivationsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        train_targets,
        test_data=None,
        test_targets=None,
        val_split=0.2,        # Percentage of training data to use for validation
        batch_size=64,
        random_seed=42        # For reproducibility
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed

        # Convert data to tensors if not already tensors
        self.train_data = torch.tensor(train_data, dtype=torch.float32)
        self.train_targets = torch.tensor(train_targets, dtype=torch.long)

        if test_data is not None and test_targets is not None:
            self.test_data = torch.tensor(test_data, dtype=torch.float32)
            self.test_targets = torch.tensor(test_targets, dtype=torch.long)
        else:
            self.test_data = None
            self.test_targets = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Nothing to do here since data is already provided
        pass

    def setup(self, stage=None):
        # Create the full dataset
        full_dataset = ActivationsDataset(
            data=self.train_data,
            targets=self.train_targets
        )

        # Calculate lengths for train and validation splits
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        # Split the dataset
        torch.manual_seed(self.random_seed)  # For reproducibility
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Create test dataset if test data is provided
        if self.test_data is not None and self.test_targets is not None:
            self.test_dataset = ActivationsDataset(
                data=self.test_data,
                targets=self.test_targets
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size
            )
        else:
            return None
class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.leave = False  # Prevent progress bar from persisting after an epoch
        
ind_name= "imagenet200" #cif100, imagenet200
s_version = "_s2"
test_acts = np.load(f"../openood/OpenOOD/{ind_name}_acts{s_version}/resnet18_{ind_name}_test.npy")
train_acts = np.load(f"../openood/OpenOOD/{ind_name}_acts{s_version}/resnet18_{ind_name}_train.npy")
test_targets = np.load(f"../openood/OpenOOD/{ind_name}_acts{s_version}/resnet18_{ind_name}_test_y.npy")
train_targets= np.load(f"../openood/OpenOOD/{ind_name}_acts{s_version}/resnet18_{ind_name}_train_y.npy")




def denormalize(normalized, basis):
    return basis.min() + (basis.max() - basis.min())*((normalized + 1)/2)
def normalize(unnormalized, basis):
    return (2 * (unnormalized - basis.min()) / (basis.max() - basis.min()) - 1)
def normalize_per_feat(unnormalized, basis):
    return (2 * (unnormalized - basis.min(0)) / (basis.max(0) - basis.min(0)) - 1)

def normalize_std(unnormalized, basis):
    return (unnormalized - basis.mean(0))/basis.std(0)

################# normed per feature
#train_acts_normed =normalize_per_feat(train_acts,train_acts)
#test_acts_normed = normalize_per_feat(test_acts ,train_acts)
train_acts_normed =normalize_std(train_acts,train_acts)
test_acts_normed = normalize_std(test_acts ,train_acts)

train_data = train_acts_normed      # Shape: [num_samples, input_dim]
test_data = test_acts_normed         # (Optional)

# Instantiate the data module
data_module = ActivationsDataModule(
    train_data=train_data,
    train_targets=train_targets,
    test_data=test_data,          # Optional
    test_targets=test_targets,    # Optional
    val_split=0.01,                # Use 20% of training data for validation
    batch_size=128,
    random_seed=42
)

folder_path = "../logs_jl/vdm_run_40/"

vdm_model = VariationalDiffusion(
    backbone=UNetFC2(input_dim=512,
        output_dim=512,
        time_emb_dim=128,
        hidden_dims=[4096,2048,1024,512,256],
        use_context=True,
        context_dim=128,
        n_fourier= (7,8,1)
        
    ),
    #schedule=LearnableSchedule( hid_dim=[50,50], gate_func='silu',), # Fully learnable schedule with support for reduced variance
    schedule=LinearSchedule(),
    
    img_shape=512,
    sampling_step= 100,
    vocab_size=300,
    optimizer_conf= { "name"          : "AdamW", "lr" : 2e-4,  "weight_decay"  : 1e-2, "betas":(0.9, 0.99),},
    embedding_dim= 128,
    classes= 201, ##101, 201,11 -----------------------------------------------------------------change depending on the dataset
    folder_path=folder_path,
    use_recon = True,
    use_data_like=False,
    use_recon_original=False,
    p_uncond=0.1 #probabilty of each training instance to be treated as unconditional
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
logger = pl.loggers.CSVLogger(save_dir=f".{folder_path}")


trainer = pl.Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=2000,
    callbacks=[CustomProgressBar(),pl.callbacks.ModelCheckpoint(dirpath=f".{folder_path}",every_n_epochs=10,save_top_k=-1),lr_monitor],
    logger= logger
)
trainer.fit(vdm_model, datamodule=data_module)