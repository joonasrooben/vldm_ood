import torch
import sys
sys.path.append('..')
from .src.unetfc2 import UNetFC2
import os
from .src.vdm import VariationalDiffusion
from .src.schedule import LearnableSchedule, LinearSchedule, FixedLinearSchedule
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, Dataset, random_split
from openood.OpenOOD.openood.networks import ResNet18_32x32, ResNet18_224x224
import numpy as np
from openood.OpenOOD.openood.evaluators.metrics import compute_all_metrics
from tqdm import tqdm
import math
import torch.nn as nn
from torchdiffeq import odeint
import yaml


class ODEfunc(nn.Module):
    #inspired by https://github.com/yang-song/score_sde
    def __init__(self, model, beta_min, beta_max, labels, uncond,N=1):
        super().__init__()
        self.model = model
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.labels = labels
        self.uncond = uncond
        self.N = N

    def schedules(self,t):
        """
        t :  shape [batch] or scalar, requires_grad=True
        returns sigma, alpha, beta  (each same shape as t)
        """
        gamma  = self.model.schedule(t.reshape(-1,1))                      # γ_η(t)
        sigma2 = torch.sigmoid(gamma)               # σ²(t)
        sigma  = torch.sqrt(sigma2 )
        alpha2 = 1. - sigma2
        alpha  = torch.sqrt(alpha2 )

        # derivative wrt t  (needs create_graph=True so β is differentiable)
        #dgamma_dt  = (self.beta_max - self.beta_min) #if linear schedule
        dgamma_dt = torch.autograd.grad(gamma.sum(), t, create_graph=True)[0]
        beta = sigma2 * dgamma_dt                 # guaranteed ≥0
        
        return sigma, alpha, beta
    
    def reverse_dynamics(self, t, x):
        eps_theta = self.model.get_score_fn(x, t.item(), self.uncond,torch.tensor(self.labels,device=device))
        t = t.detach().requires_grad_(True)
        sigma_t, alpha, beta_t = self.schedules(t)
        score = -eps_theta/(sigma_t)
        drift =- 0.5* beta_t *( x + score )#
        return drift, beta_t

    def forward(self, t, states):
        x, logp = states
        batch = x.size(0)
        drift_rev, beta_t = self.reverse_dynamics(t, x)
        # Hutchinson trace estimator
        divergence = 0
        for i in range(self.N):
            z = torch.randn_like(x)
            inner = (drift_rev * z).view(batch, -1).sum(1)
            grads = torch.autograd.grad(outputs = inner.sum(), inputs = x, create_graph=True)[0]
            divergence += (grads*z).sum(dim=[*range(1,x.dim())]) / self.N
        return drift_rev, -divergence

def compute_log_prob(
    model, x0, labels, beta_min=-13.3, beta_max=5.0, T=1.0,
    n_steps=50, dtype=torch.float32, method='rk4', uncond = False,N=1
):
    """
    Compute log p(x0) by fixed-step integration (RK4) over n_steps.
    """
    device = next(model.parameters()).device
    x0 = x0.to(device=device, dtype=dtype).requires_grad_(True)
    batch = x0.shape[0]

    # Dummy initial logp at t=0
    logp0 = torch.zeros(batch, device=device, dtype=dtype)
    if method == 'rk4':
        ts = torch.linspace(0.,T, steps=n_steps+1, device=device, dtype=dtype)
    else:
        ts= torch.tensor([0., T], device=device)
    
    odefunc = ODEfunc(model, beta_min, beta_max, labels,uncond,N)
    x_ts, logp_ts= odeint(odefunc, (x0, logp0), ts, rtol=1e-3, atol=1e-3 ,method=method, options = {"dtype": torch.float32,})

    # Take final values at t=T
    xT         = x_ts[-1]
    logp_delta = logp_ts[-1]
    # Compute true log p_T(x_T) under N(0,I)
    D = xT.view(batch, -1).size(1)
    logp_T = -0.5 * (xT.view(batch, -1).pow(2).sum(1) + D * math.log(2*math.pi))
    # Recover log p_0(x0) = log p_T(x_T) - ∫ div
    logp0_exact = logp_T + logp_delta

    bpd = -(logp0_exact) / np.log(2)
    N = 512
    bpd = bpd / N
    # A hack to convert log-likelihoods to bits/dim
    return logp0_exact.detach().cpu().numpy(), logp_T.detach().cpu().numpy(), bpd.detach().cpu().numpy()


def denormalize(normalized, basis):
    return basis.min() + (basis.max() - basis.min())*((normalized + 1)/2)

def normalize(unnormalized, basis):
    return (2 * (unnormalized - basis.min()) / (basis.max() - basis.min()) - 1)

def normalize_per_feat(unnormalized, basis):
    return (2 * (unnormalized - basis.min(0)) / (basis.max(0) - basis.min(0)) - 1)

def denormalize_per_feat(normalized, basis):
    return basis.min(0) + (basis.max(0) - basis.min(0))*((normalized + 1)/2)

def denormalize_std(normalized, basis):
    return normalized * basis.std(0) + basis.mean(0)

def normalize_std(unnormalized, basis):
    return (unnormalized - basis.mean(0))/basis.std(0)


def calculate_density(ind_name, tag, gen_softs = None,
                    gen_acts = None, train_acts = None, clip = True,
                    norm_type="feat",
                    batch_s = 1000,nr_div_evals = 1,method="dopri5"):
    print(tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tag != "gen":

        acts = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}.npy")
        softs = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}_softs.npy")
        if (tag == "train" and ind_name == "imagenet200"):
            indices = np.random.randint(0,len(acts),20_000)
            acts = acts[indices]
            softs = softs[indices]
        
        if norm_type == "feat":
            normed_acts = normalize_per_feat(acts,train_acts)
        elif norm_type == "std":
            normed_acts = normalize_std(acts, train_acts)
        else:
            normed_acts = normalize(acts,train_acts)
        cl = torch.nn.functional.softmax(torch.tensor(softs).to(device), dim=1).argmax(1)
    else:
        normed_acts = gen_acts
        softs = gen_softs
        cl = softs.argmax(1).to(device)
        softs = softs.detach().cpu().numpy()

    tk = len(cl)
    iters = tk//batch_s 

    if tk % batch_s != 0:
        iters += 1
    densities = np.zeros((3,tk))
    uncond_densities = np.zeros((3,tk))
    for j in tqdm(range(iters),desc="iters"):
        if j == iters - 1:
            normed_acts_batch = normed_acts[j*batch_s:]
            cl_cond =  torch.tensor(softs.argmax(1)[j*batch_s:]).to(device)
            cl_uncond = torch.tensor([num_classes]*tk).to(device)[j*batch_s:]

        else:
            interval = range(j*batch_s,(j+1)*batch_s)
            normed_acts_batch = normed_acts[interval]
            cl_cond =  torch.tensor(softs.argmax(1)[interval]).to(device)    
            cl_uncond = torch.tensor([num_classes]*batch_s).to(device)
        
        res = compute_log_prob(model,torch.tensor(normed_acts_batch).to(device),labels =cl_cond ,N=nr_div_evals,method=method)
        if j == iters - 1:
             densities[:,j*batch_s:] += res           
        else:
            densities[:,interval] += res
        res = compute_log_prob(model,torch.tensor(normed_acts_batch).to(device),labels =cl_uncond,N=nr_div_evals,method=method)
        if j == iters - 1:
            uncond_densities[:,j*batch_s:] += res           
        else:
            uncond_densities[:,interval] += res

    np.save(f'{output_folder}/densities_{tag}.npy',densities)
    np.save(f'{output_folder}/uncond_densities_{tag}.npy',uncond_densities)

def calculate_loss(ind_name, tag, repetitions, num_step, gen_softs = None,
                    gen_acts = None, train_acts = None, clip = True,
                      uncond =False, norm_type="feat",
                      top_k = 1, t_lim = 0.):
    print(tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tag != "gen":
        acts = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}.npy")
        softs = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}_softs.npy")
        if norm_type == "feat":
            normed_acts = normalize_per_feat(acts,train_acts)
        elif norm_type == "std":
            normed_acts = normalize_std(acts, train_acts)
        else:
            normed_acts = normalize(acts,train_acts)
        cl = torch.nn.functional.softmax(torch.tensor(softs).to(device), dim=1).argmax(1)
    else:
        normed_acts = gen_acts
        softs = gen_softs
        cl = softs.argmax(1).to(device)
        softs = softs.detach().cpu().numpy()

    tk = len(cl)
    if uncond:
        cl = torch.tensor([num_classes]*tk).to(device)
    if top_k > 1:
        class_vectors = np.argsort(softs, axis=1)[:,-top_k:].T[::-1]

    for j in tqdm(range(top_k),desc="Top k"):
        dgamma_bool = True
        log_probs = np.zeros((num_step + 1,tk))
        L_diffs = np.zeros((num_step,tk))
        rec_diffs = np.zeros((num_step,tk))
        cl = torch.LongTensor(class_vectors[j]).to(device) if top_k > 1 else torch.tensor(softs.argmax(1)).to(device)
        for i in tqdm(range(repetitions), desc="Evaluating likelihood"):
            res = model.evaluate_likelihood(torch.tensor(normed_acts).to(device),cl, num_step, clip, dgamma_bool,t_lim)
            log_probs += np.array(res[0])/repetitions
            if i >= 1: #change this if want dgamma
                dgamma_bool = False
            else:
                dgammas = np.array(res[1]).reshape((num_step + 1,tk))
                np.save(f'{output_folder}/dgammas_{tag}_steps_{num_step}_reps_{repetitions}_{j}.npy',dgammas)

            L_diffs += np.array(res[2]).reshape((num_step ,tk))/repetitions
            rec_diffs += np.array(res[3]).reshape((num_step ,tk))/repetitions

        np.save(f'{output_folder}/dif_loss_{tag}_steps_{num_step}_reps_{repetitions}_{j}.npy',L_diffs)
        np.save(f'{output_folder}/log_probs_{tag}_steps_{num_step}_reps_{repetitions}_{j}.npy',log_probs)
        np.save(f'{output_folder}/rec_loss_{tag}_steps_{num_step}_reps_{repetitions}_{j}.npy',rec_diffs)


def calculate_recs(ind_name, tag, repetitions, t, gen_softs = None, gen_acts = None, train_acts = None):
    print(tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tag != "gen":
        acts = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}.npy")
        softs = np.load(f"{acts_input}/resnet18_{ind_name}_{tag}_softs.npy")
        normed_acts = normalize(acts,train_acts)
        cl = torch.nn.functional.softmax(torch.tensor(softs).to(device), dim=1).argmax(1)
    else:
        normed_acts = gen_acts
        softs = gen_softs
        cl = softs.argmax(1).to(device)
    
    for rep in range(repetitions):
        loss = model.compute_loss_at_fixed_t(torch.Tensor(normed_acts).to(device), t,cl, recon_return = True)
        recon = denormalize(loss[2],train_acts)
#        losses += np.concatenate([loss[0],loss[1], cos_eps, cos_rec],1).T/repetitions
    cl = cl.detach().cpu().numpy()
    unique_classes = np.unique(cl)
    normed_acts = denormalize(normed_acts,train_acts)

    for c in unique_classes:
        # Find indices for the current class c
        indices = np.where(cl == c)[0]

        # Determine the base directory based on the tag
        if tag == ind_name:
            base_path = os.path.join(output_folder, f'IND-{ind_name}')
            recon_dir = os.path.join(base_path, 'reconstruction', 'prelogit', str(c))
            orig_dir  = os.path.join(base_path, 'original', 'prelogit', str(c))
        else:
            base_path = os.path.join(output_folder, f'OOD-{tag}')
            recon_dir = os.path.join(base_path, 'reconstruction', 'prelogit', f'cl_{c}')
            orig_dir  = os.path.join(base_path, 'original', 'prelogit', str(c))
        
        # Create directories if they do not exist
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(orig_dir, exist_ok=True)
        
        # Aggregate all the recon and normed_acts corresponding to class c
        # Assumes that recon and normed_acts are NumPy arrays. If denormalize is vectorized, you could pass normed_acts[indices] directly.
        recon_data = recon[indices]
        normed_data = normed_acts[indices]
        
        # Save the aggregated arrays to file
        np.save(os.path.join(recon_dir, f'{c}.npy'), recon_data)
        np.save(os.path.join(orig_dir, f'{c}.npy'), normed_data)



# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set parameters from config
ldm_version = config["ldm_version"]
special_index = config["special_index"]
ind_name = config["ind_name"]
folder_name = config["folder_name"].format(ldm_version=ldm_version)

output_folder = config["output_folder"].format(ind_name=ind_name, special_index=special_index)
if not os.path.exists(output_folder):
        # Create the folder (including any necessary parent directories)
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
else:
    print(f"Folder already exists: {output_folder}")

acts_input = config["acts_input"].format(ind_name=ind_name)
epoch = config["epoch"]
step = config["step_factor"] * (epoch + 1)
num_imgs = config["num_imgs"]
num_step = config["num_step"]
num_step_time = config["num_step_time"]
num_classes = config["num_classes"]
top_k = config["top_k"]
clip = config["clip"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shall_generate = config["shall_generate"]
shall_calculate = config["shall_calculate"]
shall_calculate_recs = config["shall_calculate_recs"]
shall_calculate_uncond = config["shall_calculate_uncond"]
shall_calculate_densities = config["shall_calculate_densities"]
norm_type = config["norm_type"]
method = config["density_method"]

btb_on = config["btb_on"]
ff = config["ff"]
cos = config["cos"]
ind_tags = config["ind_tags"]
ood_tags = config["ood_tags"]
repetitions = config["repetitions"]
t_values = np.linspace(*config["t_values"])[9:]
t_lim = config["t_lim"]

nr_div_evals = config["nr_div_evals"]

model = VariationalDiffusion.load_from_checkpoint(
    config["model_ckpt"].format(folder_name=folder_name, epoch=epoch, step=step),
    backbone=UNetFC2(**config["backbone"]),
    #schedule=LearnableSchedule( hid_dim=[50,50], gate_func='silu',),#
    schedule = LinearSchedule(),
    #schedule = FixedLinearSchedule(),
    img_shape=config["img_shape"],
    vocab_size=config["vocab_size"],
    optimizer_conf=config["optimizer_conf"],
    embedding_dim=config["embedding_dim"],
    classes=config["classes"]
)
model.eval()

with open(os.path.join(output_folder, 'used_config.yaml'), 'w') as f:
    yaml.dump(config, f)

#generating from diffusion model
gen_train_data = []

if shall_generate:
    for cl in range(num_classes):
        print(str(cl))
        print("generating")
        num_imgs = num_imgs
        num_step = num_step
        context = torch.LongTensor([cl]*num_imgs).to(device)
        samples = model(num_imgs, num_step, verbose=True, context = context).detach().cpu().numpy()
        gen_train_data.append(samples)
    print("---generation finished")
    gen_data_normed = np.array(gen_train_data).reshape((-1,512))
    np.save(f'../logs_jl/vdm_open/acts/resnet18_{ind_name}_gen_{num_imgs*num_classes}k_{epoch}_vdm{ldm_version}.npy',gen_data_normed)


#net = ResNet18_32x32(num_classes=num_classes)
net = ResNet18_224x224(num_classes=num_classes)

net.load_state_dict(
    #torch.load( '../openood/OpenOOD/results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.7810.ckpt', map_location=device)
    #torch.load( '../openood/OpenOOD/results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s1/best_epoch100_acc0.7710.ckpt', map_location=device)
    #torch.load( '../openood/OpenOOD/results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s2/best_epoch90_acc0.7760.ckpt', map_location=device)

    #torch.load( '../openood/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch96_acc0.9470.ckpt', map_location=device)
    #torch.load( '../openood/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best_epoch95_acc0.9500.ckpt', map_location=device)
    #torch.load( '../openood/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best_epoch99_acc0.9450.ckpt', map_location=device)


    #torch.load('../openood/OpenOOD/results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt', map_location=device)
    torch.load('../openood/OpenOOD/results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/best_epoch89_acc0.8560.ckpt', map_location=device)
    #torch.load('../openood/OpenOOD/results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best_epoch88_acc0.8480.ckpt', map_location=device)


)
print(torch.cuda.is_available())
net.cuda()
net.eval()

gen_data_normed = np.load(f'../logs_jl/vdm_open/acts/resnet18_{ind_name}_gen_{num_imgs*num_classes}k_{epoch}_vdm{ldm_version}.npy')
gen_targets = np.array([np.ones(num_imgs)*cl for cl in range(num_classes)]).flatten()
train_acts = np.load(f"{acts_input}/resnet18_{ind_name}_train.npy")

if norm_type == "feat":
    gen_data = denormalize_per_feat(gen_data_normed,train_acts)
elif norm_type == "std":
    gen_data = denormalize_std(gen_data_normed,train_acts)
else:
    gen_data = denormalize(gen_data_normed,train_acts)

gen_softs = torch.nn.functional.softmax(net.fc(torch.Tensor(gen_data).to(device)), dim=1)

if shall_calculate_recs:
    print("starting to calculate....")
    ind_tags.extend(ood_tags)
    for tag in ind_tags:
        calculate_recs(ind_name,tag,1,0.5,gen_softs,gen_data_normed,train_acts)
    print("...calculations done.")

if shall_calculate_densities:
    print("starting to calculate densities....")
    ind_tags.extend(ood_tags)
    for tag in ind_tags:
        calculate_density(ind_name,tag,gen_softs,gen_data_normed,train_acts,clip,norm_type=norm_type,batch_s=100,nr_div_evals=nr_div_evals,method=method)
    print("...calculations done.")


if shall_calculate:
    print("starting to calculate....")
    ind_tags.extend(ood_tags)
    for tag in ind_tags:
        calculate_loss(ind_name, tag, repetitions, num_step_time, gen_softs, gen_data_normed, train_acts, clip = clip, uncond=shall_calculate_uncond,norm_type=norm_type,top_k=top_k,t_lim=t_lim)
    print("...calculations done.")

