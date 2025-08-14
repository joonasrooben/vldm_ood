import matplotlib.pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import warnings

import torch.nn as nn
import torch.optim as optim
from .module.embedding import FourierEmbedding

from torch import Tensor
from torch import autograd
from torch import sqrt, sigmoid, prod
from torch import exp, expm1, log
from typing import Any, Tuple, Dict, List
from itertools import pairwise

from pytorch_lightning import LightningModule

from tqdm.auto import tqdm

from einops import reduce
from einops import rearrange

from .utils import exists
from .utils import default
from .utils import enlarge_as

from .unet import UNet
from .schedule import LinearSchedule
from .schedule import LearnableSchedule
import math
from scipy.stats import wasserstein_distance

loge2 = torch.log(torch.tensor(2))

def standard_cdf(x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def gaussian_log_prob(x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """
    Computes the element-wise log probability of x under a Gaussian with given mean and scale.
    Returns a tensor of log-probabilities summed over non-batch dimensions.
    """
    # Ensure scale is positive (scale = standard deviation)
    log_prob = -0.5 * (math.log(2 * math.pi) + 2 * torch.log(scale) + ((x - mean) ** 2) / (scale ** 2))
    # Sum over all dimensions except batch (assumes batch is dim 0)
    reduce_dims = list(range(1, log_prob.dim()))
    return log_prob.sum(dim=reduce_dims)



class VariationalDiffusion(LightningModule):
    '''
    Adapted from https://github.com/myscience/variational-diffusion/blob/main/src/vdm.py
    '''

    def __init__(
        self,
        backbone : nn.Module,
        schedule : nn.Module | None = None,  
        img_shape : Tuple[int, int] = (64, 64),
        vocab_size : int = 256,         
        data_key : str = 'imgs',
        ctrl_key : str | None = None,  
        sampling_step : int = 50,   
        optimizer_conf : Dict[str, Any] | None = None,
        classes : int = 100 + 1,
        embedding_dim : int = 64,
        folder_path  : str = "logs_jl/vdm_generic",
        use_recon : bool = False, 
        use_data_like : bool = False,
        use_recon_original : bool = False,
        p_uncond = 0.1,
    ) -> None:
        super().__init__()

        self.backbone : nn.Module = backbone
        self.schedule : nn.Module = default(schedule, LinearSchedule())
        self.folder_path = folder_path
        img_chn =  0 #self.backbone.inp_chn
        self.use_recon = use_recon
        self.use_recon_original = use_recon_original
        self.use_data_like = use_data_like
        self.img_shape = img_shape#(img_chn, img_shape)
        self.vocab_size = vocab_size
        self.p_uncond = p_uncond
        self.data_key = data_key
        self.ctrl_key = ctrl_key
        self.opt_conf : Dict[str, Any] = default(optimizer_conf, {'lr' : 1e-3})

        self.num_step = sampling_step
        self.classes = classes
        self.embedding_dim = embedding_dim
        self.automatic_optimization = True

        self.embedding_vectors = nn.Embedding(self.classes, self.embedding_dim,device=self.device)


    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def on_fit_start(self):
        # Freeze all parameters
        for name, param in self.named_parameters():
            param.requires_grad = True 
        
        self.schedule.q.requires_grad = False
        self.schedule.m.requires_grad = False

    def training_step(self, batch , batch_idx : int) -> Tensor:
        x_0,y = batch
        class_labels = y.clone()  
        # create random mask for unconditional instances
        uncond_mask = torch.rand(class_labels.shape, device=self.device) < self.p_uncond

        # replace selected labels with unconditional class idx
        class_labels[uncond_mask] = self.classes  - 1

        if self.automatic_optimization == False:
            opt = self.optimizers()  # Get the optimizer if using single or first optimizer
            opt.zero_grad()
            loss, stat = self.compute_loss(x_0, class_labels)
            self.manual_backward(loss, retain_graph=True)
            self.reduce_variance(*stat["var_args"])
            opt.step()
        else:
            loss, stat = self.compute_loss(x_0, class_labels)

        
        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)
        self.log('loss', loss, prog_bar = True)
        #self.log('recon_loss', stat["recon_loss"], prog_bar = True)
        self.log('dif_loss', stat["diffusion_loss"], prog_bar = True)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], on_step=True,prog_bar=False,logger=True)
        #self.log_dict({f'train_{k}' : v for k, v in stat}, logger = True, on_step = True, sync_dist = True)

        return loss
    
    @torch.enable_grad() 
    def validation_step(self, batch, batch_idx : int) -> Tensor:
        ctrl = None
        x_0 , y = batch
        # Compute the VLB loss
        loss, stat = self.compute_loss(x_0,y)
       
        self.log('val_loss'  ,loss, logger=True, on_step=True,prog_bar=True)

        if self.automatic_optimization == False:
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step(loss)
        #self.log_dict({f'val_{k}' : v for k, v in stat.items()}, logger=True, on_step=True, sync_dist=True)

        return x_0 , ctrl
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        val_loader = self.trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        x, y = val_batch  
        
        x = x.to(self.device).detach().cpu()
        y = y.to(self.device).detach().cpu()

        imgs = self(
                num_imgs=128,
                num_steps=self.num_step,
                # ctrl = ctrl,
                verbose = False,
                context =torch.LongTensor([self.classes-1]*128).to(self.device)
            ).detach().cpu()
        # measure the wasserstein distance 
        wd = torch.tensor([wasserstein_distance(imgs.T[i].reshape(-1).numpy(),x.T[i].reshape(-1).numpy()) for i in range(512)], device = self.device).mean()
        self.log('wd'  ,wd, logger=True, prog_bar=True)

        if self.current_epoch % 100 == 0:
            plt.matshow(torch.concatenate([imgs, x]))
            plt.title(f"the max {imgs.max()} and min {imgs.min()}")
            plt.savefig(f".{self.folder_path}step{self.global_step}.png")
            plt.clf()
            plt.close()
#comment out for linear schedule
            if self.automatic_optimization == False:
                scheduler =LinearSchedule() 
                time = torch.linspace(0.,1.,100).reshape(-1,1).to(device = self.device)#.repeat(num_imgs,1)

                gamma = scheduler(time).to(device = self.device)
                gamma2 = self.schedule(time).to(device = self.device)

                plt.plot(time.detach().cpu().numpy(),torch.sqrt(1-torch.sigmoid(gamma)).detach().cpu().numpy())
                plt.plot(time.detach().cpu().numpy(),torch.sqrt(1-torch.sigmoid(gamma2)).detach().cpu().numpy())
                plt.title(f"SNR vs time; epoch {self.global_step}")
                plt.savefig(f".{self.folder_path}schedule_at_{self.global_step}.png")
            
                plt.clf()
                plt.plot(time.detach().cpu().numpy(), -gamma.detach().cpu().numpy())
                plt.plot(time.detach().cpu().numpy(),-gamma2.detach().cpu().numpy())
                plt.title(f"log SNR vs time; epoch {self.global_step}")
                plt.savefig(f".{self.folder_path}gamma_schedule_at_{self.global_step}.png")
            
                plt.clf()
 
    def configure_optimizers(self):
        opt_name = self.opt_conf.pop('name')
        match opt_name:
            case 'AdamW': Optim = optim.AdamW
            case 'SGD'  : Optim = optim.SGD
            case _: raise ValueError(f'Unknown optimizer: {opt_name}')

        params = list(self.backbone.parameters()) +\
                 list(self.schedule.parameters())
        
        opt_kw = self.opt_conf

        opt = Optim(params, **opt_kw)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience = 100,factor = 0.9,verbose=True)
        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    @torch.no_grad()
    def forward(
        self,
        num_imgs : int,
        num_steps : int,
        seed_noise : Tensor | None = None,
        verbose : bool = False,
        plot : bool = False,
        context : Tensor | None = None,
    ) -> Tensor:
        
        device = self.device
        z_s = default(seed_noise, torch.randn((num_imgs, self.img_shape), device=device))#samples ranodm noise as x_t

        time = torch.linspace(1., 0., num_steps +1 , device=device).repeat((num_imgs,1))
        time = rearrange(time, 'a b ->  b a 1')
        gamma = self.schedule(time)
        self.embedding_vectors.to(self.device)
        context_emb = self.embedding_vectors(context)
        iterator = pairwise(gamma)
        iterator = tqdm(iterator, total=num_steps) if verbose else iterator
        for gamma_t, gamma_s in iterator:
            # Sample from the backward diffusion process
            z_s = self._coalesce(z_s, gamma_t, gamma_s, last_gamma=gamma[-1],cond=context_emb, clip_denoised=False)
            if plot == True:
                plt.matshow(z_s.detach().numpy())

        return z_s
    
    @torch.no_grad()
    def forward_loss(
        self,
        x_0,
        num_imgs : int,
        num_steps : int,
        verbose : bool = False,
        context : Tensor | None = None,
    ) -> Tensor:
        device = self.device
        # noise schedule values (gammas)
        time = torch.linspace(1., 0., num_steps +1 , device=self.device).repeat((num_imgs,1))
        time = rearrange(time, 'a b ->  b a 1')
        gamma = self.schedule(time)
        
        time_1 = torch.ones_like(x_0,device=self.device)
        gamma_1 = self.schedule(time_1)
        eps = torch.randn_like(x_0)
        z_s = self._diffuse(x_0 = x_0, gamma_t= gamma_1, noise=eps)

        #print(self.schedule.m)
        self.embedding_vectors.to(self.device)
        context_emb = self.embedding_vectors(context.to(self.device))
        #print(gamma.shape, gamma)
        iterator = pairwise(gamma)
        iterator = tqdm(iterator, total=num_steps) if verbose else iterator
        
        rec_losses = []

        for gamma_t, gamma_s in iterator:
            #print(gamma_s[0],gamma_t[0])
            # Sample from the backward diffusion process
            z_s, x_rec = self._coalesce(z_s, gamma_t, gamma_s, last_gamma=gamma[-1],cond=context_emb, return_z_s = True)
            rec_losses.append(reduce(((x_0- x_rec) ** 2), 'b ... -> b 1', 'sum').detach().cpu())    
        return z_s, rec_losses

    def compute_loss(self, imgs : Tensor, condition : Tensor) -> Tensor:
        bs, *img_shape = imgs.shape
        bpd = 1. / prod(torch.tensor(img_shape)) * loge2
        self.embedding_vectors.to(device = self.device)
        cond = self.embedding_vectors(condition)

        diffusion_loss, SNR_t, eps_theta,_, z_t= self._diffusion_loss(imgs, cond)

        gamma_0 : Tensor = self.schedule(torch.tensor([0.], device=self.device))
        gamma_1 : Tensor = self.schedule(torch.tensor([1.], device=self.device))
        latent_loss = self._latent_loss(imgs, gamma_1)
        #recon_loss = self._recon_loss(imgs,gamma_t,eps_theta,z_t,gamma_0,fourier=False) if self.use_recon else torch.ones(latent_loss.size()) 

        loss = (diffusion_loss+ latent_loss).mean()

        stat = {
            'tot_loss' : loss.item(),
            'var_args' : (SNR_t, diffusion_loss),
            'gamma_0'  : gamma_0.item(),
            'gamma_1'  : gamma_1.item(),
            #'recon_loss'     : recon_loss.mean().round(),
            'latent_loss'    : bpd * latent_loss.mean().round(),
            'diffusion_loss' : diffusion_loss.mean().round(),
        }

        return loss, stat
    
    def reduce_variance(
        self,
        SNR_t : Tensor,
        diff_loss : Tensor,
    ):
        msg = '''Noise schedule parameters have zero gradient. This is probably due
                to the function `reduce_variance` been called before `backward` has
                been called to the VLB loss. Reduce variance need the gradients and
                is thus now ineffective. Please only call `reduce_variance` after
                loss.backward() has been called. 
            '''

        for par in self.schedule.parameters():
            if torch.all(par.grad == 0): warnings.warn(msg)

            par.grad *= autograd.grad(
                outputs=SNR_t,
                inputs=par,
                grad_outputs=2 * diff_loss,
                create_graph=True,
                retain_graph=True,
            )[0]


    def compute_diffusion_loss_at_fixed_t(self, x_0, t_fixed, cond : Tensor):
        batch_size = x_0.size(0)
        cond = self.embedding_vectors(cond)
        # Use the fixed t for all instances
        times = torch.full((batch_size, 1), t_fixed, requires_grad=True,device=self.device)
        # Proceed with the standard loss computation using times
        gamma = self.schedule(times)
        # ... rest of your loss computation ...
        SNR_t = exp(-gamma)
        eps = torch.randn_like(x_0)
        z_t = self._diffuse(x_0, gamma, noise=eps)
        # Compute the latent noise eps_theta using the backbone model
        eps_theta = self.backbone(z_t, time=gamma, context = cond) 
        # Compute dgamma_dt
        #dgamma_dt, = torch.autograd.grad( gamma.sum(), times)
#            outputs=gamma,
#            inputs=times,
#            grad_outputs=torch.ones_like(gamma),
#            create_graph=True,
#            retain_graph=True,
#        )

        # Compute loss
        loss = 0.5  * reduce(((eps - eps_theta) ** 2), 'b ... -> b 1', 'sum') 
        return loss, SNR_t, eps_theta, gamma, z_t, eps

    def compute_loss_at_fixed_t(self, imgs : Tensor, t_fixed, condition : Tensor, recon = True, data = True, original = True, recon_return = False) -> Tensor:

        batch_size = imgs.size(0)
        gamma_1 : Tensor = self.schedule(torch.ones((batch_size, 1), device=self.device))
        gamma_0 : Tensor = self.schedule(torch.zeros((batch_size, 1), device=self.device))
        gamma_t = self.schedule(torch.full((batch_size, 1), t_fixed,device = self.device))

        diffusion_loss, _, eps_theta,_ , z_t, eps= self.compute_diffusion_loss_at_fixed_t(imgs,t_fixed ,condition.to(self.device))
        latent_loss = self._latent_loss(imgs, gamma_1)
        if recon_return:
            recon_loss, recon = self._recon_loss(imgs,gamma_t,eps_theta,z_t,gamma_0,fourier=True, recon_return=recon_return) if recon else torch.ones(latent_loss.size()) 
            return diffusion_loss.detach().cpu().numpy(), recon_loss.detach().cpu().numpy(), recon.detach().cpu().numpy(),  eps_theta.detach().cpu().numpy() ,eps.detach().cpu().numpy(), z_t.detach().cpu().numpy()
        else:
            recon_loss = self._recon_loss(imgs,gamma_t,eps_theta,z_t,gamma_0,fourier=True) if recon else torch.ones(latent_loss.size()) 

        return diffusion_loss.detach().cpu().numpy(), recon_loss.detach().cpu().numpy()

    def _diffusion_loss(self, x_0 : Tensor, cond : Tensor) -> Tensor:
        bs, *img_shape = x_0.shape
        times = self._get_times(bs).requires_grad_(True)
        gamma = self.schedule(times)
        SNR_t = exp(-gamma)
        eps = torch.randn_like(x_0)
        z_t = self._diffuse(x_0, gamma, noise=eps)
        eps_theta = self.backbone(z_t, time=gamma,context = cond) 
        #dgamma_dt, *_ = autograd.grad(
        #    outputs=gamma,
        #    inputs=times,
        #    grad_outputs=torch.ones_like(gamma),
        #    create_graph=True,
        #    retain_graph=True,   )
        #sigma_t =  torch.sqrt((sigmoid(+gamma)))
        difference = reduce(((eps - eps_theta) ** 2), 'b ... -> b 1', 'sum')
        loss = .5 * difference #* sigma_t
        return loss , SNR_t, eps_theta,gamma,z_t
    

    def get_score_fn(self, x, t, uncond = True, cond = None):
        '''Function to calculate the likelihoods with ODE'''
        batch_size = x.size(0)
        context = torch.tensor([self.classes-1]*batch_size, device=self.device) if uncond else cond
        condition = self.embedding_vectors(context)
        gamma = self.schedule(torch.full((batch_size, 1), t,device=self.device))
        # Compute the latent noise eps_theta using the backbone model
        eps_theta = self.backbone(x, time=gamma,context = condition)
        return eps_theta
    

    def _recon_loss(self, x_0 : Tensor, gamma_t : Tensor, eps_theta : Tensor,z_t : Tensor, gamma_0, fourier = False, recon_return = False) -> Tensor:
        gamma_s = gamma_0
        alpha_s_sq = sigmoid(-gamma_s)
        alpha_t_sq = sigmoid(-gamma_t)
        sigma_t = sqrt(sigmoid(+gamma_t))
        x_start = (z_t - sigma_t * eps_theta)/sqrt(alpha_t_sq)
        if fourier:
            n_fourier = (7,8,1)
            x_0_reshaped = x_0.reshape(-1,1,512)
            x_start_reshaped = x_start.reshape(-1,1,512)
            fourier_emb = FourierEmbedding(*n_fourier)
            x_0_ff = fourier_emb(x_0_reshaped).reshape(-1,512)
            x_start_ff = fourier_emb(x_start_reshaped).reshape(-1,512)
            loss = reduce(((x_0_ff - x_start_ff) ** 2), 'b ... -> b 1', 'sum').reshape(-1,3)# @ torch.tensor([1,0.01,0.01],device = self.device)
            if recon_return:
                return loss, x_start  
            else: 
                return loss
        loss  = reduce(((x_0 - x_start) ** 2), 'b ... -> b 1', 'sum') 
        return loss
    
            
    def _latent_loss(self, x_0 : Tensor, gamma_1 : Tensor) -> Tensor:
        sigma_1_sq = sigmoid(+gamma_1)
        
        alpha_1_sq = 1 - sigma_1_sq
        mu_sq = alpha_1_sq * x_0 ** 2
        loss = .5 * (sigma_1_sq + mu_sq - log(sigma_1_sq.clamp(min=1e-15)) - 1.)
        return reduce(loss, 'b ... -> b', 'sum')

    def _diffuse(self, x_0 : Tensor, gamma_t : Tensor, noise : Tensor | None = None) -> Tensor:
        noise = default(noise, torch.randn_like(x_0))
        alpha_t = enlarge_as(sqrt(sigmoid(-gamma_t)), x_0)
        sigma_t = enlarge_as(sqrt(sigmoid(+gamma_t)), x_0)

        return alpha_t * x_0 + sigma_t * noise

    
    def _coalesce(self, z_t : Tensor, gamma_t : Tensor, gamma_s : Tensor, last_gamma :Tensor, cond : Tensor, clip_denoised=True, return_z_s=False, return_stats = False) -> Tensor:
        alpha_s_sq = sigmoid(-gamma_s)
        alpha_t_sq = sigmoid(-gamma_t)

        sigma_t = sqrt(sigmoid(+gamma_t))
        
        c = -expm1(gamma_s - gamma_t)
        eps = self.backbone(z_t, gamma_t,context = cond) # NOTE: We should add here conditioning if needed
      
        scale = sqrt((1 - alpha_s_sq) * c) 
        if clip_denoised:
            eps.clamp_((z_t - sqrt(alpha_t_sq)) / sigma_t, (z_t + sqrt(alpha_t_sq)) / sigma_t)
        x_start = (z_t - sigma_t * eps) / sqrt(alpha_t_sq)
        mean = sqrt(alpha_s_sq) * (z_t * (1 - c) / sqrt(alpha_t_sq) + c * x_start)

        new_eps = torch.randn_like(z_t)
        next_sample = mean + scale * new_eps
        if last_gamma[0] == gamma_s[0]:
            if return_stats:
                return next_sample, x_start, eps, torch.zeros_like(z_t), mean, scale
            if return_z_s:
                return next_sample, x_start, eps, new_eps
            return x_start
        
        if return_stats:
            return next_sample, x_start, eps, new_eps, mean, scale
        if return_z_s:
            return next_sample, x_start, eps, new_eps
        return next_sample
    
    
    def evaluate_likelihood(self, x0: Tensor, cond: Tensor, num_steps = 50, clip = True, dgamma = True, t_lim=0.) -> Tensor:
        """
        Evaluate a lower bound on the log-likelihood for a given image x0.
        cond: conditioning tensor for the backbone model.
        
        This function:
          1. Computes the forward diffusion trajectory of x0 using the closed-form q(z_t|x0).
          2. For each reverse step from t to t-1, uses _coalesce (with return_stats=True)
             to get the Gaussian parameters (mean, scale) of the reverse transition.
          3. Computes the log probability of the true forward latent (obtained via reparameterization)
             under the reverse model.
          4. Adds the log probability of the prior at t=T.
        
        Returns a tensor of log likelihood lower bounds (one per batch element).
        """
        batch_size = x0.shape[0]
        times= torch.linspace(t_lim, 1., num_steps +1 , requires_grad=True,device=self.device).repeat((batch_size,1))
        times = rearrange(times, 'a b ->  b a 1') #?

        gamma = self.schedule(times)
        T = gamma.shape[0] - 1  
        log_probs = []
        L_diffs = []
        rec_diffs = []

        eps = torch.randn_like(x0)
        z_t = self._diffuse(x0,gamma[T],eps)
        log_prob = gaussian_log_prob(z_t, torch.zeros_like(z_t), torch.ones_like(z_t))
        log_probs.append(log_prob.detach().cpu().numpy())

        self.embedding_vectors.to(self.device)
        context_emb = self.embedding_vectors(cond)       
        
        with torch.no_grad():
            for t in range(T, 0, -1):
                gamma_t = gamma[t]
                gamma_prev = gamma[t - 1]
                last_gamma = gamma[-1]  
        
                _, x_rec, eps_theta, _ , mean, scale = self._coalesce(
                    z_t, gamma_t, gamma_prev, last_gamma, context_emb,
                    clip_denoised=clip, return_z_s=False, return_stats=True
                )
                
                L_diffs.append(reduce(((eps - eps_theta) ** 2), 'b ... -> b 1', 'sum').detach().cpu().numpy()) #shape (T,bs)
                rec_diffs.append(reduce(((x_rec - x0)**2),'b ... -> b 1', 'sum').detach().cpu().numpy()) #shape (T,bs)

                eps = torch.randn_like(x0)
                z_true = self._diffuse(x0,gamma[t-1],eps)
            
                log_prob_t = gaussian_log_prob(z_true, mean, scale).detach().cpu().numpy()
                log_probs.append(log_prob_t) #shape (T + 1, bs)
                z_t = z_true
        if dgamma:
            dgamma_dt, *_ = autograd.grad(
                outputs=gamma,
                inputs=times,
                grad_outputs=torch.ones_like(gamma),
                create_graph=True,
                retain_graph=True,
            )
        # Detach gamma so that further operations don't track gradients
            dgamma_dt = dgamma_dt.detach().cpu().numpy()
        else:
            dgamma_dt = 0
            
        gamma = gamma.detach()

        # log_prob now holds the ELBO (a lower bound on the log likelihood) per data point.
        return log_probs, dgamma_dt,L_diffs ,rec_diffs   


    def _get_times(self, batch_size : int, sampler : str = 'low-var') -> Tensor:

        samplers = ('low-var', 'naive')

        match sampler:
            case 'low-var':
                t_0 = torch.rand(1).item() / batch_size
                ts = torch.arange(t_0, 1., 1 / batch_size, device=self.device)

                # Add single channel dimension
                return rearrange(ts, 'b -> b 1')
            
            case 'naive':
                return torch.rand((batch_size, 1), device=self.device)
            
        raise ValueError(f'Unknown sampler: {sampler}. Available samplers are: {samplers}')





