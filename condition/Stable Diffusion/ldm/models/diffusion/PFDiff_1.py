"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, sampled_timestep=None):
        if sampled_timestep is None:
            self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                    num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        else:
            self.ddim_timesteps = sampled_timestep
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               sampled_timestep=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                    
        if sampled_timestep is not None:
            sampled_timestep = sorted(sampled_timestep)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, sampled_timestep=sampled_timestep)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    ea_timesteps=sampled_timestep,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, ea_timesteps=None):
        
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if ea_timesteps is None:
            if timesteps is None:
                timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            elif timesteps is not None and not ddim_use_original_steps:
                subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
                timesteps = self.ddim_timesteps[:subset_end]
        else:
            timesteps = ea_timesteps

        timesteps = np.asarray(timesteps)
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        seq = list(time_range)
        seq_next = list(time_range[1:]) + [-1]
        seq_next_next = list(time_range[2:]) + [-1] + [-2]

        xs = [img]

        et_list = []
        for index, (i_t, j_t, k_t) in enumerate(zip(seq, seq_next, seq_next_next)):
            if index != 0 and (index-1) % 2 !=0:
                continue
            i = total_steps - index - 1
            j = total_steps - index - 2
            k = total_steps - index - 3

            if len(et_list) == 0:
                t = torch.full((b,), i_t, device=device, dtype=torch.long)
                xt = xs[-1]

                et = self.my_model(xt, cond, t,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning)
                
                xt_next = self.phi_any(et, xt, i, j)
                
                et_list.append(et)
                xs.append(xt_next)
            else:
                t = torch.full((b,), i_t, device=device, dtype=torch.long)
                next_t = torch.full((b,), j_t, device=device, dtype=torch.long)
                xt = xs[-1]
                if j_t == -1:
                    et = self.my_model(xt, cond, t,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning)
                    xt_next = self.phi_any(et, xt, i, j)
                    xs.append(xt_next)
                    break

                et_= et_list[-1]
                x_next = self.phi_any(et_, xt, i, j)

                et_next = self.my_model(x_next, cond, next_t,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning)
                        
                xt_next_next = self.phi_any(et_next, xt, i, k)
                et_list.append(et_next)

                xs.append(xt_next_next)

                if k_t == -1:
                    break

        return xs[-1], ""


    @torch.no_grad()
    def phi_any(self, theta, xt, t, t_next):
        b, *_, device = *xt.shape, xt.device
        at = torch.full((b, 1, 1, 1), self.ddim_alphas[t], device=device)
        if t_next == -1:
            at_next = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[0], device=device)
        else:
            at_next = torch.full((b, 1, 1, 1), self.ddim_alphas[t_next], device=device)

        x0_t = (xt - theta * (1 - at).sqrt()) / at.sqrt()

        c2 = (1 - at_next).sqrt()
        xt_next_mu = at_next.sqrt() * x0_t  + c2 * theta

        return xt_next_mu

    @torch.no_grad()
    def my_model(self, x, c ,t, 
                score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None):
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        
        return e_t


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec