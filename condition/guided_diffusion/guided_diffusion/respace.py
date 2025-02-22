import numpy as np
import torch as th


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            if desired_count == 37:
                s = set(range(0, num_timesteps, 27))
                s.remove(999)
            elif desired_count == 43:
                s = set(range(0, num_timesteps, 23))
                s.remove(989)
            elif desired_count == 57:
                s = set(range(0, num_timesteps, 17))
                s.remove(986)
                s.remove(969)
            elif desired_count == 58:
                s = set(range(0, num_timesteps, 17))
                s.remove(986)
            elif desired_count == 45:
                s = set(range(0, num_timesteps, 22))
                s.remove(990)
            return s
            # seq = np.linspace(0, num_timesteps - 2, desired_count)
            # return set([int(s) for s in list(seq)])
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [10, 20]
    size_per = num_timesteps // len(section_counts) # 500  1000
    extra = num_timesteps % len(section_counts)  # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 500
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (500 - 1) / (10 - 1) = 55.4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def createSpacedDiffusion(
    algorithm,
    use_timesteps,
    betas,
    model_mean_type,
    model_var_type,
    loss_type,
    rescale_timesteps,
):
    if algorithm == "origin":
        from guided_diffusion.gaussian_diffusion import GaussianDiffusion
    elif algorithm == "PFDiff_1":
        from guided_diffusion.PFDiff_1 import GaussianDiffusion
    elif algorithm == "PFDiff_2_1":
        from guided_diffusion.PFDiff_2_1 import GaussianDiffusion
    elif algorithm == "PFDiff_2_2":
        from guided_diffusion.PFDiff_2_2 import GaussianDiffusion
    elif algorithm == "PFDiff_3_1":
        from guided_diffusion.PFDiff_3_1 import GaussianDiffusion
    elif algorithm == "PFDiff_3_2":
        from guided_diffusion.PFDiff_3_2 import GaussianDiffusion
    elif algorithm == "PFDiff_3_3":
        from guided_diffusion.PFDiff_3_3 import GaussianDiffusion
    else:
        raise NotImplementedError

    class SpacedDiffusion(GaussianDiffusion):
        """
        A diffusion process which can skip steps in a base diffusion process.

        :param use_timesteps: a collection (sequence or set) of timesteps from the
                            original diffusion process to retain.
        :param kwargs: the kwargs to create the base diffusion process.
        """
        def __init__(self, use_timesteps, **kwargs):
            self.use_timesteps = set(use_timesteps)
            self.timestep_map = []
            self.original_num_steps = len(kwargs["betas"])

            base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod) # 通过长序列的 \overline{alpha} 解 短序列的 \beta，保证两条序列对应位置上的 alpha_cumprod 相同
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            kwargs["betas"] = np.array(new_betas)
            super().__init__(**kwargs)  # 根据短序列的 beta 求解 新的 alpha、overline_alpha 和 各种概率函数

        def p_mean_variance(
            self, model, *args, **kwargs
        ):  # pylint: disable=signature-differs
            return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

        def training_losses(
            self, model, *args, **kwargs
        ):  # pylint: disable=signature-differs
            return super().training_losses(self._wrap_model(model), *args, **kwargs)

        def condition_mean(self, cond_fn, *args, **kwargs):
            return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

        def condition_score(self, cond_fn, *args, **kwargs):
            return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

        def _wrap_model(self, model):
            if isinstance(model, _WrappedModel):
                return model
            return _WrappedModel(
                model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
            )

        def _scale_timesteps(self, t):
            # Scaling is done by the wrapped model.
            return t

            
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
