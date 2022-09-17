import argparse
import contextlib
import einops
import numpy
import omegaconf
import os
import PIL
import pytorch_lightning
import sys
import torch
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ldm.util
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms

def load_model(config, model_file_path):
    model = ldm.util.instantiate_from_config(config.model)
    sd = torch.load(model_file_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to('mps')
    model.eval()
    return model

def text_to_image(options):
    os.makedirs(options.output, exist_ok=True)

    pytorch_lightning.seed_everything(options.seed)

    config = omegaconf.OmegaConf.load(options.config)
    model = load_model(config, model_file_path=options.model)
    device = torch.device('mps')
    model = model.to(device)

    sampler = ldm.models.diffusion.plms.PLMSSampler(model) if options.sampler == 'plms' else ldm.models.diffusion.ddim.DDIMSampler(model)
    shape = [
        options.channels,
        options.height // options.downsampling_factor,
        options.width // options.downsampling_factor
    ]
    start_code = torch.randn(
        [
            options.batch_size,
            options.channels,
            options.height // options.downsampling_factor,
            options.width // options.downsampling_factor
        ],
        device="cpu"
    ).to(torch.device(device)) if options.fixed_code else None

    data = [options.batch_size * [options.prompt]]
    count = 1
    with torch.no_grad():
        with contextlib.nullcontext(device.type):
            with model.ema_scope():
                for _ in tqdm.trange(options.iterations, desc="Sampling"):
                    for prompts in tqdm.tqdm(data, desc="data"):
                        conditioning = model.get_learned_conditioning(prompts)
                        unconditional_conditioning = None if options.guidance_scale == 1.0 else model.get_learned_conditioning(options.batch_size * [""])
                        ddim_samples, _ = sampler.sample(
                            batch_size=options.batch_size,
                            conditioning=conditioning,
                            eta=options.ddim_eta,
                            S=options.ddim_steps,
                            shape=shape,
                            unconditional_conditioning=unconditional_conditioning,
                            unconditional_guidance_scale=options.guidance_scale,
                            verbose=False,
                            x_T=start_code
                        )
                        ddim_samples = model.decode_first_stage(ddim_samples)
                        ddim_samples = torch.clamp((ddim_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        ddim_samples = ddim_samples.cpu().permute(0, 2, 3, 1).numpy()
                        samples = torch.from_numpy(ddim_samples).permute(0, 3, 1, 2)
                        for sample in samples:
                            sample = 255. * einops.rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                            image = PIL.Image.fromarray(sample.astype(numpy.uint8))
                            image.save(os.path.join(options.output, f"{count}.png"))
                            count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--ddim_eta", type=float, required=True)
    parser.add_argument("--ddim_steps", type=int, required=True)
    parser.add_argument("--downsampling_factor", type=int, required=True)
    parser.add_argument("--fixed_code", default=False, action='store_true')
    parser.add_argument("--guidance_scale", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--sampler", choices=["ddim", "plms"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    options = parser.parse_args()
    text_to_image(options)

if __name__ == "__main__":
    main()
