import torch

import torch
from diffusers.utils.torch_utils import randn_tensor

def prepare_latents(
    pipe,
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
):
    """
    Prepares latents for the diffusion process.

    Args:
        pipe: The pipeline object.
        batch_size: The batch size.
        num_channels_latents: The number of channels in the latents.
        height: The height of the image.
        width: The width of the image.
        dtype: The data type.
        device: The device.
        generator: The random number generator.
        latents: Optional pre-existing latents.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The prepared latents and latent image IDs.
    """
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipe.vae_scale_factor * 2))

    shape = (batch_size, num_channels_latents, height, width)

    if latents is not None:
        latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)  # Corrected
        return latents.to(device=device, dtype=dtype), latent_image_ids

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)

    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype) # Corrected

    return latents, latent_image_ids


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

@staticmethod
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

@staticmethod
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents