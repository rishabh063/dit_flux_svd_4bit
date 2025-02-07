import torch
from typing import Optional
from local_utils import retrieve_latents
from latent_util import _pack_latents
def prepare_mask_latents(
    pipe,
    mask,
    masked_image,
    batch_size,
    num_channels_latents,
    num_images_per_prompt,
    height,
    width,
    dtype,
    device,
    generator,
):

    # 1. calculate the height and width of the latents
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipe.vae_scale_factor * 2))

    # 2. encode the masked image
    if masked_image.shape[1] == num_channels_latents:
        masked_image_latents = masked_image
    else:
        masked_image_latents = retrieve_latents(pipe.vae.encode(masked_image), generator=generator)

    masked_image_latents = (masked_image_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

    # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    batch_size = batch_size * num_images_per_prompt
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                " of masks that you pass is divisible by the total requested batch size."
            )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        if not batch_size % masked_image_latents.shape[0] == 0:
            raise ValueError(
                "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                " Make sure the number of images that you pass is divisible by the total requested batch size."
            )
        masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

    # 4. pack the masked_image_latents
    # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
    masked_image_latents = _pack_latents(
        masked_image_latents,
        batch_size,
        num_channels_latents,
        height,
        width,
    )

    # 5.resize mask to latents shape we we concatenate the mask to the latents
    mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
    mask = mask.view(
        batch_size, height, pipe.vae_scale_factor, width, pipe.vae_scale_factor
    )  # batch_size, height, 8, width, 8
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(
        batch_size, pipe.vae_scale_factor * pipe.vae_scale_factor, height, width
    )  # batch_size, 8*8, height, width

    # 6. pack the mask:
    # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
    mask = _pack_latents(
        mask,
        batch_size,
        pipe.vae_scale_factor * pipe.vae_scale_factor,
        height,
        width,
    )
    mask = mask.to(device=device, dtype=dtype)

    return mask, masked_image_latents