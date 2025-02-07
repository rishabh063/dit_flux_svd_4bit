import torch
from typing import Union, List, Optional
import logging
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)    
def get_t5_prompt_embeds(
    pipe,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Generates T5 prompt embeddings.

    Args:
        pipe: The pipeline object (acts like 'self' from the original class method).
        prompt: The text prompt(s).
        num_images_per_prompt: Number of images to generate per prompt.
        max_sequence_length: Maximum sequence length for T5.
        device: The device (CPU or GPU) to use.
        dtype: The data type to use.

    Returns:
        torch.Tensor: The prompt embeddings.
    """
    device = device or pipe._execution_device
    dtype = dtype or pipe.text_encoder_2.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = pipe.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = pipe.tokenizer_2.batch_decode(untruncated_ids[:, pipe.tokenizer_2.model_max_length - 1 : -1])  #Corrected line
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = pipe.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = pipe.text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def get_clip_prompt_embeds(
    pipe,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device or pipe._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)


    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = pipe.tokenizer.batch_decode(untruncated_ids[:, pipe.tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {pipe.tokenizer_max_length} tokens: {removed_text}"
        )
    prompt_embeds = pipe.text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output  # Corrected line: Access the pooled output correctly
    prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    pipe,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
):
    device = device or pipe._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(pipe, FluxLoraLoaderMixin):
        pipe._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if pipe.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(pipe.text_encoder, lora_scale)
        if pipe.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(pipe.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = get_clip_prompt_embeds(
            pipe,
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = get_t5_prompt_embeds(
            pipe,
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

    if pipe.text_encoder is not None:
        if isinstance(pipe, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(pipe.text_encoder, lora_scale)

    if pipe.text_encoder_2 is not None:
        if isinstance(pipe, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(pipe.text_encoder_2, lora_scale)

    dtype = pipe.transformer.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids