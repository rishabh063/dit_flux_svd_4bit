# SVDQuant Integration for Hugging Face Diffusers



UnOfficial experimental implementation of **SVDQuant** for 4-bit weight & activation quantization in diffusion models. Achieves 3.5× memory reduction and 8.7× latency improvement on 16GB GPUs while maintaining visual fidelity.

uses single value decomposition to counter outlier thus giving higher accuracy than naive int4


![SVDQuant Visual Comparison](https://hanlab.mit.edu/projects/svdquant/assets/teaser.jpg)

## Key Features ✨
- 🔥 **4-bit Weights & Activations** - First Diffusers implementation with full W4A4 support
- 🚀 **3.5× Memory Reduction** - Run 12B FLUX models on consumer laptops
- ⚡ **3x Faster** vs NF4 W4A16 baselines
- 🎨 **Preserved Visual Quality** - Outperforms W4A8 baselines on PixArt-Σ
- 🔧 **Post-Training Quantization** - No retraining required

As of Now only w4 is implemented in a quick ipynb notebook , 
