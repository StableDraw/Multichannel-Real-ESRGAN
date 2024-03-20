<p align="center">
  <img src="assets/realesrgan_logo_multichannel.png" height=120>
</p>

## This version is a fork of the original [**Xinntao Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) with multi-channel support (including alpha channel support) and some other optimizations and features. Only differences from the original version will be written here. You can read the description and features of the original version on it's page

<div align="center">

👀[**Demos**](#-demos-videos) **|** 🚩[**Updates**](#-updates) **|** ⚡[**Usage**](#-quick-inference) **|** 🏰[**Model Zoo**](docs/model_zoo.md) **|** 🔧[Install](#-dependencies-and-installation)  **|** 💻[Train](docs/Training.md) **|** ❓[FAQ](docs/FAQ.md) **|** 🎨[Contribution](docs/CONTRIBUTING.md)

[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)

</div>

🔥 **AnimeVideo-v3 model (动漫视频小模型)**. Please see [[*anime video models*](docs/anime_video_model.md)] and [[*comparisons*](docs/anime_comparisons.md)]<br>
🔥 **RealESRGAN_x4plus_anime_6B** for anime images **(动漫插图模型)**. Please see [[*anime_model*](docs/anime_model.md)]

Online demo for Multichannel-Real-ESRGAN: [*StableDraw*](https://stabledraw.com/)

Real-ESRGAN aims at developing **Practical Algorithms for General Image/Video Restoration**.<br>
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.


### 📖 Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

<!---------------------------------- Updates --------------------------->
## 🚩 Updates

- ✅ Add the **realesr-general-x4v3** model - a tiny small model for general scenes. It also supports the **-dn** option to balance the noise (avoiding over-smooth results). **-dn** is short for denoising strength.
- ✅ Update the **RealESRGAN AnimeVideo-v3** model. Please see [anime video models](docs/anime_video_model.md) and [comparisons](docs/anime_comparisons.md) for more details.
- ✅ Add small models for anime videos. More details are in [anime video models](docs/anime_video_model.md).
- ✅ Add the ncnn implementation [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).
- ✅ Add [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), which is optimized for **anime** images with much smaller model size. More details and comparisons with [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) are in [**anime_model.md**](docs/anime_model.md)
- ✅ Support finetuning on your own data or paired data (*i.e.*, finetuning ESRGAN). See [here](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)
- ✅ Integrate [GFPGAN](https://github.com/TencentARC/GFPGAN) to support **face enhancement**.
- ✅ Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN). Thanks [@AK391](https://github.com/AK391)
- ✅ Support arbitrary scale with `--outscale` (It actually further resizes outputs with `LANCZOS4`). Add *RealESRGAN_x2plus.pth* model.
- ✅ [The inference code](inference_realesrgan.py) supports: 1) **tile** options; 2) images with **alpha channel**; 3) **gray** images; 4) **16-bit** images.
- ✅ The training codes have been released. A detailed guide can be found in [Training.md](docs/Training.md).

---

<!---------------------------------- Demo videos --------------------------->

## 🔧 Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/Robolightning/Multichannel-Real-ESRGAN
    cd Multichannel-Real-ESRGAN
    ```

1. Install dependent packages

    ```bash
    # We use BasicSR for both training and inference
    # facexlib and gfpgan are for face enhancement
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## ⚡ Quick Inference

There are usually three ways to inference Real-ESRGAN.

1. [From console](#console-usage)
2. [From Python script](#python-script)

### Python script

Description is coming soon

### Console usage

#### Usage of python script in console

1. You can use X4 model for **arbitrary output size** with the argument `outscale`. The program will further perform cheap resize operation after the Real-ESRGAN output.

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

#### Inference general images

4-channeled models is comming soon

Download pre-trained models: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

Inference!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

Results are in the `results` folder

#### Inference anime images

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

Pre-trained models: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)<br>
 More details and comparisons with [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) are in [**anime_model.md**](docs/anime_model.md)

```bash
# download model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
# inference
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

Results are in the `results` folder

---


## 📧 Contact

If you have any question, please email `irobolightning@gmail.com` or to author of the original project `xintao.wang@outlook.com` or `xintaowang@tencent.com`.

<!---------------------------------- Projects that use Real-ESRGAN --------------------------->
## 🧩 Projects that use Multichannel-Real-ESRGAN

If you develop/use Multichannel-Real-ESRGAN in your projects, welcome to let me know.

- Minecraft mod: [MC-textures-upscaler-mod](coming soon) by [Robolightning](https://github.com/Robolightning)

**GUI**

- Web graphic editor with neural networks: [StableDraw](https://stabledraw.com/) by [Robolightning](https://github.com/Robolightning)