import cv2
import os
import torch
from ..basicsr.utils.img_util import img2tensor, tensor2img
from ..basicsr.utils.download_util import load_file_from_url
from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
#from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from .archs.gfpganv1_clean_arch import GFPGANv1Clean
from RestoreFormer.models.vqgan_v1 import RestoreFormerModel
from .archs.restoreformer_arch import RestoreFormer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale = 2, arch = "clean", channel_multiplier = 2, bg_upsampler = None, device = None, input_is_latent = True):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if model_path.startswith("https://"):
            model_path = load_file_from_url(url = model_path, model_dir = os.path.join(ROOT_DIR, "gfpgan/weights"), progress = True, file_name = None)
                # initialize face helper
        self.face_helper = FaceRestoreHelper(upscale, face_size = 512, crop_ratio = (1, 1), det_model = "retinaface_resnet50", save_ext = "png", use_parse = True, device = self.device, model_rootpath = "gfpgan/weights")

        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(out_size = 512, num_style_feat = 512, channel_multiplier = channel_multiplier, decoder_load_path = None, fix_decoder = False, num_mlp = 8, input_is_latent = input_is_latent, different_w = True, narrow = 1, sft_half = True)
            if not(os.path.isfile(model_path)):
                model_urls = [
                    "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth",
                    "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
                    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                    ]
                mn = model_path[model_path.rfind('/'):]
                for mu in model_urls:
                    if mn in mu:
                        mud = mu
                model_path = load_file_from_url(url = mud, model_dir = os.path.join(ROOT_DIR, "gfpgan/weights"), progress = True, file_name = None)
            loadnet = torch.load(model_path)
            if "params_ema" in loadnet:
                keyname = "params_ema"
            else:
                keyname = "params"
            state_dict = loadnet[keyname]
        # Not working now
        # elif arch == "bilinear":
        #     self.gfpgan = GFPGANBilinear(out_size = 512, num_style_feat = 512, channel_multiplier = channel_multiplier, decoder_load_path = None, fix_decoder = False, num_mlp = 8, input_is_latent = input_is_latent, different_w = True, narrow = 1, sft_half = True)
        # elif arch == 'original':
        #     self.gfpgan = GFPGANv1(out_size = 512, num_style_feat = 512, channel_multiplier = channel_multiplier, decoder_load_path = None, fix_decoder = True, num_mlp = 8, input_is_latent = input_is_latent, different_w = True, narrow = 1, sft_half = True)
        elif "RestoreFormer" in arch:
            if not(os.path.isfile(model_path)):
                import tempfile, tarfile, requests
                response = requests.get("https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/Eb73S2jXZIxNrrOFRnFKu2MBTe7kl4cMYYwwiudAmDNwYg?e=Xa4ZDf")
                file = tempfile.TemporaryFile()
                file.write(response.content)
                if file.endswith("tar.gz"):
                    tar = tarfile.open(file, "r:gz")
                elif file.endswith("tar"):
                    tar = tarfile.open(file, "r:")
                tar.extract("RestoreFormer/last.ckpt", "RestoreFormer/weights/RestoreFormer.ckpt")
                file.close()
                tar.close()
            loadnet = torch.load(model_path)
            sd = loadnet["state_dict"]
            keys = list(sd.keys())
            ddconfig = {
                "target": "RestoreFormer.modules.vqvae.vqvae_arch.VQVAEGANMultiHeadTransformer",
                'params': {
                "embed_dim": 256,
                "n_embed": 1024,
                "double_z": False,
                "z_channels": 256,
                "resolution": 512,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 64,
                "ch_mult": [1, 2, 2, 4, 4, 8],  # num_down = len(ch_mult)-1
                "num_res_blocks": 2,
                "dropout": 0.0,
                "attn_resolutions": [16],
                "enable_mid": True,
                "fix_decoder": False,
                "fix_codebook": True,
                "fix_encoder": False,
                "head_size": 8
                }
            }
            lossconfig = {
                "target": "RestoreFormer.modules.losses.vqperceptual.VQLPIPSWithDiscriminatorWithCompWithIdentity",
                "params": {
                    "disc_conditional": False,
                    "disc_in_channels": 3,
                    "disc_start": 10001,
                    "disc_weight": 0.8,
                    "codebook_weight": 1.0,
                    "use_actnorm": False,
                    "comp_weight": 1.5,
                    "comp_style_weight": 2e3, #2000.0
                    "identity_weight": 3, #1.5
                    "lpips_style_weight": 1e9,
                    "identity_model_path": "RestoreFormer/weights/pretrained_models/arcface_resnet18.pth"
                }
            }
            if arch == "RestoreFormer":
                self.gfpgan = RestoreFormerModel(ddconfig, lossconfig)
            else:
                self.gfpgan = RestoreFormer(n_embed = 1024,
                 embed_dim = 256,
                 ch = 64,
                 out_ch = 3,
                 ch_mult = (1, 2, 2, 4, 4, 8),
                 num_res_blocks = 2,
                 attn_resolutions = (16, ),
                 dropout = 0.0,
                 in_channels = 3,
                 resolution = 512,
                 z_channels = 256,
                 double_z = False,
                 enable_mid = True,
                 fix_decoder = False,
                 fix_codebook = True,
                 fix_encoder = False,
                 head_size = 8)
            state_dict = self.gfpgan.state_dict()
            require_keys = state_dict.keys()
            keys = sd.keys()
            for k in require_keys:
                if k in keys: 
                    state_dict[k] = sd[k]
        self.gfpgan.load_state_dict(state_dict, strict = True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned = False, only_center_face = False, paste_back = True):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face = only_center_face, eye_dist_threshold = 5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb = True, float32 = True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace = True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr = True, min_max = (-1, 1))
            except RuntimeError as error:
                print(f"\tFailed inference for GFPGAN: {error}.")
                restored_face = cropped_face

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale = self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img = bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None