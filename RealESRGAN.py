import cv2
import io
import os
import numpy as np
import torch
from PIL import Image
from edited_scripts.basicsr.archs.rrdbnet_arch import RRDBNet
from edited_scripts.basicsr.utils.download_util import load_file_from_url
from edited_scripts.realesrgan_utils import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer
from tqdm import tqdm

def RealESRGAN_upscaler(binary_data, args):
    # определяет модели в соответствии с выбранной моделью
    model_name = args["model"]
    if model_name == "RealESRGAN_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
    elif model_name == "RealESRNet_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # модель x4 RRDBNet с 6 блоками
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 6, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]
    elif model_name == "RealESRGAN_x2plus":  # модель x2 RRDBNet
        model = RRDBNet(num_in_ch = 4, num_out_ch = 4, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 2)
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
    elif model_name == "realesr-animevideov3":  # модель x4 VGG-стиля (размера XS)
        model = SRVGGNetCompact(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_conv = 16, upscale = 4, act_type = "prelu")
        netscale = 4
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
    elif model_name == "realesr-general-x4v3":  # модель x4 VGG-стиля (размера S)
        model = SRVGGNetCompact(num_in_ch = 4, num_out_ch = 4, num_feat = 64, num_conv = 32, upscale = 2, act_type = "prelu")
        netscale = 2
        file_url = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"]
    


    '''
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # путь к модели будет обновлён
            model_path = load_file_from_url(url = url, model_dir = os.path.join(ROOT_DIR, 'weights'), progress = True, file_name = None)
    '''
    

    model_path = "C:\\repos\\Real-ESRGAN\\experiments\\debug_train_RealESRGANx2plus_400k_B12G4_pairdata\\models\\net_g_" + args["temp_param"] + "000.pth"



    # использовать dni для контроля силы удаления шума
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and args["denoise_strength"] != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args["denoise_strength"], 1 - args["denoise_strength"]]
    # восстановитель
    upsampler = RealESRGANer(scale = netscale, model_path = model_path, dni_weight = dni_weight, model = model, tile = args["tile"], tile_pad = args["tile_pad"], pre_pad = args["pre_pad"], half = not args["fp32"], gpu_id = args["gpu-id"])
    if args["face_enhance"]:  # Использовать GFPGAN для улучшения лиц
        face_enhancer = GFPGANer(model_path = "/checkpoints/RealESRGAN/GFPGANv1.3.pth", upscale = args["outscale"], arch = "clean", channel_multiplier = 2, bg_upsampler = upsampler)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(binary_data)).convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    try:
        if args["face_enhance"]:
            _, _, output = face_enhancer.enhance(img, has_aligned = False, only_center_face = False, paste_back = True)
        else:
            output, _ = upsampler.enhance(img, outscale = args["outscale"])
    except RuntimeError as error:
        print("Ошибка", error, "\nЕсли у вас появляется ошибка \"CUDA out of memory\", попробуйте поставить параметру \"tile\" меньшее значение")
    else:
        im_buf_arr = cv2.imencode(".png", output)[1]
        result_binary_data = im_buf_arr.tobytes()
    torch.cuda.empty_cache()
    return result_binary_data

def get_img_type(i_name):
    img = Image.open(tp + "test\\" + filename).convert("RGBA")
    w, h = img.size
    bl = False
    wh = False
    o_clr = 0
    for i in range(w):
        for j in range(h):
            clr = img.getpixel((i, j))
            if clr[3] == 255:
                bl = True
            elif clr[3] == 0:
                wh = True
            else:
                if o_clr > 0:
                    return (0)
                o_clr += 1
                th = clr
    img.close()
    if o_clr == 0:
        return (bl, wh)
    return (bl, wh, th)

if __name__ == '__main__':
    
    it = 405
    start = 5

    tp = "C:\\repos\\Real-ESRGAN\\experiments\\debug_train_RealESRGANx2plus_400k_B12G4_pairdata\\visualization\\"
    for j in tqdm(range(start, it + 1, 5)):
        params = {
            "model": "RealESRGAN_x2plus",       #Модель для обработки ("RealESRGAN_x4plus" - модель x4 RRDBNet, "RealESRNet_x4plus" - модель x4 RRDBNet, "RealESRGAN_x4plus_anime_6B" - модель x4 RRDBNet с 6 блоками, "RealESRGAN_x2plus" - модель x2 RRDBNet, "realesr-animevideov3" - модель x4 VGG-стиля (размера XS), "realesr-general-x4v3" - модель x4 VGG-стиля (размера S)) 
            "denoise_strength": 0.0,            #Сила удаления шума. 0 для слабого удаления шума (шум сохраняется), 1 для сильного удаления шума. Используется только для модели "realesr-general-x4v3"
            "outscale": 2,                      #Величина того, во сколько раз увеличть разшрешение изображения (модель "RealESRGAN_x2plus" x2, остальные x4)
            "tile": 0,                          #Размер плитки, 0 для отсутствия плитки во время тестирования
            "tile_pad": 10,                     #Заполнение плитки
            "pre_pad": 0,                       #Предварительный размер заполнения на каждой границе
            "face_enhance": False,              #Использовать GFPGAN улучшения лиц
            "fp32": True,                       #Использовать точность fp32 во время вывода. По умолчанию fp16 (половинная точность)
            "alpha_upsampler": "realesrgan",    #Апсемплер для альфа-каналов. Варианты: realesrgan | bicubic
            "gpu-id": None,                      #Устройство gpu для использования (по умолчанию = None) может быть 0, 1, 2 для обработки на нескольких GPU
            #на данный момент "max_dim": pow(1024, 2) ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
            "temp_param": str(j)
        }

        dp = tp + str(j) + "k"
        if not os.path.exists(dp):
            os.mkdir(dp)
        for root, _, files in os.walk(tp + "test\\"):  
            for filename in files:
                with open(tp + "test\\" + filename, "rb") as f:
                    init_img_binary_data = f.read()
                t = get_img_type(tp + "test\\" + filename)
                binary_data = RealESRGAN_upscaler(init_img_binary_data, params)
                if t != 0:
                    ri = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                    w, h = ri.size
                    if len(t) == 2:
                        for i in range(w):
                            for j in range(h):
                                clr = list(ri.getpixel((i, j)))
                                clr[3] = 255 * round(clr[3] / 255.0)
                                ri.putpixel((i, j), tuple(clr))
                    else:
                        if t[0] == False and t[1] == False:
                            for i in range(w):
                                for j in range(h):
                                    clr = list(ri.getpixel((i, j)))
                                    clr[3] = t[2]
                                    ri.putpixel((i, j), tuple(clr))
                        elif t[0] == True and t[1] == True:
                            ba = float(255 - t[2])
                            tclr = float(t[2])
                            for i in range(w):
                                for j in range(h):
                                    clr = list(ri.getpixel((i, j)))
                                    if clr[3] < t[2]: 
                                        clr[3] = t[2] * round(clr[3] / tclr)
                                    else:
                                        clr[3] = t[2] + (255 - t[2]) * round((clr[3] - t[2]) / ba)
                                    ri.putpixel((i, j), tuple(clr))
                        elif t[0] == True:
                            for i in range(w):
                                for j in range(h):
                                    ba = 255 - t[2]
                                    clr = list(ri.getpixel((i, j)))
                                    if clr[3] < t[2]:
                                        clr[3] = t[2]
                                    else:
                                        clr[3] = t[2] + ba * round((clr[3] - t[2]) / float(ba))
                                    ri.putpixel((i, j), tuple(clr))
                        else:
                            for i in range(w):
                                for j in range(h):
                                    clr = list(ri.getpixel((i, j)))
                                    if clr[3] > t[2]:
                                        clr[3] = t[2]
                                    else:
                                        clr[3] = t[2] * round(clr[3] / float(t[2]))
                                    ri.putpixel((i, j), tuple(clr))
                            
                    ri.resize((4096, 4096), 4).save(dp + "\\" + filename[:-4] + " big.png") 
                else:
                    Image.open(io.BytesIO(binary_data)).convert("RGBA").resize((4096, 4096), 4).save(dp + "\\" + filename[:-4] + " big.png")
                Image.open(io.BytesIO(init_img_binary_data)).convert("RGBA").resize((4096, 4096), 4).save(dp + "\\" + filename[:-4] + " small.png")
                