import os
from PIL import Image
from tqdm import tqdm

c = 405
step = 2
ho = 10
path = "C:\\repos\\Real-ESRGAN\\experiments\\debug_train_RealESRGANx2plus_400k_B12G4_pairdata\\visualization\\"
s = 4096
count = (c // 5)
cc = int(count / step)
co = cc // ho
if cc % ho != 0:
    co += 1
fl = []
for root, _, files in os.walk(path + "test\\"):  
    for filename in files:
        fl.append(filename)
for i in tqdm(range(25)):
    img = Image.new("RGBA", (ho * s, s * co))
    ic = 0
    for j in range(5, count, step):
        pimg = Image.open(path + str(j * 5) + "k\\" + fl[i][:-4] + " big.png").convert("RGBA")
        img.paste(pimg, (s * (ic % ho), s * (ic // ho)))
        ic += 1
    img.save(path + "common\\" + fl[i])