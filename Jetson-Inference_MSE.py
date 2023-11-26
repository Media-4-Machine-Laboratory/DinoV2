import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image

def load_image(str) -> Image:
    return Image.open(str).convert("RGB")

###############################

files = ["0000","0001","0002","0003","0004","0005","0006","0007","0008","0013","0014","0017","0020","0024","0025", "0028", "0029","0033","0034", "0038"]

depthimages = []
GT_normed_map = []

for i in range(len(files)) :

    depthimages.append(load_image("/src/input/depth_" + files[i] + ".png"))

    GT_map = np.array(np.array(depthimages[i]))[:,:,0]
    GT_min = GT_map.min(); GT_max = GT_map.max()

    GT_normed_map.append((GT_map - GT_min) / (GT_max - GT_min))

###############################

GTmasks = []

for i in range(len(files)) :
    GTmasks.append(np.where(GT_normed_map[i] > 0.6, 0, 1).astype('uint8')) # 특정 거리 이상은 배경이라 간주할 mask 저장.
    
    # Mask 및 Image에 Mask를 씌운 결과물을 출력하고 싶다면 아래를 주석 해제.
    # mask2 = GTmasks[i] * 255
    # cv.imwrite('/src/output/mask/mask_'+ files[i] + '.jpg', mask2)
    # cv.imwrite('/src/output/mask/img_masked_' + files[i] + '.jpg' , cv.cvtColor(images[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))
    # cv.imwrite('/src/output/mask/gc_masked_' + files[i] + '.jpg' , cv.cvtColor(grabcutimages[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))

###############################

jetson_estimated_normed_map = []
jet_mse_values = []
masked_jet_mse_values = []

for i in range(len(files)) :
    jetson_depth = np.array(Image.open("/src/output/jetson-inference/ji_estimate_" + files[i] + ".jpg").convert("L"))
    jetson_depth_min = jetson_depth.min(); jetson_depth_max = jetson_depth.max()
    jetson_estimated_normed_map.append((jetson_depth - jetson_depth_min) / (jetson_depth_max - jetson_depth_min))

for i in range(len(files)) :
    jetson_mse = np.mean(np.square(GT_normed_map[i] - jetson_estimated_normed_map[i]))
    print(files[i], ' Unmasked : ', jetson_mse)
    masked_jetson_mse = np.mean(np.square(GT_normed_map[i] - jetson_estimated_normed_map[i])[GTmasks[i] == 1])
    print(files[i], 'Masked :', masked_jetson_mse)

    jet_mse_values.append(jetson_mse)
    masked_jet_mse_values.append(masked_jetson_mse)
    print(jet_mse_values[i])

print('-----------------------------')
print('Unmasked Jetson-Inference MSE mean : ', np.mean(np.array(jet_mse_values)))
print('Masked Jetson-Inference MSE mean : ', np.mean(np.array(masked_jet_mse_values)))

###############################

col_names = files
data = [jet_mse_values, masked_jet_mse_values]
DF = pd.DataFrame(data, columns=col_names)
DF.index = ["Jetson-Inference", "Masked Jetson-Inference"]
DF.to_csv("Jetson_MSE.csv")

print(DF)
print('-----------------\nMeans of MSE')
print(DF.mean(axis=1))
