import math
import itertools
import cv2 as cv
import torch
import torch.nn.functional as F
import urllib
import mmcv
import matplotlib
import pandas as pd
import numpy as np

from torchvision import transforms
from functools import partial
from mmcv.runner import load_checkpoint
from dinov2.eval.depth.models import build_depther
from PIL import Image

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def load_image(str) -> Image:
    return Image.open(str).convert("RGB")

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

def render_depth(values, colormap_name="gray") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

###############################

BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

###############################

HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()

###############################

files = ["0000","0001","0002","0003","0004","0005","0006","0007","0008","0013","0014","0017","0020","0024","0029","0033","0034","0038"]

images = []
depthimages = []
GT_normed_map = []

for i in range(len(files)) :
    images.append(load_image("/src/input/rgb_" + files[i] + ".jpg"))
    depthimages.append(load_image("/src/input/depth_" + files[i] + ".png"))

    GT_map = np.array(np.array(depthimages[i]))[:,:,0]
    GT_min = GT_map.min(); GT_max = GT_map.max()

    GT_normed_map.append((GT_map - GT_min) / (GT_max - GT_min))

###############################

grabcutimages = []

for i in range(len(images)) :

    be_converted_img = np.array(images[i])	# 영상 읽기

    be_converted_img = cv.cvtColor(be_converted_img, cv.COLOR_BGR2RGB) # OpenCV로 읽어왔기 때문에 BGR을 RGB로 해야 원래 색으로 변환됨.

    rc = (100,100,400,380)      # 대략적인 얼굴 Grabcut 영역 생성.

    mask=np.zeros((be_converted_img.shape[0],be_converted_img.shape[1]),np.uint8)
    mask[:,:]=cv.GC_PR_BGD		# 모든 화소를 배경일 것 같음으로 초기화

    cv.grabCut(be_converted_img, mask, rc, None, None, 5, cv.GC_INIT_WITH_RECT) # 생성한 영역을 기반으로 사각형 영역을 Grabcut.
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8') # 추정된 배경과 물체를 각각 실제 배경과 물체로 확정.
    grab=np.where(mask2[:,:,np.newaxis] == 0, 255, be_converted_img*mask2[:,:,np.newaxis]) # 배경이면 255(흰색), 아니면 기존값(기존값 * 1)

    cv.imwrite('/src/output/grabcut/GC_image_' + files[i] + '.jpg', grab) # 파일 저장.
    grabcutimages.append(load_image("/src/output/grabcut/GC_image_" + files[i] + ".jpg")) # 리스트에 저장된 파일들 삽입.

###############################

estimated_depth_map = [] # 추정된 Depth를 보관할 리스트 선언.
estimated_depth_normed_map = [] # 정규화한 Depth를 보관할 리스트 선언.

for i in range(len(images)) :

    transform = make_depth_transform()

    scale_factor = 1
    rescaled_image = images[i].resize((scale_factor * images[i].width, scale_factor * images[i].height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    estimated_depth = np.array(np.array(render_depth(result.squeeze().cpu()))[:,:,0]) # 추출한 Depth를 보관할 map 선언 및 초기화.
    estimated_depth_min = estimated_depth.min(); estimaed_depth_max = estimated_depth.max() # min 값과 max값 저장.
    estimated_depth_normed_map.append((estimated_depth - estimated_depth_min) # min - max 정규화 실시 및 리스트에 저장.
                                      / (estimaed_depth_max - estimated_depth_min))
    
    cv.imwrite('/src/output/dinov2/normal_' + files[i] + '.jpg', estimated_depth)

###############################

grabcut_estimated_depth_map = [] # Grabcut 한 이미지를 바탕으로 추출할 Depth를 보관할 리스트.
grabcut_estimated_depth_normed_map = [] # 추출한 Depth를 정규화한 값을 저장할 리스트.

for i in range(len(grabcutimages)) :
    scale_factor = 1
    rescaled_image = grabcutimages[i].resize((scale_factor * grabcutimages[i].width, scale_factor * grabcutimages[i].height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    grabcut_estimated_depth = np.array(np.array(render_depth(result.squeeze().cpu()))[:,:,0]) # 추출한 Depth 저장.
    grabcut_estimated_depth_min = grabcut_estimated_depth.min(); grabcut_estimated_depth_max = grabcut_estimated_depth.max() # Depth의 min, max값 저장.
    grabcut_estimated_depth_normed_map.append((grabcut_estimated_depth - grabcut_estimated_depth_min) # min - max 정규화 실시 및 리스트에 저장.
                                      / (grabcut_estimated_depth_max - grabcut_estimated_depth_min))
    
    cv.imwrite('/src/output/dinov2/GC_' + files[i] + '.jpg', grabcut_estimated_depth)

###############################

GTmasks = []

for i in range(len(files)) :
    GTmasks.append(np.where(GT_normed_map[i] > 0.6, 0, 1).astype('uint8')) # 특정 거리 이상은 배경이라 간주할 mask 저장.
    mask2 = GTmasks[i] * 255
    cv.imwrite('/src/output/mask/mask_'+ files[i] + '.jpg', mask2)

    # cv.imwrite('/src/output/mask/img_masked_' + files[i] + '.jpg' , cv.cvtColor(images[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))
    # cv.imwrite('/src/output/mask/gc_masked_' + files[i] + '.jpg' , cv.cvtColor(grabcutimages[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))

###############################

mse_values = []

for i in range(len(files)) :
    mse = np.mean(np.square(GT_normed_map[i] - estimated_depth_normed_map[i]))
    print(files[i], ' Unmasked : ', mse)
    mse = np.mean(np.square(GT_normed_map[i] - estimated_depth_normed_map[i])[GTmasks[i] == 1])
    print(files[i], ' Masked : ', mse)
    mse_values.append(mse)

print('---------------------')
print(np.mean(np.array(mse_values)))

###############################

grabcut_mse_values = []

for i in range(len(files)) :
    grabcut_mse = np.mean(np.square(GT_normed_map[i] - grabcut_estimated_depth_normed_map[i]))
    print(files[i], ' Unmasked : ', grabcut_mse)
    grabcut_mse = np.mean(np.square(GT_normed_map[i] - grabcut_estimated_depth_normed_map[i])[GTmasks[i] == 1])
    print(files[i], 'Masked :', grabcut_mse)

    # grabcut_mse_values.append(((GT_normed_map[i] - grabcut_estimated_depth_normed_map[i]) ** 2).sum() / (GT_normed_map[i].shape[0] * GT_normed_map[i].shape[1]))
    # print(grabcut_mse_values[i])
    
    grabcut_mse_values.append(grabcut_mse)

print('---------------------')
print(np.mean(np.array(grabcut_mse_values)))

###############################

jetson_estimated_normed_map = []
jet_mse_values = []

for i in range(len(files)) :
    jetson_depth = np.array(Image.open("/src/output/jetson-inference/ji_estimate_" + files[i] + ".jpg").convert("L"))
    jetson_depth_min = jetson_depth.min(); jetson_depth_max = jetson_depth.max()
    jetson_estimated_normed_map.append((jetson_depth - jetson_depth_min) / (jetson_depth_max - jetson_depth_min))

for i in range(len(files)) :
    jetson_mse = np.mean(np.square(GT_normed_map[i] - jetson_estimated_normed_map[i]))
    print(files[i], ' Unmasked : ', jetson_mse)
    jetson_mse = np.mean(np.square(GT_normed_map[i] - jetson_estimated_normed_map[i])[GTmasks[i] == 1])
    print(files[i], 'Masked :', jetson_mse)

    jet_mse_values.append(jetson_mse)
    print(jet_mse_values[i])

print('-----------------------------')
print(np.mean(np.array(jet_mse_values)))

###############################

col_names = files
data = [mse_values, grabcut_mse_values, jet_mse_values]
DF = pd.DataFrame(data, columns=col_names)
DF.index = ["Normal", "Grabcut", "Jetson-Inference"]
DF.to_csv("MSE.csv")

print(DF)
print('-----------------\nMeans of MSE')
print(DF.mean(axis=1))
