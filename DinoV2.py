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

############################### 파일 이름 지정.

files = ["0000","0001","0002","0003","0004","0005","0006","0007","0008","0013","0014","0017","0020","0024","0029","0033","0034","0038"]

###############################

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

BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

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

GTmasks = []

for i in range(len(files)) :
    GTmasks.append(np.where(GT_normed_map[i] > 0.6, 0, 1).astype('uint8')) # 특정 거리 이상은 배경이라 간주할 mask 저장.
    
    # Image에 Mask를 씌운 결과물을 출력하고 싶다면 아래를 주석 해제.
    # mask2 = GTmasks[i] * 255
    # cv.imwrite('/src/output/mask/mask_'+ files[i] + '.jpg', mask2)
    # cv.imwrite('/src/output/mask/img_masked_' + files[i] + '.jpg' , cv.cvtColor(images[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))
    # cv.imwrite('/src/output/mask/gc_masked_' + files[i] + '.jpg' , cv.cvtColor(grabcutimages[i] * GTmasks[i][:,:,np.newaxis], cv.COLOR_BGR2RGB))

###############################

mse_values = []
masked_mse_values = []

for i in range(len(files)) :
    mse = np.mean(np.square(GT_normed_map[i] - estimated_depth_normed_map[i]))
    print(files[i], ' Unmasked : ', mse)
    masked_mse = np.mean(np.square(GT_normed_map[i] - estimated_depth_normed_map[i])[GTmasks[i] == 1])
    print(files[i], ' Masked : ', masked_mse)

    mse_values.append(mse)
    masked_mse_values.append(masked_mse)

print('---------------------')
print('Unmasked DinoV2 MSE mean : ', np.mean(np.array(mse_values)))
print('Masked DinoV2 MSE mean : ', np.mean(np.array(masked_mse_values)))

###############################

col_names = files
data = [mse_values, masked_mse_values]
DF = pd.DataFrame(data, columns=col_names)
DF.index = ["DinoV2", "Masked DinoV2"]
DF.to_csv("DinoV2_MSE.csv")

print(DF)
print('-----------------\nMeans of MSE')
print(DF.mean(axis=1))
