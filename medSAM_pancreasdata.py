#%matplotlib ipympl
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
import torch
from ipywidgets import interact, widgets
import ipympl
from segment_anything import sam_model_registry
from utils.demo import BboxPromptDemo, PointPromptDemo, BboxPromptDemo_nii, BboxPromptDemo_nii_without

image_path= "D:/Stage/Pancreas_dataset/t1/imagesTr/NYU_0162_0000.nii"
modality='MRI'
nii_data = nib.load(image_path).get_fdata()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% 

#Load model
model_path = 'D:\\Stage\\MedSAM' 
MedSAM_CKPT_PATH = model_path + "\\medsam_vit_b.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


#%%Loop over slices
for slice in range(nii_data.shape[2]-50):
    image_data = nii_data[:,:,slice]

    #%%Normalization
    if modality =='CT':
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0)
    else:
        lower_bound, upper_bound = np.percentile(
                        image_data[image_data > 0], 0.5
                    ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
                        (image_data_pre - np.min(image_data_pre))
                        / (np.max(image_data_pre) - np.min(image_data_pre))
                        * 255.0
                    )           
        image_data_pre[image_data == 0] = 0

    image_data_pre = np.uint8(image_data_pre)
    #%% Reshape
    if len(image_data_pre.shape) == 2:
        img_3c = np.repeat(image_data_pre[:, :, None], 3, axis=-1)
    else:
        img_3c = image_data_pre
    H, W, _ = img_3c.shape


    # %% image preprocessing
    img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True, mode= 'constant'
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # plt.imshow(img_1024, cmap='gray')
    # plt.axis('off')
    # plt.show()

    box_np = [95,255,190,350]
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    bbox_prompt_demo = BboxPromptDemo_nii(medsam_model)
    bbox_prompt_demo.show(image=img_1024, image_tensor=img_1024_tensor, image_path=image_path, slice=slice)
    

    # with torch.no_grad():
    #     image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    # medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    # io.imsave(
    # join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    # medsam_seg,
    # check_contrast=False,
    # )



