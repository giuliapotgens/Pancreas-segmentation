
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
import torch
import cv2

path= r"D:\Stage\Pancreas_dataset\t1\imagesTr\AHN_0001_0000.nii\AHN_0001_0000.nii"
modality = 'MRI'
nii_data = nib.load(path).get_fdata()

path_label = r"D:\Stage\Pancreas_dataset\t1\labelsTr\AHN_0001.nii\AHN_0001.nii"
nii_data_label = nib.load(path_label).get_fdata()

plt.imshow(nii_data_label[:,:,25], cmap='gray')
plt.axis('off')
plt.show()


print(nii_data.shape)
print(nii_data.max(), nii_data.min())

# plt.imshow(nii_data[:,:,10], cmap='gray')
# plt.axis('off')
# plt.show()

nii_slice=nii_data[:,:,12]

#Normalization
if modality == 'MRI':
    lower_bound, upper_bound = np.percentile(
                    nii_slice[nii_slice > 0], 0.5
                ), np.percentile(nii_slice[nii_slice > 0], 99.5)
    image_data_pre = np.clip(nii_slice, lower_bound, upper_bound)
    image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
                
    image_data_pre[nii_slice == 0] = 0

image_data_pre = np.uint8(image_data_pre)

#%% Reshape
if len(image_data_pre.shape) == 2:
    img_3c = np.repeat(image_data_pre[:, :, None], 3, axis=-1)
else:
    img_3c = image_data_pre
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True, mode='constant'
).astype(np.uint8)
#img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0)
)

# plt.imshow(img_1024, cmap='gray')
# plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(nii_slice, cmap='gray')
plt.axis('off')  # Hide axes

# Plot the second image in the second subplot (1 row, 2 columns, 2nd position)
plt.subplot(1, 2, 2)
plt.imshow(img_1024, cmap='gray')
plt.axis('off')  # Hide axes

plt.tight_layout()  # Adjust spacing to prevent overlap
plt.show()

# for i in range(nii_data.shape[2]):
#     plt.figure()
#     plt.imshow(nii_data[:,:,i], cmap='gray')
#     plt.axis('off')
#     plt.show()
