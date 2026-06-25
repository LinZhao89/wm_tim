import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import pickle
import numpy as np 
from skimage.filters import rank 
from skimage.morphology import disk 
from PIL import Image 
import random
from skimage.restoration import denoise_nl_means, estimate_sigma
from patchcore.datasets.wafer_transforms import (
    DEFAULT_WAFER_BACKGROUND,
    transform_wafer,
)
_CLASSNAMES = [
    "bottle",
    "testname"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Wafermap-specific statistics (computed from training data)
# To compute these, run: python bin/compute_dataset_stats.py dataset/wm811k <subdataset>
# Replace these values with output from compute_dataset_stats.py
WM811K_MEAN = [0.485, 0.456, 0.406]  # TODO: Update with computed values
WM811K_STD = [0.229, 0.224, 0.225]   # TODO: Update with computed values

# Choose which normalization to use (set via constructor parameter)
DEFAULT_MEAN = WM811K_MEAN  # Change to IMAGENET_MEAN to use ImageNet stats
DEFAULT_STD = WM811K_STD    # Change to IMAGENET_STD to use ImageNet stats


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "train"
    TEST = "test"


class wm811kDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        use_imagenet_stats=False,
        grayscale=False,
        apply_filter=True,
        filter_window_size=3,
        filter_threshold=1.25,
        transform_mode="resize_pad",
        pad_color=DEFAULT_WAFER_BACKGROUND,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
            grayscale: [bool]. If True, converts images to single channel.
            apply_filter: [bool]. If True, applies the constrained_mean_filter.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.grayscale = grayscale
        self.apply_filter = apply_filter
        self.filter_window_size = filter_window_size
        self.filter_threshold = filter_threshold
        self.resize = resize
        self.center_crop_size = imagesize
        self.transform_mode = transform_mode
        self.pad_color = tuple(pad_color)

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        if self.grayscale:
            # For grayscale, always use a 1-channel mean/std. Safely take the first value.
            norm_mean = [WM811K_MEAN[0]] if isinstance(WM811K_MEAN, list) and WM811K_MEAN else [0.5]
            norm_std = [WM811K_STD[0]] if isinstance(WM811K_STD, list) and WM811K_STD else [0.5]
        else:
            # For 3-channel, determine the correct mean/std
            norm_mean = IMAGENET_MEAN if use_imagenet_stats else WM811K_MEAN
            norm_std = IMAGENET_STD if use_imagenet_stats else WM811K_STD

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.imagesize = (3, imagesize, imagesize)

    def _transform_image(self, image, interpolation, fill=None):
        fill = self.pad_color if fill is None else fill
        return transform_wafer(
            image, self.resize, self.center_crop_size, interpolation,
            mode=self.transform_mode, fill=fill)
    
    
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        #print(f'classname: {classname}')
        #print(f'anomaly: {anomaly}')
        #print(f'image_path:{image_path}')
        #print(f'mask_path: {mask_path}')
        #print(f'----------------------\n')
        image = PIL.Image.open(image_path).convert("RGB")
        
        if self.apply_filter:
            image = self.constrained_mean_filter(
                image,
                filter_window_size=self.filter_window_size,
                threshold=self.filter_threshold,
            )
        #image = self.non_local_means_filter(image, 6,3, 0.25, 0.22)
        #if('/good/' not in image_path):
            #print(image_path)
        #    image.save(image_path.split('/')[-1])

        transformed = self._transform_image(
            image, transforms.InterpolationMode.BILINEAR)
        if self.grayscale:
            transformed = transforms.Grayscale(num_output_channels=1)(transformed)
        raw_image = transforms.ToTensor()(transformed)
        image = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)(raw_image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self._transform_image(
                mask, transforms.InterpolationMode.NEAREST, fill=0)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "raw_image": raw_image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }
    # slow
    # def constrained_mean_filter(self, wbm, filter_window_size, threshold): #1.25(used for experiment)
    #     gray_img = wbm.convert("L")
    #     gray_img_arr = np.array(gray_img)
    #     wbm = np.ones_like(gray_img)
    #     for x in range(wbm.shape[0]):
    #         for y in range(wbm.shape[1]):
    #             if gray_img_arr[x,y]>200:
    #                 wbm[x, y]=2
    #             elif  gray_img_arr[x, y]>100 and gray_img_arr[x, y]<=200:
    #                 wbm[x, y]=0.5
    #             else:
    #                 wbm[x, y]=0

    #     padded_wbm = np.pad(wbm, ((filter_window_size // 2, filter_window_size // 2), (filter_window_size // 2, filter_window_size // 2)), 'constant')
    #     filtered_wbm = wbm.copy()
    #     for x in range(wbm.shape[0]):
    #         for y in range(wbm.shape[1]):
    #             if wbm[x, y] == 2:  # Defective grain
    #                 neighborhood = padded_wbm[x:x + filter_window_size, y:y + filter_window_size]
    #                 mean_value = np.mean(neighborhood)
    #                 if mean_value < threshold:
    #                     filtered_wbm[x, y] = 1  # Convert to normal grain
    #     # test, seems like with below the result is not so good
    #     for x in range(filtered_wbm.shape[0]):
    #         for y in range(filtered_wbm.shape[1]):
    #             if filtered_wbm[x,y]==2:
    #                 filtered_wbm[x, y]=255
    #             else:
    #                 filtered_wbm[x, y]=0
        

    #     filtered_image = PIL.Image.fromarray(filtered_wbm)
    #     rgb_img = filtered_image.convert("RGB")
    #     return rgb_img
    @staticmethod
    def constrained_mean_filter(wbm, filter_window_size, threshold):
        # Keep original colors so we can rebuild a 3-color output later.
        rgb_arr = np.array(wbm.convert("RGB"))
        gray_img_arr = np.array(wbm.convert("L"))

        # Vectorized thresholding into three semantic regions.
        class_map = np.ones_like(gray_img_arr, dtype=np.float32)
        class_map[gray_img_arr > 200] = 2  # bright anomalies/noise
        class_map[(gray_img_arr > 100) & (gray_img_arr <= 200)] = 0.5  # wafer surface
        class_map[gray_img_arr <= 100] = 0  # background

        from scipy.ndimage import uniform_filter
        mean_map = uniform_filter(class_map, size=filter_window_size, mode="constant")

        filtered_classes = class_map.copy()
        anomaly_mask = (class_map == 2) & (mean_map < threshold)
        filtered_classes[anomaly_mask] = 1  # treat as wafer after smoothing

        # Build per-class colors using the original image as reference to retain the
        # dataset's visual cues (purple background, green wafer, yellow anomalies).
        def _mean_color(mask, default):
            if np.any(mask):
                return np.mean(rgb_arr[mask], axis=0)
            return np.array(default, dtype=np.float32)

        background_color = _mean_color(class_map == 0, default=(40, 20, 60))
        wafer_color = _mean_color(class_map == 0.5, default=(110, 190, 90))
        anomaly_color = _mean_color(class_map == 2, default=(250, 240, 80))

        colorized = np.zeros_like(rgb_arr, dtype=np.uint8)
        bg_mask = filtered_classes == 0
        wafer_mask = (filtered_classes == 0.5) | (filtered_classes == 1)
        anomaly_mask_final = filtered_classes == 2

        colorized[bg_mask] = background_color
        colorized[wafer_mask] = wafer_color
        colorized[anomaly_mask_final] = anomaly_color

        return PIL.Image.fromarray(colorized, mode="RGB")
    
    def non_local_means_filter(self, imgrgb, patch_size=6, patch_distance=3, h_idx = 0.2,  th = 0.3,  h=None):
        # Read the image
        gray_img = imgrgb.convert("L")
        gray_img_arr = np.array(gray_img)
        wbm=np.ones_like(gray_img)
    
        for x in range(wbm.shape[0]):
            for y in range(wbm.shape[1]):
                if gray_img_arr[x,y]>200:
                    wbm[x, y]=255
                elif  gray_img_arr[x, y]>100 and gray_img_arr[x, y]<=200:
                    wbm[x, y]=0
                else:
                    wbm[x, y]=0
        # Estimate the noise standard deviation from the image (optional)
        #gray_img = img.convert("L")
        if h is None:
            sigma_est = np.mean(estimate_sigma(wbm))
            #print(sigma_est) # 29.822
            h = h_idx* sigma_est #0.2 is good
                       
        # Apply Non-Local Means filter
        filtered_image = denoise_nl_means(wbm, h=h, fast_mode=False,patch_size=patch_size, patch_distance=patch_distance)
                                                                        
        for x in range(filtered_image.shape[0]):
            for y in range(filtered_image.shape[1]):
                if filtered_image[x,y]>th:
                   filtered_image[x, y]=255
                else:
                   filtered_image[x, y]=0
        rgb_img = PIL.Image.fromarray(filtered_image)
        imL = rgb_img.convert("RGB")
        return imL



    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            #print(f'classname:{classname}\n')
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)
            print(f'classpath:{classpath}\n')
            #print(f'maskpath:{maskpath}\n')
           

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                # To disable "good" class:
                # if anomaly == "good": continue

                # print("anomaly type:", anomaly)
                #print(f'classpath:{classpath}\n')
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                #print(f'anomaly files: {anomaly_files}\n')
                #print('---------------------------------')
                #print(classname)
                #print(f'anomaly is {anomaly}')
                #print(f'imgpaths_per_class: {imgpaths_per_class[classname][anomaly]}')

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    if os.path.isdir(anomaly_mask_path):
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        
                        # Match mask filenames to image filenames
                        # Assumes mask filename is derived from image filename (e.g., same name, or known suffix)
                        # Here we assume mask has same basename as image
                        
                        full_mask_paths = []
                        mask_by_stem = {}
                        for mask_name in anomaly_mask_files:
                            stem = os.path.splitext(mask_name)[0]
                            while stem.lower().endswith("_mask"):
                                stem = stem[:-5]
                            mask_by_stem.setdefault(stem, []).append(mask_name)
                        
                        for img_p in imgpaths_per_class[classname][anomaly]:
                            img_name = os.path.basename(img_p)
                            # Logic for mask matching: 
                            # If mask file exists with same name, use it.
                            # If not, use None (which will become zero mask later).
                            
                            # Note: Filesystems might be case-sensitive or have different extensions?
                            # Assuming exact match for simplicity or _mask suffix if standard MVTec
                            # But code before just listed directory. We need robust matching.
                            
                            # Try exact match
                            image_stem = os.path.splitext(img_name)[0]
                            mask_name = self._select_mask_name(
                                image_stem, mask_by_stem.get(image_stem, [])
                            )
                            full_mask_paths.append(
                                os.path.join(anomaly_mask_path, mask_name)
                                if mask_name else None
                            )
                                
                        maskpaths_per_class[classname][anomaly] = full_mask_paths
                    else:
                         # No ground_truth folder for this anomaly type
                         maskpaths_per_class[classname][anomaly] = [None] * len(imgpaths_per_class[classname][anomaly])

                else:
                    maskpaths_per_class[classname]["good"] = None


        #print(maskpaths_per_class)
        #print(imgpaths_per_class)
        # Unrolls the data dictionary to an easy-to-iterate list.
        #print(f'imgpaths_per_class:{imgpaths_per_class}')
        #print(f'maskpaths_per_class:{maskpaths_per_class}')
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    #print(i)
                    #print(f'image pah: {image_path}')
                    data_tuple = [classname, anomaly, image_path]
                    #print(f'data tuple: {data_tuple}')
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        #print(classname)
                        #print(anomaly)
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        #print('**********')
        #print(data_to_iterate)
        #print('*********')
        return imgpaths_per_class, data_to_iterate

    @staticmethod
    def _select_mask_name(image_stem, candidates):
        if not candidates:
            return None
        by_name = {name.lower(): name for name in candidates}
        for preferred in (f"{image_stem}_mask.png", f"{image_stem}.png"):
            if preferred.lower() in by_name:
                return by_name[preferred.lower()]
        return sorted(candidates, key=lambda name: (name.lower().count("_mask"), name))[0]
