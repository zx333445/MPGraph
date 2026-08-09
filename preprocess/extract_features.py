import argparse
import os
import h5py
import numpy as np
import openslide
import timm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from conch.open_clip_custom import create_model_from_pretrained
except ImportError:
    create_model_from_pretrained = None


"""
WSI Feature Extraction Pipeline (Supports UNI, Virchow2, CONCH, ResNet50)
Usage:
    python extract_features.py \
        --data_dir /path/to/wsis \
        --coords_dir /path/to/coords \
        --save_dir /path/to/output_features \
        --model_name uni \
        --ckpt_path /path/to/UNImodel.bin \
        --slide_ext .svs \
        --batch_size 128
"""


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, h5path, wsi):
        self.wsi = wsi
        self.file_path = h5path
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset) # type: ignore

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx] # type: ignore
        img = self.wsi.read_region(coord[:2], self.patch_level, (self.patch_size, self.patch_size)).convert('RGB') # type: ignore
        trans = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225))])   
        img = trans(img).unsqueeze(0) # type: ignore
        return img, coord[:2] # type: ignore

    @staticmethod
    def collate_features(batch):
        img = torch.cat([item[0] for item in batch], dim = 0)
        coords = np.stack([item[1] for item in batch])
        return [img, coords]


def build_model(model_name: str, ckpt_path: str):
    """Factory function to load pretrained pathology backbone models."""
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity() # type: ignore
        return model

    elif model_name == "uni":
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path), strict=True)
        else:
            print(f"[Warning] No checkpoint loaded for UNI from '{ckpt_path}'")
        return model

    elif model_name == "virchow2":
        timm_kwargs = {
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "reg_tokens": 4,
            "mlp_ratio": 5.3375,
            "global_pool": "",
            "dynamic_img_size": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path), strict=True)

        # Wrap forward method to extract required embeddings
        original_forward = model.forward

        def virchow_forward(x):
            output = original_forward(x)
            class_token = output[:, 0]
            patch_tokens = output[:, 5:]
            return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

        model.forward = virchow_forward
        return model

    elif model_name == "conch":
        if create_model_from_pretrained is None:
            raise ImportError(
                "CONCH requires 'conch' package. Please install it first."
            )
        model = create_model_from_pretrained(
            "conch_ViT-B-16",
            return_transform=False,
            checkpoint_path=ckpt_path,
        )
        return model

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0) # type: ignore
            dset[-data_shape[0]:] = val # type: ignore
    file.close()
    return output_path


def extract_features(args):
    """Main feature extraction pipeline."""
    # Ensure correct extension format
    ext = (
        args.slide_ext
        if args.slide_ext.startswith(".")
        else f".{args.slide_ext}"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize model
    print(f"Loading model: {args.model_name}...")
    model = build_model(args.model_name, args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # type: ignore
    model.eval() # type: ignore

    # Search slides
    slide_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(ext)])
    print(f"Total WSIs found: {len(slide_files)} (extension: '{ext}')")

    for slide_name in slide_files:
        slide_id = os.path.splitext(slide_name)[0]
        out_h5_path = os.path.join(args.save_dir, f"{slide_id}.h5")

        if os.path.exists(out_h5_path):
            print(f"[Skip] {slide_name} already processed.")
            continue

        slide_path = os.path.join(args.data_dir, slide_name)
        coords_h5_path = os.path.join(args.coords_dir, f"{slide_id}.h5")

        if not os.path.exists(coords_h5_path):
            print(f"[Warning] Coords file missing for {slide_name}, skipping...")
            continue

        wsi = openslide.open_slide(slide_path)
        bagset = Whole_Slide_Bag_FP(coords_h5_path, wsi)

        bagloader = DataLoader(
            bagset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=Whole_Slide_Bag_FP.collate_features,
            shuffle=False,
        )

        print(f"\nProcessing {slide_name} | Total patches: {len(bagset)} | Output: {out_h5_path}")

        mode = "w"
        for imgs, coords in tqdm(bagloader, desc=f"Extracting {slide_id}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                if args.model_name.lower() == "conch":
                    features = model.encode_image(imgs, proj_contrast=False, normalize=False) # type: ignore
                else:
                    features = model(imgs) # type: ignore
                features = features.view(imgs.shape[0], -1).cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(out_h5_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    print("\n[Done] Feature extraction finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WSI Feature Extraction for Deep Learning Computational Pathology"
    )

    # Directory arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing original WSI slide files")
    parser.add_argument("--coords_dir", type=str, required=True, help="Directory containing H5 coordinate files",)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where output feature H5 files will be saved")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="uni", choices=["uni", "virchow2", "conch", "resnet50"], help="Backbone model for feature extraction.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to pretrained model weight checkpoint")

    # Extraction parameters
    parser.add_argument("--slide_ext", type=str, default=".svs",help="Slide extension format (.svs, .tif, .ndpi).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers.")

    args = parser.parse_args()
    extract_features(args)