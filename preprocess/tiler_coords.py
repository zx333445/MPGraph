"""
Note: Written for histoprep==1.0.8.
This script extracts tissue patch coordinates and saves them into HDF5 format,
following the CLAM saving format for whole slide image (WSI) processing.
"""

import argparse
import os
import histoprep as hp
import h5py

from tqdm import tqdm


def process_slide(slide_path, 
                  output_dir, 
                  patch_size, 
                  overlap, 
                  level, 
                  max_background, 
                  save_tiles, 
                  tile_format):
    """
    Processes a single WSI to detect tissue and extract patch coordinates.
    """
    try:
        reader = hp.SlideReader(slide_path)
        threshold, tissue_mask = reader.detect_tissue()

        slide_id = reader.slide_name
        print(f"[Processing] Slide: {slide_id} | Dims level 0: {reader.level_dimensions[0]}")

        # 1. Extract coordinates passing the background threshold
        coordinates = reader.get_tile_coordinates(
            width=patch_size,
            overlap=overlap,
            level=level,
            tissue_mask=tissue_mask,
            max_background=max_background
        )

        if len(coordinates) == 0:
            print(f"[Warning] No valid tissue detected for slide {slide_id}. Coordinates count is 0.")
            return

        # 2. Save coordinates in HDF5 format
        h5_dir = os.path.join(output_dir, "coords")
        os.makedirs(h5_dir, exist_ok=True)
        h5_path = os.path.join(h5_dir, f"{slide_id}.h5")

        with h5py.File(h5_path, "w") as hdf5_file:
            hdf5_file.create_dataset("coords", data=coordinates)
            hdf5_file["coords"].attrs["patch_level"] = level
            hdf5_file["coords"].attrs["patch_size"] = patch_size

        # 3. Optional: Extract and save physical patch tile images
        if save_tiles:
            tiles_dir = os.path.join(output_dir, "images")
            os.makedirs(tiles_dir, exist_ok=True)
            
            reader.save_tiles(
                output_dir=tiles_dir,
                coordinates=coordinates,
                level=level,
                image_format=tile_format,
                quality=100
            )

        print(f"[Finished] Slide: {slide_id} | Total Patches: {len(coordinates)} -> Saved to: {h5_path}")

    except Exception as e:
        print(f"[Error] Failed to process slide {slide_path}. Exception: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WSI Tissue Segmentation and Patch Coordinates Extraction")
    
    # Path arguments
    parser.add_argument("--source", type=str, required=True, help="Path to the directory containing input WSI files")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the directory where coordinates and outputs will be saved")
    parser.add_argument("--slide_ext", type=str, default=".svs", help="Extension of slide files (.tif, .svs, .ndpi, .czi).")
    
    # Patch extraction arguments
    parser.add_argument("--patch_size", type=int, default=256, help="Width and height of the patch in pixels.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between adjacent patches in pixels.")
    parser.add_argument("--patch_level", type=int, default=0, help="Pyramid resolution level for extraction.")
    parser.add_argument("--max_background", type=float, default=0.4, help="Maximum allowed background ratio per patch (0.0 - 1.0).")
    
    # Tile saving arguments
    parser.add_argument("--save_tiles", action="store_true", help="Whether to save cropped tile images.")
    parser.add_argument("--tile_format", type=str, default="png", choices=["png", "jpeg"], help="Image format for saved tile images.")

    args = parser.parse_args()

    ext = args.slide_ext if args.slide_ext.startswith(".") else f".{args.slide_ext}"
    os.makedirs(args.save_dir, exist_ok=True)

    slide_files = sorted([
        os.path.join(args.source, f) for f in os.listdir(args.source) if f.endswith(ext)
    ])

    print("=" * 60)
    print(f"Total slides found: {len(slide_files)} (extension: '{ext}')")
    print(f"Source Directory  : {args.source}")
    print(f"Save Directory    : {args.save_dir}")
    print(f"Patch Configuration: size={args.patch_size}, level={args.patch_level}, max_bg={args.max_background}")
    print("=" * 60)

    if not slide_files:
        print("No match files found. Program exiting.")
        exit(0)

    for slide_path in tqdm(slide_files, desc="Processing WSIs", unit="slide"):
        process_slide(
            slide_path=slide_path,
            output_dir=args.save_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            level=args.patch_level,
            max_background=args.max_background,
            save_tiles=args.save_tiles,
            tile_format=args.tile_format
        )

    print("\n[Done] All slides processed successfully.")