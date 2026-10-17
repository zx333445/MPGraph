# A generalizable framework for interpretable histopathology prognostic biomarker identification

<img src='docs/main.jpg' width="1400px" align="center"/>


## Installation
1. Install [Anaconda](https://www.anaconda.com/distribution/)

2. Clone this reposity:
```shell
git clone https://github.com/zx333445/MPGraph.git
```

3. Create conda environment and install dependencies:
```shell
conda env create -f env.yaml
```

## Checkpoint Download
The pretrained MPGraph model can be accessed from [here](https://drive.google.com/file/d/1ASLomY_9GHqK_bE42ovbJ-zpHOeY0jF2/view?usp=drive_link)


## Image Processing Pipeline
### Extract Tiles from Whole Slide Images
This script extracts tissue patch coordinates and saves them into HDF5 format, `preprocess/tiler_coords.py`
following the [CLAM](https://github.com/mahmoodlab/CLAM) saving format for whole-slide image (WSI) processing.

```python
python tiler_coords.py \
    --data_dir /path/to/wsis \
    --coords_dir /path/to/coords \
    --save_dir /path/to/output_features \
    --model_name uni \
    --ckpt_path /path/to/UNImodel.bin \
    --slide_ext .svs \
    --batch_size 128
```

## Acknowledgements
The code for **MPGraph** was adapted and inspired by many amazing repositories:[UNI](https://github.com/mahmoodlab/UNI), [CLAM](https://github.com/mahmoodlab/CLAM), and [PANTHER](https://github.com/mahmoodlab/PANTHER). We thank the authors and developers for their contributions.

## License
MPGraph is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.

## Citation
If you find our work useful in your research, please consider citing our paper:

```
```

