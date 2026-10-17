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
### Extract image patches from WSIs
Use [tiler_coords.py](./preprocess/tiler_coords.py) to extract tissue patch coordinates and saves them into HDF5 format, 
which following the [CLAM](https://github.com/mahmoodlab/CLAM) saving format for whole-slide image (WSI) processing.

Usage:
```python
python tiler_coords.py \
    --data_dir /path/to/your/wsi_files \
    --save_dir /path/to/output \
    --slide_ext .svs \
    --patch_size 256 \
    --overlap 0 \
    --patch_level 0 \
    --max_background 0.4
```

### Extract patches feature embeddings
1. Download the pretrained model weights [UNI](https://huggingface.co/MahmoodLab/UNI), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [CONCH](https://huggingface.co/MahmoodLab/CONCH), put it to yours *./ckpt_path* .
2. Use [extract_features.py](./preprocess/extract_features.py) to extract patches embeddings



## Acknowledgements
The code for **MPGraph** was adapted and inspired by many amazing repositories:[UNI](https://github.com/mahmoodlab/UNI), [CLAM](https://github.com/mahmoodlab/CLAM), and [PANTHER](https://github.com/mahmoodlab/PANTHER). We thank the authors and developers for their contributions.

## License
MPGraph is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.

## Citation
If you find our work useful in your research, please consider citing our paper:

```
```

