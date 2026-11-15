# Spectral Interaction Modeling for Hyperspectral Image Classification

**Haoliang Meng¹, Miao Shang¹, Kaige Mao¹, Yabin Wang¹, and Xiaopeng Hong¹˒²**

¹ Faculty of Computing, Harbin Institute of Technology, Harbin, China  
² Pengcheng Laboratory, Shenzhen, China

*IEEE Transactions on Multimedia*, vol. 28, pp. 13518–13529, 2026. 

DOI: 10.1109/TMM.2026.3724350.

[Paper](https://ieeexplore.ieee.org/abstract/document/11656685/)

## Run

We use Python 3.10. The tested GPU environment is PyTorch 1.13.1+cu117 with CUDA 11.7.

### Datasets

Place each dataset's two MATLAB files in `datasets/<ID>/`, for example `datasets/ip/Indian_pines_corrected.mat`. Use `--data-root` to change the parent directory.

| ID | Dataset | Image file | Ground-truth file | Train / validation (%) |
|---|---|---|---|---|
| `ip` | Indian Pines | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` | 4 / 4 |
| `botswana` | Botswana | `Botswana.mat` | `Botswana_gt.mat` | 1 / 1 |
| `pu` | Pavia University | `PaviaU.mat` | `PaviaU_gt.mat` | 0.1 / 0.1 |
| `ksc` | Kennedy Space Center | `KSC_corrected.mat` | `KSC_gt.mat` | 1 / 1 |
| `sa` | Salinas | `Salinas_corrected.mat` | `Salinas_gt.mat` | 0.1 / 0.1 |
| `whuhh` | WHU-Hi-HongHu | `WHU_Hi_HongHu.mat` | `WHU_Hi_HongHu_gt.mat` | 0.08 / 0.08 |
| `whulk` | WHU-Hi-LongKou | `WHU_Hi_LongKou.mat` | `WHU_Hi_LongKou_gt.mat` | 0.03 / 0.03 |

To switch datasets, change `--dataset` to the corresponding ID. Expected MATLAB variable names are defined in `DATASETS` in [hs_trans/data.py](hs_trans/data.py).

To change the split, edit `train_ratio` and `val_ratio` in the selected `DATASETS` entry. For example, `train_ratio=0.10` and `val_ratio=0.05` allocate 10% of labeled pixels to training, 5% to validation, and the remaining 85% to testing.

### Training and hyperparameters

```bash
python train.py --dataset ip --data-root datasets --output runs/ip --runs 10 --seed 202401 --lr 0.001 --weight-decay 0.05
```

Change the argument values to configure training:

| Argument | Meaning / default |
|---|---|
| `--epochs` | Epochs per run |
| `--runs`, `--seed` | Number of consecutive seeds and first seed; defaults `10`, `202401` |
| `--batch-size` | Batch size |
| `--lr`, `--weight-decay` | AdamW settings; defaults `0.001`, `0.05` |
| `--shift-mode` | `cyclic` or `noncyclic`; affects SIGI only |
| `--device` | `auto` (default), `cpu`, or `cuda:0` |
| `--threads` | PyTorch intra-op CPU threads; default `4` |

Patch size is fixed at `8×8`. Model architecture settings are in [hs_trans/model.py](hs_trans/model.py).

Runs execute sequentially. Use a fresh `--output` directory for each experiment and launch one training command at a time. Each run saves its configuration, split, training history, the best checkpoint, and test metrics under `seed_<seed>/`. Training also evaluates that checkpoint on the test split and writes `summary.json` after all runs finish.

### Evaluation

```bash
python evaluate.py --checkpoint runs/ip/seed_202401/best.pt --data-root datasets
```

Results are saved beside the checkpoint in `evaluation/`. The model settings, split, and normalization are restored from the checkpoint. Add `--output-dir <directory>` to change the destination or `--full-map` to also save a whole-scene `prediction.npy` with 1-based class labels.

## Acknowledgments
We thank the repo of [GSC-ViT](https://github.com/flyzzie/TGRS-GSC-VIT) for providing helpful components, functions, and baselines in our work.

## Citation
If you find this work useful, please consider citing our paper:
```bibtex
@article{meng2026spectral,
  title={Spectral Interaction Modeling for Hyperspectral Image Classification},
  author={Meng, Haoliang and Shang, Miao and Mao, Kaige and Wang, Yabin and Hong, Xiaopeng},
  journal={IEEE Transactions on Multimedia},
  volume= {28},
  pages= {13518--13529},
  year={2026},
  publisher={IEEE},
  doi= {10.1109/TMM.2026.3724350}
}
```
## Contact
For questions about this work, please contact **Haoliang Meng** at [menghaoliang2002@163.com](mailto:menghaoliang2002@163.com).
