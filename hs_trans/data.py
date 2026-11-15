from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
DATASETS = {'ip': dict(image_file='Indian_pines_corrected.mat', image_key='indian_pines_corrected', gt_file='Indian_pines_gt.mat', gt_key='indian_pines_gt', bands=200, classes=16, train_ratio=0.04, val_ratio=0.04), 'botswana': dict(image_file='Botswana.mat', image_key='Botswana', gt_file='Botswana_gt.mat', gt_key='Botswana_gt', bands=145, classes=14, train_ratio=0.01, val_ratio=0.01), 'pu': dict(image_file='PaviaU.mat', image_key='paviaU', gt_file='PaviaU_gt.mat', gt_key='paviaU_gt', bands=103, classes=9, train_ratio=0.001, val_ratio=0.001), 'ksc': dict(image_file='KSC_corrected.mat', image_key='KSC', gt_file='KSC_gt.mat', gt_key='KSC_gt', bands=176, classes=13, train_ratio=0.01, val_ratio=0.01), 'sa': dict(image_file='Salinas_corrected.mat', image_key='salinas_corrected', gt_file='Salinas_gt.mat', gt_key='salinas_gt', bands=204, classes=16, train_ratio=0.001, val_ratio=0.001), 'whuhh': dict(image_file='WHU_Hi_HongHu.mat', image_key='WHU_Hi_HongHu', gt_file='WHU_Hi_HongHu_gt.mat', gt_key='WHU_Hi_HongHu_gt', bands=270, classes=22, train_ratio=0.0008, val_ratio=0.0008), 'whulk': dict(image_file='WHU_Hi_LongKou.mat', image_key='WHU_Hi_LongKou', gt_file='WHU_Hi_LongKou_gt.mat', gt_key='WHU_Hi_LongKou_gt', bands=270, classes=9, train_ratio=0.0003, val_ratio=0.0003)}

def _file_case_insensitive(directory, filename):
    expected = directory / filename
    if expected.is_file():
        return expected
    if directory.is_dir():
        matches = [p for p in directory.iterdir() if p.is_file() and p.name.casefold() == filename.casefold()]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f'Required dataset file is missing: {expected}')

def load_dataset(name, root):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    spec = DATASETS[name]
    directory = Path(root) / name
    image_path = _file_case_insensitive(directory, spec['image_file'])
    gt_path = _file_case_insensitive(directory, spec['gt_file'])
    image_mat = loadmat(image_path, variable_names=[spec['image_key']])
    gt_mat = loadmat(gt_path, variable_names=[spec['gt_key']])
    if spec['image_key'] not in image_mat:
        raise ValueError(f"{image_path} has no variable {spec['image_key']!r}")
    if spec['gt_key'] not in gt_mat:
        raise ValueError(f"{gt_path} has no variable {spec['gt_key']!r}")
    cube = np.asarray(image_mat[spec['image_key']], dtype=np.float32)
    gt = np.asarray(gt_mat[spec['gt_key']])
    if cube.ndim != 3 or cube.shape[-1] != spec['bands']:
        raise ValueError(f"{name}: expected HWC cube with {spec['bands']} bands, got {cube.shape}")
    if gt.ndim != 2 or cube.shape[:2] != gt.shape:
        raise ValueError(f'{name}: cube spatial shape {cube.shape[:2]} and labels {gt.shape} differ')
    if not np.isfinite(cube).all() or not np.isfinite(gt).all():
        raise ValueError(f'{name}: data or labels contain NaN/Inf')
    if not np.equal(gt, np.floor(gt)).all():
        raise ValueError(f'{name}: labels must be integers')
    expected_labels = np.arange(spec['classes'] + 1)
    if not np.array_equal(np.unique(gt), expected_labels):
        raise ValueError(f"{name}: expected background 0 and all classes 1..{spec['classes']}")
    return (cube, gt.astype(np.int64) - 1)

def normalize(cube, stats=None):
    image = np.array(cube, dtype=np.float32, copy=True)
    if image.ndim != 3 or not image.size or (not np.isfinite(image).all()):
        raise ValueError('Normalization requires a nonempty finite HWC cube')
    if stats is None:
        minimum = float(image.min())
        maximum = float(image.max())
        mean = None
    else:
        minimum = float(stats['minimum'])
        maximum = float(stats['maximum'])
        mean = np.asarray(stats['band_mean'], dtype=np.float32)
        if mean.shape != (image.shape[-1],) or not np.isfinite(mean).all():
            raise ValueError('Stored normalization band means do not match the cube')
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError('Normalization maximum must be finite and greater than minimum')
    image -= np.float32(minimum)
    image /= np.float32(maximum - minimum)
    if mean is None:
        mean = image.mean(axis=(0, 1))
    image -= mean
    return (image, {'minimum': minimum, 'maximum': maximum, 'band_mean': mean.tolist()})

def split_indices(gt, train_ratio, val_ratio, seed):
    if not np.isfinite([train_ratio, val_ratio]).all() or train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio >= 1):
        raise ValueError('Train/validation ratios must be positive and sum to less than 1')
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.isfinite(gt).all() or (not np.equal(gt, np.floor(gt)).all()):
        raise ValueError('Ground truth must be a finite HW integer array')
    coordinates = np.argwhere(gt >= 0).astype(np.int64)
    if not len(coordinates):
        raise ValueError('No labeled pixels are available for splitting')
    labels = gt[tuple(coordinates.T)]
    total_ratio = train_ratio + val_ratio
    try:
        (trainval, test) = train_test_split(coordinates, train_size=total_ratio, random_state=seed, stratify=labels)
        trainval = trainval[np.lexsort((trainval[:, 1], trainval[:, 0]))]
        (train, val) = train_test_split(trainval, train_size=train_ratio / total_ratio, random_state=seed, stratify=gt[tuple(trainval.T)])
    except ValueError as error:
        raise ValueError(f'The stratified train/validation/test split is impossible at these ratios and class counts: {error}') from error
    return {name: values[np.lexsort((values[:, 1], values[:, 0]))] for (name, values) in (('train', train), ('val', val), ('test', test))}

class PatchDataset(Dataset):

    def __init__(self, cube, gt, coordinates, patch_size=8, augment=False, padded=None):
        cube = np.asarray(cube)
        gt = np.asarray(gt)
        coordinates = np.asarray(coordinates)
        if not isinstance(patch_size, (int, np.integer)) or isinstance(patch_size, bool) or patch_size < 1:
            raise ValueError('patch_size must be a positive integer')
        if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
            raise ValueError('Patches require an HWC cube and spatially aligned HW labels')
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError('Coordinates must have shape (N, 2)')
        if not np.issubdtype(coordinates.dtype, np.integer):
            raise ValueError('Coordinates must be integer row/column indices')
        if len(coordinates) and ((coordinates < 0).any() or (coordinates >= np.array(gt.shape)).any()):
            raise ValueError('Patch coordinates lie outside the scene')
        self.coordinates = coordinates.astype(np.int64, copy=False)
        self.labels = gt[tuple(self.coordinates.T)].astype(np.int64)
        if (self.labels < 0).any():
            raise ValueError('Patch coordinates include unlabeled pixels')
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        pad = self.patch_size // 2
        expected_shape = (cube.shape[0] + 2 * pad, cube.shape[1] + 2 * pad, cube.shape[2])
        if padded is None:
            self.data = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        else:
            if padded.shape != expected_shape:
                raise ValueError('Shared padded cube has an incompatible shape')
            self.data = padded

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        (row, column) = self.coordinates[index]
        patch = self.data[row:row + self.patch_size, column:column + self.patch_size]
        if self.augment and np.random.random() > 0.5:
            probability = np.random.random()
            if probability <= 0.2:
                patch = np.fliplr(patch)
            elif probability <= 0.4:
                patch = np.flipud(patch)
            elif probability <= 0.6:
                patch = np.rot90(patch, k=1)
            elif probability <= 0.8:
                patch = np.rot90(patch, k=2)
            else:
                patch = np.rot90(patch, k=3)
        tensor = torch.from_numpy(np.array(patch.transpose(2, 0, 1), dtype=np.float32, copy=True))
        return (tensor, torch.tensor(self.labels[index], dtype=torch.long))
