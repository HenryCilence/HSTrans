import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import DATASETS, PatchDataset, load_dataset, normalize, split_indices
from .model import HS_Trans

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def environment():
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def device_for(name):
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu, cuda:N, or auto')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA was requested but is unavailable')
    return device

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def fingerprint(cube, gt):
    digest = hashlib.sha256()
    for array in (cube, gt):
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode('ascii'))
        digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()

def classification_metrics(target, prediction, classes):
    (target, prediction) = (np.asarray(target), np.asarray(prediction))
    if not len(target) or target.shape != prediction.shape:
        raise ValueError('Metrics require nonempty aligned predictions and targets')
    if ((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)).any():
        raise ValueError('Class indices are outside the configured range')
    matrix = np.bincount(target * classes + prediction, minlength=classes ** 2).reshape(classes, classes)
    counts = matrix.sum(1)
    if (counts == 0).any():
        raise ValueError('Every class must appear in the evaluation split for AA')
    recalls = matrix.diagonal() / counts
    oa = matrix.trace() / matrix.sum()
    expected = counts.astype(np.float64) @ matrix.sum(0) / matrix.sum() ** 2
    kappa = (oa - expected) / (1 - expected) if expected < 1 else 1.0
    return dict(oa_percent=float(100 * oa), aa_percent=float(100 * recalls.mean()), kappa=float(kappa), per_class_accuracy_percent=(100 * recalls).tolist(), samples=int(matrix.sum()), confusion_matrix=matrix.tolist())

@torch.inference_mode()
def predict(model, dataset, device, batch_size):
    model.eval()
    (predictions, targets) = ([], [])
    for (inputs, labels) in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        predictions.append(model(inputs.to(device)).argmax(1).cpu().numpy())
        targets.append(labels.numpy())
    if not predictions:
        raise ValueError('Cannot evaluate an empty dataset')
    return (np.concatenate(predictions), np.concatenate(targets))

def prepare_data(dataset, data_root):
    (cube, gt) = load_dataset(dataset, data_root)
    signature = fingerprint(cube, gt)
    (image, stats) = normalize(cube)
    return (image, gt, stats, signature)

def train_run(config, image, gt, stats, signature, output, seed):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Run directory is not empty; choose a new --output: {output}')
    if config['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    device = device_for(config['device'])
    spec = DATASETS[config['dataset']]
    split = split_indices(gt, spec['train_ratio'], spec['val_ratio'], seed)
    np.savez_compressed(output / 'split.npz', **split)
    model_args = dict(in_bands=spec['bands'], num_classes=spec['classes'], patch_size=8, variant=config['variant'], shift_mode=config['shift_mode'], components=config.get('components', 'igi+sigi+cgi'))
    model = HS_Trans(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    train_set = PatchDataset(image, gt, split['train'], augment=True)
    val_set = PatchDataset(image, gt, split['val'], padded=train_set.data)
    loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    best = -1.0
    metadata = dict(schema_version=1, dataset=config['dataset'], dataset_sha256=signature, model_args=model_args, normalization=stats, seed=seed, config={k: v for (k, v) in config.items() if k != 'data_root'}, environment=environment())
    write_json(output / 'config.json', dict(config, seed=seed, model_args=model_args, dataset_sha256=signature, split_counts={k: len(v) for (k, v) in split.items()}, parameters=sum((p.numel() for p in model.parameters()))))
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        (loss_sum, count) = (0.0, 0)
        for (inputs, targets) in loader:
            (inputs, targets) = (inputs.to(device), targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            count += len(targets)
        (pred, truth) = predict(model, val_set, device, config['batch_size'])
        accuracy = float((pred == truth).mean())
        row = dict(epoch=epoch, train_loss=loss_sum / count, val_oa_percent=accuracy * 100)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if accuracy > best:
            best = accuracy
            checkpoint = dict(schema_version=1, metadata_json=json.dumps(dict(metadata, epoch=epoch, val_oa_percent=best * 100), allow_nan=False), split={k: torch.from_numpy(v.copy()) for (k, v) in split.items()}, model_state={k: v.detach().cpu() for (k, v) in model.state_dict().items()})
            torch.save(checkpoint, output / 'best.pt')
        if epoch == 1 or epoch % 10 == 0 or epoch == config['epochs']:
            print(f"seed={seed} epoch={epoch}/{config['epochs']} loss={row['train_loss']:.5f} val_OA={accuracy * 100:.2f}%", flush=True)
    del loader, train_set, val_set, optimizer, model
    result = evaluate_checkpoint(output / 'best.pt', None, config['device'], config['batch_size'], output, prepared=(image, gt, signature))
    return result

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Not a release checkpoint (schema_version=1 required)')
    checkpoint.update(json.loads(checkpoint.pop('metadata_json')))
    return checkpoint

def evaluate_checkpoint(checkpoint_path, data_root, device='auto', batch_size=128, output_dir=None, full_map=False, prepared=None):
    checkpoint = load_checkpoint(checkpoint_path)
    if prepared is None:
        (raw, gt) = load_dataset(checkpoint['dataset'], data_root)
        signature = fingerprint(raw, gt)
        (image, _) = normalize(raw, checkpoint['normalization'])
    else:
        (image, gt, signature) = prepared
    if signature != checkpoint['dataset_sha256']:
        raise ValueError('Dataset content differs from the dataset used for this checkpoint')
    device = device_for(device)
    model = HS_Trans(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    coordinates = checkpoint['split']['test'].numpy()
    test_set = PatchDataset(image, gt, coordinates, patch_size=checkpoint['model_args']['patch_size'])
    (predictions, targets) = predict(model, test_set, device, batch_size)
    result = classification_metrics(targets, predictions, DATASETS[checkpoint['dataset']]['classes'])
    result.update(dataset=checkpoint['dataset'], seed=checkpoint['seed'], best_epoch=checkpoint['epoch'], val_oa_percent=checkpoint['val_oa_percent'], model_args=checkpoint['model_args'], dataset_sha256=signature)
    if output_dir is not None:
        output_dir = Path(output_dir)
        write_json(output_dir / 'metrics.json', result)
        np.savez_compressed(output_dir / 'test_predictions.npz', coordinates=coordinates, predictions=predictions, targets=targets)
    if full_map:
        if output_dir is None:
            raise ValueError('full_map requires an output directory')
        all_coordinates = np.indices(gt.shape).reshape(2, -1).T
        map_set = PatchDataset(image, np.zeros_like(gt), all_coordinates, padded=test_set.data)
        (full_prediction, _) = predict(model, map_set, device, batch_size)
        np.save(output_dir / 'prediction.npy', full_prediction.reshape(gt.shape).astype(np.int16) + 1)
    print(f"test {checkpoint['dataset']} seed={checkpoint['seed']} OA={result['oa_percent']:.2f}% AA={result['aa_percent']:.2f}% kappa={result['kappa']:.4f}", flush=True)
    return result
