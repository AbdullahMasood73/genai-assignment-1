"""Inspect a frozen checkpoint on varied validation images only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from restoration.data import Pets
from restoration.train import load_checkpoint, model_from_checkpoint
from restoration.evaluate import draw_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=BASE/'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(4)
    saved = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(saved).to(device).eval()
    condition = saved['task'] if saved['task'] in ('salt','blur','occlusion') else None
    data = Pets(args.data, 'validation', specialist=condition)
    indices = np.random.default_rng(42).choice(len(data), 12, replace=False)
    rows = []
    examples = []
    with torch.no_grad():
        for i, index in enumerate(indices):
            image, target, label = data[int(index)]
            output = model(image[None].to(device))[0].cpu()
            row = data.rows[int(index)]
            examples.append((f"{row['id']} | {row['corruption']['kind']}",
                             (target,image,output,(output-target).abs())))
            rows.append({'id':row['id'],'corruption':row['corruption'],
                         'output_l1':float((output-target).abs().mean()),
                         'input_l1':float((image-target).abs().mean())})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    draw_grid(args.output, examples, f"Validation checkpoint, epoch {saved['epoch']+1}")
    args.output.with_suffix('.json').write_text(json.dumps({'split':'validation',
        'condition_filter':condition,
        'selected_epoch':saved['epoch']+1,'best_objective':saved['best'],
        'columns':['target','input','output','absolute error'],'examples':rows},indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
