"""Inference only; missing weights never trigger training."""
import argparse
from pathlib import Path
import torch
import numpy as np
from train import (MODEL_PATH, build_models, build_panel, load_model, load_sections,
                   score_panel, scores_to_frame, limit_threads)


def predict(datasources, start_date, end_date, model_path=MODEL_PATH):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f'Train first: missing checkpoint {model_path}')
    limit_threads()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_model(model_path, map_location=device)
    models = build_models(ckpt, device)
    panel = build_panel(datasources['bar1m'], start_date, end_date, with_labels=False)
    post = ckpt.get('post', {})
    scores = score_panel(models, panel, (ckpt['ctx_mean'], ckpt['ctx_std']), device,
                         smooth=float(post.get('smooth', 0.0)))
    sections = load_sections(start_date, end_date)
    result = scores_to_frame(scores, panel, sections, 'factor')
    result = result.sort_values(['date', 'instrument']).reset_index(drop=True)
    if result.duplicated(['date', 'instrument']).any() or len(result) != len(sections):
        raise RuntimeError('Output keys do not match the section table')
    if not np.isfinite(result['factor']).all():
        raise RuntimeError('Non-finite factor')
    return result[['date', 'instrument', 'factor']]


def main(datasources, start_date, end_date):
    """Competition entry point, with the frozen checkpoint alongside train.py."""
    return predict(datasources, start_date, end_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--bar-table', default='cpt_jyc_2026_stock_bar1m')
    parser.add_argument('--output', default='outputs/factors.parquet')
    args = parser.parse_args()
    result = predict({'bar1m': args.bar_table}, args.start_date, args.end_date, args.model_path)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.csv':
        result.to_csv(path, index=False)
    else:
        result.to_parquet(path, index=False)
    print(f'Saved {len(result)} rows to {path}')
