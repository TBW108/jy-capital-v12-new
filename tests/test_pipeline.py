"""Small synthetic checks; no competition data or original weights are bundled."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import numpy as np
import pandas as pd
import train
from predict import predict


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architecture_is_causal(self):
        torch.manual_seed(42)
        model = train.RawFormer(**train.MODEL_CFG).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 244810)
        x = torch.randn(4, 220, 25)
        hist = torch.randn(4, 20, 240, 25)
        mask = torch.ones(4, 20, dtype=torch.bool)
        gid = torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            baseline = model(x, gid, 2, hist, mask)
            changed = x.clone()
            changed[:, 70:] = torch.randn_like(changed[:, 70:]) * 10
            altered = model(changed, gid, 2, hist, mask)
            prefix = model(x[:, :70], gid, 2, hist, mask)
        torch.testing.assert_close(baseline[:, :3], altered[:, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(baseline[:, :3], prefix, rtol=1e-5, atol=1e-6)

    def test_local_training_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'JY_DATA_DIR': ''}):
            os.environ['JY_DATA_DIR'] = folder
            root = Path(folder)
            rng = np.random.default_rng(42)
            bars, sections, labels = [], [], []
            for day in pd.bdate_range('2024-01-02', periods=3):
                minutes = list(pd.date_range(day + pd.Timedelta('9h16min'), periods=10, freq='min'))
                minutes += list(pd.date_range(day + pd.Timedelta('9h31min'), periods=120, freq='min'))
                minutes += list(pd.date_range(day + pd.Timedelta('13h1min'), periods=120, freq='min'))
                for i in range(32):
                    name = f'SYNTH{i:03d}'
                    price = 10 + i * 0.1 + rng.normal(0, .01, len(minutes)).cumsum()
                    row = {'date': minutes, 'instrument': name, 'adjust_factor': 1.0}
                    row.update({k: price for k in train.PRICE_COLS})
                    row.update({k: np.full(len(minutes), 1000.) for k in train.VOL_COLS + train.CNT_COLS})
                    row['amount'] = price * 1000
                    bars.append(pd.DataFrame(row))
                    for t in ['09:30','10:00','10:30','11:00','11:30','13:30','14:00','14:30']:
                        date = pd.Timestamp(f'{day.date()} {t}')
                        sections.append({'date':date, 'instrument':name})
                        labels.append({'date':date, 'instrument':name, 'vwap_return':rng.normal(0,.01), 'volume':float(rng.uniform(20000,40000))})
            pd.concat(bars).to_parquet(root/'bar1m.parquet', index=False)
            pd.DataFrame(sections).to_parquet(root/'sections.parquet', index=False)
            pd.DataFrame(labels).to_parquet(root/'labels.parquet', index=False)
            dates = ('2024-01-02', '2024-01-04')
            p = train.build_panel('cpt_jyc_2026_stock_bar1m', *dates, with_labels=True)
            y, m = train.gauss_rank(p['Y'], p['M'])
            aux = tuple(torch.from_numpy(a) for a in train.aux_target(p))
            with patch.object(train, 'BATCH_DAYS', 1):
                model = train.fit_one(train.to_tensors(p), torch.from_numpy(y), torch.from_numpy(m),
                                      train.MODEL_CFG, 42, torch.device('cpu'), steps=2, aux=aux)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            ckpt = {'format':train.FORMAT, 'feature_names':train.BAR_COLS,
                    'model_cfg':train.MODEL_CFG, 'state_dicts':[model.state_dict()], 'post':{'smooth':0.0}}
            path = root/'model.json'
            train.save_model(ckpt, path)
            loaded = train.build_models(train.load_model(path), torch.device('cpu'))
            p_unlabeled = train.build_panel('cpt_jyc_2026_stock_bar1m', *dates, with_labels=False)
            before = train.score_panel([model], p_unlabeled, ([], []), torch.device('cpu'))
            after = train.score_panel(loaded, p_unlabeled, ([], []), torch.device('cpu'))
            # JSON serialization uses five significant figures, so it is intentionally lossy.
            np.testing.assert_allclose(before, after, atol=.005, rtol=.005)
            (root/'labels.parquet').unlink()  # inference must succeed without any labels
            with patch('torch.cuda.is_available', return_value=False):
                result = predict({'bar1m':'cpt_jyc_2026_stock_bar1m'}, *dates, model_path=path)
            self.assertEqual(list(result.columns), ['date','instrument','factor'])
            self.assertEqual(len(result), len(sections))
            self.assertFalse(result.duplicated(['date','instrument']).any())
            self.assertTrue(np.isfinite(result.factor).all())


if __name__ == '__main__':
    unittest.main()
