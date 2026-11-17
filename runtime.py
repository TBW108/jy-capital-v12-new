"""Platform DAI or an explicit local Parquet directory; no bundled market data."""
import importlib
import logging
import os
from pathlib import Path
import re


class Logger:
    def info(self, message, **fields):
        logging.getLogger('v12_new').info('%s %s', message, fields)


def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return Logger()


class Result:
    def __init__(self, frame):
        self.frame = frame

    def df(self):
        return self.frame


class DataAccess:
    tables = {
        'cpt_jyc_2026_stock_bar1m': 'bar1m',
        'cpt_jyc_2026_instruments': 'sections',
        'cpt_jyc_2026_vwap': 'labels',
    }

    def query(self, sql, filters=None):
        root = os.environ.get('JY_DATA_DIR')
        if not root:
            try:
                platform = importlib.import_module('dai')
            except ImportError as exc:
                raise RuntimeError('Use the authorized BigQuant environment, or set JY_DATA_DIR to local Parquet exports. Do not install an unrelated PyPI dai package.') from exc
            return platform.query(sql, filters=filters)
        import pandas as pd
        import pyarrow as pa
        import pyarrow.dataset as ds
        match = re.fullmatch(r'SELECT\s+([\w,\s]+)\s+FROM\s+(\w+)', sql.strip(), re.I)
        if not match or match[2] not in self.tables:
            raise ValueError(f'Unsupported local query: {sql}')
        name = self.tables[match[2]]
        path = Path(root) / f'{name}.parquet'
        if not path.exists():
            path = Path(root) / name
        if not path.exists():
            raise FileNotFoundError(f'Missing {name}.parquet or {name}/ in {root}')
        dataset = ds.dataset(path, format='parquet')
        if not pa.types.is_timestamp(dataset.schema.field('date').type):
            raise TypeError('The date column must be a Parquet timestamp, not a string.')
        filters = filters or {}
        predicate = None
        if 'date' in filters:
            lo, hi = (pd.Timestamp(x).to_pydatetime() for x in filters['date'])
            predicate = (ds.field('date') >= lo) & (ds.field('date') <= hi)
        if 'instrument' in filters:
            p = ds.field('instrument').isin(list(filters['instrument']))
            predicate = p if predicate is None else predicate & p
        columns = [x.strip() for x in match[1].split(',')]
        return Result(dataset.to_table(columns=columns, filter=predicate).to_pandas())


dai = DataAccess()
