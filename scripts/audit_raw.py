"""Audit the selected SKUs from a separately obtained Dominick's movement file."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='wbjc.csv or a zip containing that CSV')
    parser.add_argument('--output', type=Path, default=ROOT/'generated/raw_audit')
    args = parser.parse_args()
    skus = json.loads((ROOT/'configs/model_contract.json').read_text())['skus']
    parts = []
    for chunk in pd.read_csv(args.input, chunksize=500000):
        chunk.columns = chunk.columns.str.lower()
        parts.append(chunk[chunk.upc.isin(skus)])
    raw = pd.concat(parts, ignore_index=True)
    if raw.duplicated(['store','upc','week']).any():
        raise ValueError('Duplicate SKU-store-week keys; inspect source before continuing')
    valid = raw[raw.ok.eq(1) & raw.move.ge(0)].copy().sort_values(['store','upc','week'])
    records = []
    for scope, frame in [('all',valid),('before_holdout',valid[valid.week.lt(348)])]:
        for (sku,store), group in frame.groupby(['upc','store']):
            q = group.move
            positive = q[q.gt(0)]
            delta = q.diff().where(group.week.diff().eq(1))
            records.append(dict(scope=scope,upc=sku,store=store,n=len(q),mean=q.mean(),variance=q.var(),
                cv=q.std()/q.mean() if q.mean() else np.nan,zero_share=q.eq(0).mean(),
                max_min=q.max()/q.min() if q.min()>0 else np.nan,
                max_min_positive=q.max()/positive.min() if len(positive) else np.nan,
                delta_std=delta.std(),span_coverage=len(q)/(group.week.max()-group.week.min()+1)))
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(args.output/'sku_store_statistics.csv',index=False)
    report = dict(selected_raw_rows=len(raw),valid_rows=len(valid),valid_zeros=int(valid.move.eq(0).sum()),
        stores=int(raw.store.nunique()),first_week=int(raw.week.min()),last_week=int(raw.week.max()))
    (args.output/'counts.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__ == '__main__':
    main()
