"""Fit C1 or C2 from aligned, past-only prepared inputs (no nuisance fitting)."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from retail_pricing.response import fit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('context',type=Path,help='Numeric context CSV; rows align with NPZ arrays')
    p.add_argument('arrays',type=Path,help='NPZ: actions, residuals, mask; C2 also e_plus/e_minus')
    p.add_argument('--method',choices=['C1','C2'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-log-action',type=float,default=None)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    skus=json.loads((root/'configs/model_contract.json').read_text())['skus']
    h=pd.read_csv(args.context);a=np.load(args.arrays,allow_pickle=False)
    d=a['actions'];u=a['residuals'];mask=a['mask'].astype(bool)
    assert d.shape==u.shape==mask.shape==(len(h),len(skus))
    assert np.isfinite(d).all() and np.isfinite(u[mask]).all()
    assert (h.source_train_max<=h.week-2).all(), 'Input predictions must be strictly historical'
    keep=np.ones(len(h),bool) if args.max_log_action is None else np.max(abs(d),axis=1)<=args.max_log_action
    if not keep.any():raise ValueError('No retained training rows')
    plus,minus=np.maximum(d,0),np.minimum(d,0)
    if args.method=='C2':
        plus=plus-a['e_plus'];minus=minus-a['e_minus']
    model=fit(h.loc[keep].reset_index(drop=True),u[keep],plus[keep],minus[keep],mask[keep],skus)
    args.output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,args.output)
    print(f'Saved {args.method} response fitted on {keep.sum()} contexts')
if __name__=='__main__':main()
