"""Refit Direct on a separately prepared decision-time feature table."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('features',type=Path)
    parser.add_argument('--last-training-week',type=int,default=346)
    parser.add_argument('--max-log-action',type=float,default=None)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    contract=json.loads((root/'configs/model_contract.json').read_text())
    f=pd.read_csv(args.features);f=f[f.week.between(105,args.last_training_week)].copy()
    if args.max_log_action is not None:
        actions=f[[f'sku_{s}_action' for s in contract['skus']]].to_numpy()
        f=f[np.all(np.isfinite(actions)&(abs(actions)<=args.max_log_action),axis=1)].copy()
    f=f.sort_values(['week','store','upc'])
    columns=contract['direct_features'];x=f[['upc','store']+columns].copy()
    for c in ['upc','store']:x[c]=x[c].astype(str)
    x[columns]=x[columns].fillna(-999.)
    cfg=contract['config']
    model=CatBoostRegressor(**cfg['catboost'],random_seed=cfg['seed'],loss_function='RMSE',has_time=True,verbose=False,allow_writing_files=False)
    model.fit(x,f.log_move,cat_features=['upc','store'])
    args.output.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(args.output))
    print(f'Saved Direct trained on {len(f)} rows; verify external feature provenance before use')
if __name__=='__main__':main()
