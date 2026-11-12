"""Structured asymmetric response estimator used in the historical C1/C2 fits.

Inputs are aligned context rows, residual outcomes, action components and masks.
C1 supplies raw positive/negative actions; C2 supplies their residualized versions.
Temporal out-of-fold construction is the caller's responsibility.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
PRICE_CONTEXT_FEATURES = ['pc_log_price_per_oz_lag1', 'pc_relative_unit_price_lag1',
                          'pc_other_log_price_lag1', 'pc_other_price_change_lag1']

def store_design(stores, levels, frequencies):
    ix = pd.Index(levels).get_indexer(stores)
    x = np.zeros((len(ix), len(levels)))
    known = ix >= 0
    x[np.where(known)[0], ix[known]] = 1
    x[known] -= frequencies
    return x

def fit(h,u,ap,am,mask,skus,asym=True):
 k=len(skus);levels=sorted(h.store.unique());freq=h.store.value_counts(normalize=True).reindex(levels).to_numpy();stores=store_design(h.store,levels,freq);models=[];n=mask.sum()
 for i,s in enumerate(skus):
  cols=['week_sin','week_cos',f'sku_{s}_plan_B',f'sku_{s}_price_vs_reference']+[f'sku_{s}_{x}' for x in PRICE_CONTEXT_FEATURES]
  transform=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler());z=transform.fit_transform(h[cols])
  def design(a):return np.column_stack([a,a[:,i,None]*z,a[:,i,None]*stores])
  main=design(ap+am);delta=design(ap-am);a=np.column_stack([main,delta]) if asym else main
  pen=np.full(k,.1/(k*k));pen[i]=.001/(k*k);pen=np.r_[pen,np.full(len(cols),.5/(k*4)),np.full(len(levels),.5/(k*len(levels)))]*n
  penalty=np.r_[pen,pen] if asym else pen
  keep=mask[:,i].astype(bool);coef=np.linalg.solve(a[keep].T@a[keep]+np.diag(penalty),a[keep].T@u[keep,i]);models.append(dict(cols=cols,transform=transform,coef=coef,block=len(pen)))
 return dict(models=models,levels=levels,frequencies=freq,skus=skus,asym=asym)

def correction(model,h,d):
 k=len(model['skus']);stores=store_design(h.store,model['levels'],model['frequencies']);pred=[]
 for i,m in enumerate(model['models']):
  z=m['transform'].transform(h[m['cols']])
  def design(a):return np.column_stack([a,a[:,i,None]*z,a[:,i,None]*stores])
  a=np.column_stack([design(d),design(abs(d))]) if model['asym'] else design(d);pred.append(a@m['coef'])
 return np.column_stack(pred)

