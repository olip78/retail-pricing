import numpy as np
import pandas as pd
from retail_pricing.response import fit, correction, store_design, PRICE_CONTEXT_FEATURES

def context(n=400):
    h=pd.DataFrame({'store':np.tile([1,2],n//2),'week_sin':0.,'week_cos':1.})
    for sku in [10,20]:
        for name in ['plan_B','price_vs_reference']+PRICE_CONTEXT_FEATURES:
            h[f'sku_{sku}_{name}']=0.
    return h

def test_zero_action_is_anchor_and_unknown_store_is_neutral():
    h=context();rng=np.random.default_rng(7);d=rng.normal(size=(len(h),2))
    u=-2*d
    model=fit(h,u,np.maximum(d,0),np.minimum(d,0),np.ones_like(d,dtype=bool),[10,20])
    np.testing.assert_allclose(correction(model,h,np.zeros_like(d)),0,atol=1e-12)
    np.testing.assert_allclose(store_design([99],[1,2],np.array([.5,.5])),0)

def test_directional_response_learns_different_slopes():
    h=context();rng=np.random.default_rng(8);d=rng.uniform(-1,1,(len(h),2))
    u=-np.maximum(d,0)-3*np.minimum(d,0)
    model=fit(h,u,np.maximum(d,0),np.minimum(d,0),np.ones_like(d,dtype=bool),[10,20])
    pred=correction(model,h,d)
    assert np.sqrt(np.mean((pred-u)**2))<.03
    assert np.mean(pred[d>0])<0
    assert np.mean(pred[d<0])>0
