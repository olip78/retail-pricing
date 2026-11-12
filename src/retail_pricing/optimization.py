"""One-product-at-a-time gross-profit sanity check; no policy validation."""
import numpy as np

def optimize_one(previous_prices, cost, predict_units, sku, lower_ratio=.7,
                 upper_ratio=1/.7, grid_size=73, sales_caps=None):
    """predict_units(prices) -> (contexts, candidates, products).

    All other prices stay at previous-week levels. Actual target-week prices are
    deliberately absent from this API. Returned scenarios are independent.
    """
    p=np.asarray(previous_prices,float);c=np.asarray(cost,float)
    if p.shape!=c.shape or p.ndim!=2 or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('Expected aligned positive finite price and cost matrices')
    lo=np.ceil(np.maximum(lower_ratio*p[:,sku],c[:,sku])*100-1e-8)/100
    hi=np.floor(upper_ratio*p[:,sku]*100+1e-8)/100
    valid=lo<=hi
    if not valid.all():
        raise ValueError('Infeasible cost/band intersection; remove and report these contexts')
    ratios=np.exp(np.linspace(np.log(lower_ratio),np.log(upper_ratio),grid_size))
    candidates=np.round(p[:,sku,None]*ratios,2)
    candidates=np.clip(candidates,lo[:,None],hi[:,None])
    candidates=np.column_stack([candidates,lo,hi,np.clip(p[:,sku],lo,hi)])
    prices=np.repeat(p[:,None,:],candidates.shape[1],axis=1)
    prices[:,:,sku]=candidates
    units=np.asarray(predict_units(prices),float)
    if units.shape!=prices.shape or not np.isfinite(units).all() or (units<0).any():
        raise ValueError('Invalid quantity predictions')
    if sales_caps is not None:units=np.minimum(units,np.asarray(sales_caps)[:,None,:])
    profit=((prices-c[:,None,:])*units).sum(axis=-1)
    ties=profit>=profit.max(axis=1)[:,None]-1e-8
    distance=abs(np.log(candidates/p[:,sku,None]))
    idx=np.where(ties,distance,np.inf).argmin(axis=1)
    rows=np.arange(len(p))
    return dict(price=candidates[rows,idx],profit=profit[rows,idx],units=units[rows,idx],lower=lo,upper=hi)
