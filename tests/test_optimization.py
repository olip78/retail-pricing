import numpy as np
from retail_pricing.optimization import optimize_one

def test_known_profit_optimum_and_other_prices_fixed():
    previous=np.array([[2.,3.]])
    cost=np.array([[1.,1.]])
    def demand(p):
        np.testing.assert_allclose(p[:,:,1],3.)
        q=np.empty_like(p);q[:,:,0]=10/p[:,:,0]**2;q[:,:,1]=2
        return q
    result=optimize_one(previous,cost,demand,0)
    # (p-1)*10/p^2 is maximized at p=2.
    np.testing.assert_allclose(result['price'],2.,atol=.01)
