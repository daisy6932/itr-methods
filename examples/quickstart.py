"""Smoke test for the public estimators with deliberately short training."""
import json
import sys
import numpy as np
from itr_methods import SCAF,NSCF

def data():
    rng=np.random.default_rng(20261022)
    X=rng.normal(size=(240,5));A=np.arange(240)%4
    y=.3*X[:,2]+np.where(A<2,1.,-1.)*(2*X[:,0]+X[:,1])+rng.normal(scale=.25,size=240)
    return X,A,y

def main():
    import torch
    torch.set_num_threads(1)
    X,A,y=data()
    # Two NSCF epochs exercise the interface without training to convergence.
    models={
      'SCAF':SCAF(init_lambda=25.,max_updates=2,random_state=7),
      'NSCF':NSCF(hidden_layer_sizes=(8,4),dropout=0.,pretrain_epochs=2,max_cycles=1,
                  num_lambdas=3,admm_epochs=2,theta_steps=1,refit_epochs=2,
                  patience=2,eval_every=1,random_state=7),
    }
    output={}
    for name,model in models.items():
        model.fit(X[:192],y[:192],treatment=A[:192])
        predictions=model.predict_outcomes(X[192:])
        output[name]=dict(shape=list(predictions.shape),treatments=model.treatments_.tolist(),
            groups=model.group_ids_.tolist(),recommended=model.predict(X[192:])[:8].tolist(),
            factual_mse=float(np.mean((model.predict_factual(X[192:],treatment=A[192:])-y[192:])**2)))
    print(
        "Smoke test only. The short NSCF fit's factual MSE checks prediction output; "
        "see docs/NSCF_TUTORIAL.ipynb for a trained example and baseline comparison.",
        file=sys.stderr,
    )
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
