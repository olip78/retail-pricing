# Data access

Source: [Dominick's, Kilts Center](https://www.chicagobooth.edu/research/kilts/research-data/dominicks).

We acknowledge the Kilts Center for Marketing at the University of Chicago Booth School of Business. The source states that the data are for academic research purposes. Obtain files from the provider and follow its terms.

This repository does not redistribute raw or row-level source data. Place a separately obtained BJC movement CSV or ZIP here to run the data audit. File schemas can be checked against the source manual. The selected product IDs are listed in `configs/model_contract.json`.

Our model excludes zero sales; the raw-data audit retains valid observed zeros. Missing records are never silently converted into zeros.
