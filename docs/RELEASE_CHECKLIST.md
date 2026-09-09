# Public-release checklist

Before pushing this package as the manuscript's official code repository:

- [ ] Verify the unscaled-mask value: supplied source script gives MAE = **80.22**, while the current manuscript draft says 80.00.
- [ ] Add the exact RGB SVRaster `config.yaml` used for the reported experiments.
- [ ] Add the trained four-image U-Net checkpoint.
- [ ] If exact U-Net continuation matters, add the previous three-image checkpoint or document its source.
- [ ] Add/release the nine-ear RGB dataset and annotation files, or replace the manuscript claim with a stable external data link.
- [ ] Add the original SD3 experiment implementation if SD3 remains in the manuscript comparison figure.
- [ ] Verify that all public paths/file names are free of internal/local identifiers.
- [ ] Choose a software license and add `LICENSE`.
- [ ] Update `CITATION.cff` with DOI/journal information after acceptance/publication.
- [ ] Run `python -m unittest discover -s tests` after `pip install -e .`.
- [ ] Run one end-to-end cob example from RGB masks through mask lifting and counting on a clean environment.
