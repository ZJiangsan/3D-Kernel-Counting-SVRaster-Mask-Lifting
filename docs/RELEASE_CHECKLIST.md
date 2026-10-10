# Public-release checklist

Before treating this package as the manuscript's final public code release:

- [x] Report the script-recorded unscaled-mask value consistently as **MAE = 80.22**.
- [x] Document the final **RGB-only SVRaster reconstruction → post-hoc mask lifting** workflow.
- [x] Provide the minimal RGB reconstruction override as `configs/svraster_rgb_reconstruction.yaml`.
- [ ] Add the trained four-image U-Net checkpoint if redistribution is appropriate.
- [ ] If exact U-Net continuation matters, document or release the previous checkpoint used for initialization.
- [ ] Add/release the nine-ear RGB dataset and annotation files, or provide a stable external data link if available.
- [x] Clarify SD3 reproducibility scope: Figure 2 retains SD3 as a reported secondary comparison, but the original SD3 development script is not part of this cleaned release. The repository therefore claims reproducibility for the proposed SVRaster pipeline and included ablations, not for every baseline comparison shown in the manuscript.
- [x] Remove internal/local checkpoint names and obsolete experimental-stage terminology from public documentation.
- [ ] Choose a software license and add `LICENSE`.
- [ ] Update `CITATION.cff` with DOI/journal information after acceptance/publication.
- [ ] Run `python -m unittest discover -s tests` after `pip install -e .`.
- [ ] Run one end-to-end ear example from RGB reconstruction through mask lifting and counting on a clean environment when the data can be shared.
