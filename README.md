3D Kernel Counting via SVRaster-Based Mask Lifting

This repository provides the official implementation of the pipeline described in the manuscript:

“3D Kernel Counting via SVRaster-Based Mask Lifting”

The method enables accurate three-dimensional (3D) kernel counting on maize cobs using a geometry-first reconstruction paradigm. A high-quality surface geometry is first reconstructed from RGB images using SVRaster, after which sparse two-dimensional (2D) kernel masks are projected onto the reconstructed geometry through post hoc mask lifting. Kernel instances are then separated using topology-aware clustering and counted automatically.

Unlike segmentation-driven radiance-field methods, mask information in this pipeline does not influence geometry reconstruction. Instead, mask morphology is used only during post-processing, allowing sparse supervision without destabilizing reconstruction.

Key Features

Geometry-first 3D reconstruction using SVRaster

Proportional dot-mask transformation to reduce kernel merging

Post hoc mask lifting onto reconstructed surfaces

Topology-aware kernel clustering and counting

Robust handling of densely packed kernel structures

Scripts to reproduce all experiments and figures in the manuscript
