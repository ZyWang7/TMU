# TMU: Token Merge with Unmerge

**Code repository for the MSc thesis:** TMU: Token Merging with Unmerge for Efficient Hand Pose Estimation

![TMU](assets/TMU.png)

TMU is a training-free, drop-in technique for ViT-based 3D mesh regression. It plugs into existing models to improve runtime while maintaining the pose accuracy.


## At a glance

- **[HaMeR](https://github.com/geopavlakos/hamer):** ~**25–40%** faster with **negligible to moderate** accuracy drop; vertex errors remain the lowest across reduction levels.
- **[WiLoR](https://github.com/rolpotamias/WiLoR):** Preserves anatomical plausibility; shows some **sensitivity to small camera biases** introduced by in-block smoothing.
- **[SMPLest-X](https://github.com/SMPLCap/SMPLest-X):** ~**15–19%** faster with **negligible to moderate** degradation; retains lowest vertex error across reductions.


## Getting started

1. Model integrations: [HaMeR_TMU](https://github.com/ZyWang7/TMU/tree/main/HaMeR_TMU), [WiLoR_TMU](https://github.com/ZyWang7/TMU/tree/main/WiLoR_TMU), or [SMPLest-X_TMU](https://github.com/ZyWang7/TMU/tree/main/SMPLest-X_TMU). **Check the README inside that subfolder** for environment setup, data/checkpoint preparation, and any model-specific notes.
2. Adjust the token-reduction level to hit your latency/accuracy target.

