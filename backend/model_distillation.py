'''
Model Distillation - Logan Kelsch 2/20/2025 - Chick Counting

This file will be used for the implemenation of a distillation process (gradient tape through TF).
This file will only be used under a single possible scenario:
-   YOLO does not pan out for our project (cannot be trained well on abstract objects)
-   A custom box counting model is designed, specifically one with excessive computation cost.
-   The best working model needs greater generalization &| improved speed-matter &| space-matter.
'''
