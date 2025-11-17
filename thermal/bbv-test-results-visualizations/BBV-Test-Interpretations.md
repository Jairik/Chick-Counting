# Bounding Box Validation Test Results - Grid Search

Contains the results of the large GridSearchCV test in ```thermal_bbv_testing.ipynb```. 

---

## 3D Error Surfaces

![3D Error Surfaces Plot](3d_error_surfaces.png)

This plot measure relationships between the *confidence threshold* (x-axis), the *IOU treshold* (y-axis), and the *mean absolute counting error* (z-axis) (Lower = better and less error).

Each plot represents a different min_box_area parameters. From these plots, we can conclude that *min_box_area = 500* does not filter out enough boxes, with high and unstable error, *min_box_area = 700* is slightly better but is still noisy and inconsistent, and *min_box_area = 900* showcases massive improvement with error collapsing to a good region when confidence is around .4-.5 and IOU is around .4-.45

## Best Parameter Summary

![Best Parameter Summary Plot](best_parameters_summary.png)

This extremely informative graph showcases the minimal error from each set of parameters. From this visualiation, we cna see that **conf=.45** produces the lowest mean error, **IOU=.5** showcases the lowest mean error, and **min_box_area = 1100** produced (by far) the lowest mean error.

The **min_box_area** was a very impactful parameter for error. On the other than, conf and IOU showcases very little changes when adjusting. However, since both optimized parameter sets are on the higher-end.

The results also point to a potential for the IOU and confidence thresholds to be further increased to potentially decrease the mean error. This could be further explored, especially since increasing conf will lead to less results while increase IOU will lead to less results (likely avoiding over-generalizing, however an appropriate combination of these would need to be accomplished.)

## Detailed Distributions

![Detailed Distributions Plot](detailed_distributions.png)

This larger plot is broken up into four major sub-plots

### Error Distribution by Confidence Threshold (Top-Left)

This violin plot shows the distribution of mean errors by confidence threshold. As we can see, all distributions are fairly similar.

### Error Distribution by IOU Threshold (Top-Right)

This violin plot shows the distribution of mean errors by IOU threshold. Similarly to confidence thresholds, all distributions are fairly simular.

## Error Distribution by Min Box Area (Bottom-Left)

This violin plot shows the distribution of mean errors by minimum box area thresholds. These areas have a substantial impact on error distribution, with areas > 1100 likely significantly undercounting and areas <1100 likely overcounting. However, the minimum box area of *1100** has significantly lower mean errors, having a small range but low minimums. The range is from around 4-6, demonstrating that very low areas can be derived from this minimum box size. In conclusion, this is the paramemter with the biggest impact, with a clear optimum being found at 1100.

## Parameter Correlation Matrix (Bottom-Right)

This measures the pearson correlation between the confidence, iou, min_box_area, the predicted bounding box count, and error. From this, we can conclude that:

- conf and error have essentially no relationship (-.006)
- IOU and error have essentially no relationship (-.002)
- min_box_area and error have a weak positive correlation (.283). This is likely due to the non-linear relationship it has, however showing an indication that this is a substantial parameter to aid in determining true counts.
- min_box_area and bbv count has a strong negative correlation. This makes sense, as bigger minimum box thresholds will lead to fewer detections.
- bbv_count and error has a semi-substantial negative correlation (-.605). This relationship being negative points to the idea that **(within this search), the model is usually undercounting**.

## Parameter Performance Overview

![Parameter Performance Overview Plot](parameter_performance_overview.png)

This plot is broken into six subplots:

### Distribution of Prediction Errors (Top-Left)

This histogram shows the absolute counting errors accross all combinations. There are clear sections of models with errors below 10 (likely the 1100 min box size), indicating that there **are** optimal parameters to decrease error.

### Mean Error: Confidence vs IOU (Top-Middle)

This heatmap shows every error for every confidence/iou pair. The overall range/distribution is low, but showcases a **clear optimal** at the higher IOUs and lower confidences. This is very significant when making micro-optimizations through these parameters.

### Error Distribution by Confidence Threshold (Top-Right) and Error Distribution by IOU Threshold (Bottom-Left)

These boxplots don't really show too much, as there are no clear relationships among these parameters and error.

### Error Distribution by Min Box Area (Bottom-Middle)

This boxplot, similarly to other plots, shows the significance of minimum box area thresholds. Specifically, a size of 1100 is optimal.

### Predicted vs True Count (Bottom-Right)

This scatterplot shows different predicted counts (colored by error) when compared to true counts. From this, we can see that there are clusters of predictions near the true count. However, these predictions are overcounting (estimating around 250 when really 245). This indicates that significant parameters like min box area can likely be slightly **increased** in future iterations to attempt to decrease this count closer to the true count. Additionally, the impact of this parameter may warrant further investigations.

## Parameter Trends

![Parameter Trends Plot](parameter_trends.png)

This set of subplots indicate the mean error with the different parameters. Althought the IOU and confidence only has minor impacts on mean error, we can see clear dips when **confidence is .45**. This indicates a clear dominate value for this parameter. All other observations have been previously obtained from other plots.

## Textual Statistical Summaries

**Overall Performance:**

| Metric | Value |
|--------|-------|
| Total combinations tested | 300 |
| Mean error | 26.14 |
| Median error | 24.50 |
| Std deviation | 16.60 |
| Min error | 5.00 |
| Max error | 59.00 |

## Top Parameter Set Performers

| conf | iou | min_box_area | bbv_count | error |
|------|-----|--------------|-----------|-------|
| 0.45 | 0.35 | 1100 | 250 | 5 |
| 0.45 | 0.40 | 1100 | 250 | 5 |
| 0.45 | 0.45 | 1100 | 250 | 5 |
| 0.45 | 0.50 | 1100 | 250 | 5 |
| 0.45 | 0.55 | 1100 | 250 | 5 |
| 0.05 | 0.35 | 1100 | 251 | 6 |
| 0.05 | 0.50 | 1100 | 251 | 6 |
| 0.05 | 0.55 | 1100 | 251 | 6 |
| 0.10 | 0.35 | 1100 | 251 | 6 |

## Best Parameters

| Parameter | Best Value | Mean Error |
|-----------|------------|------------|
| Confidence Threshold | 0.45 | 25.67 |
| IOU Threshold | 0.5 | 26.08 |
| Min Box Area | 1100 | 6.08 |
