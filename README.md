# itmo-nn-compression

| name                           | metrics/precisionB | metrics/recallB | sparsity | postprocess |  size | metrics/mAP50B | metrics/mAP50-95B | fitness | inference | preprocess |
|:-------------------------------|-------------------:|----------------:|---------:|------------:|------:|---------------:|------------------:|--------:|----------:|-----------:|
| yolo tflight fp32              |               0.64 |            0.54 |     None |        2.21 | 12.08 |           0.61 |              0.45 |    0.46 |     77.69 |       1.91 |
| yolo tflight fp16              |               0.64 |            0.54 |     None |        2.39 | 12.02 |           0.61 |              0.45 |    0.46 |     82.04 |       1.51 |
| yolo tflight int8              |               0.64 |            0.54 |     None |        2.17 | 12.02 |           0.61 |              0.45 |    0.46 |     72.02 |       1.37 |
| yolo prune 0.05 params         |               0.65 |            0.52 |     5.00 |        3.68 | 12.08 |           0.61 |              0.45 |    0.46 |      8.42 |       1.03 |
| yolo prune 0.1 params          |               0.55 |            0.41 |    10.00 |        9.95 | 12.08 |           0.46 |              0.32 |    0.33 |      5.03 |       1.05 |
| yolo prune 0.15 params         |               0.49 |            0.46 |    15.00 |        7.18 | 12.08 |           0.46 |              0.32 |    0.33 |      9.36 |       0.92 |
| yolo prune 0.2 params          |               0.32 |            0.29 |    20.00 |        6.93 | 12.08 |           0.25 |              0.15 |    0.16 |      4.79 |       0.97 |
| yolo prune 0.25 params         |               0.45 |            0.11 |    25.00 |       13.35 | 12.08 |           0.12 |              0.07 |    0.07 |      4.95 |       1.03 |
| yolo prune 0.3 params          |               0.34 |            0.06 |    30.00 |        1.45 | 12.08 |           0.03 |              0.02 |    0.02 |     84.10 |      11.03 |
| yolo prune 0.4 params          |               0.43 |            0.01 |    40.00 |        4.88 | 12.08 |           0.01 |              0.00 |    0.00 |      4.90 |       1.42 |
| yolo run ultralytics           |               0.64 |            0.54 |     None |        2.21 | 12.08 |           0.61 |              0.45 |    0.46 |      8.62 |       1.01 |
| yolo distil params             |               0.64 |            0.54 |     None |        2.86 |  6.04 |           0.61 |              0.45 |    0.46 |     24.52 |       2.14 |
| yolo openvino                  |               0.65 |            0.53 |     None |        4.92 | 12.34 |           0.61 |              0.45 |    0.47 |     78.59 |       1.38 |
| yolo tflite                    |               0.65 |            0.53 |     None |        1.72 | 12.18 |           0.61 |              0.45 |    0.47 |    182.46 |       0.97 |
| yolo torchscript               |               0.65 |            0.53 |     None |        2.18 | 12.42 |           0.61 |              0.45 |    0.47 |    121.41 |       2.26 |
| yolo onnx                      |               0.65 |            0.53 |     None |        6.01 | 12.23 |           0.61 |              0.45 |    0.47 |     97.28 |       2.49 |
| ultralytics yolo distil params |               0.76 |            0.50 |     None |        4.93 |  6.04 |           0.60 |              0.43 |    0.44 |     15.96 |       9.12 |

Params:
- cluster_centroids_init -- KMEANS_PLUS_PLUS
- number_of_clusters - 16

| name                           | accuracy |
|:-------------------------------|---------:|
| base resnet                    |     0.21 |
| clustered resnet               |     0.10 |
