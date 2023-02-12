# OTTO – Multi-Objective Recommender System - Kaggle Competition
A 109th ranking solution + follow up improvements/explorations.

## 2-Stage-System

![image](https://user-images.githubusercontent.com/71299664/218287232-a8a68476-f1b5-4222-ad0e-e2add7c3ef1a.png)

### Stage 0 - Data Preprocessing
Pipelines for Transforming the primitive data provided by Kaggle host(in .json) to column-based format using PySpark.  
This enables more efficient memory-usage when computing a co-visitation matrix and making inferences in the later part.  
See [data_engineering](https://github.com/itong1900/otto_obj_recommendation_sys/tree/master/data_engineering) directory for more details. 


### Stage I - Item Colloborative Filtering
1. Time Weight Co-visitation matrix
2. Inverse User Frequency Co-visitation matrix  
See [item_collaborative_filter](https://github.com/itong1900/otto_obj_recommendation_sys/tree/master/item_collaborative_filter) directory for more details.

### Stage II - Reranking with LgbmRanker
1. Feature Engineering
2. Parameter Selection
See [training_pipelines](https://github.com/itong1900/otto_obj_recommendation_sys/tree/master/training_pipelines) for more details. 
