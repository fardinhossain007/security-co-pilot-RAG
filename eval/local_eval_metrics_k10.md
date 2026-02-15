# Local RAG Evaluation Report (Deterministic)

- Model: `llama3.1:8b`
- Retrieval top-k: `10`

## Summary

- **Avg citation coverage (bullets with [x])**: 73.33%
- **% answers with any citation**: 73.33%
- **% answers with `Answer:` heading**: 100.00%
- **% answers with `Cited sources:` heading**: 73.33%
- **% answers with valid bullet count (1-5)**: 100.00%
- **Avg bullet-end citation rate**: 73.33%
- **Avg valid citation ID rate (within top-k)**: 73.33%
- **Avg overlap (0–1)**: 0.7296
- **Avg weighted overlap (IDF, 0–1)**: 0.7204
- **Avg bullets per answer**: 3.87

## Per-question metrics

| id   |   num_bullets |   citation_coverage |   bullet_count_valid_1to5 |   bullet_end_citation_rate |   valid_citation_id_rate |   avg_overlap |   avg_weighted_overlap |   min_overlap |   min_weighted_overlap |
|:-----|--------------:|--------------------:|--------------------------:|---------------------------:|-------------------------:|--------------:|-----------------------:|--------------:|-----------------------:|
| Q01  |             5 |                   1 |                         1 |                          1 |                        1 |      0.963636 |               0.956323 |      0.818182 |               0.781617 |
| Q02  |             5 |                   1 |                         1 |                          1 |                        1 |      0.953918 |               0.96615  |      0.888889 |               0.940966 |
| Q03  |             1 |                   0 |                         1 |                          0 |                        0 |      0.4      |               0.381074 |      0.4      |               0.381074 |
| Q04  |             1 |                   0 |                         1 |                          0 |                        0 |      0        |               0        |      0        |               0        |
| Q05  |             5 |                   1 |                         1 |                          1 |                        1 |      0.967532 |               0.970498 |      0.909091 |               0.903919 |
| Q06  |             5 |                   1 |                         1 |                          1 |                        1 |      0.975    |               0.971802 |      0.875    |               0.859012 |
| Q07  |             5 |                   1 |                         1 |                          1 |                        1 |      0.690917 |               0.698694 |      0.4375   |               0.426418 |
| Q08  |             5 |                   1 |                         1 |                          1 |                        1 |      0.992593 |               0.990309 |      0.962963 |               0.951545 |
| Q09  |             5 |                   1 |                         1 |                          1 |                        1 |      0.951754 |               0.953864 |      0.842105 |               0.882468 |
| Q10  |             4 |                   1 |                         1 |                          1 |                        1 |      0.844697 |               0.854355 |      0.545455 |               0.520934 |
| Q11  |             5 |                   1 |                         1 |                          1 |                        1 |      0.927474 |               0.924947 |      0.782609 |               0.809284 |
| Q12  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.128049 |      0.2      |               0.128049 |
| Q13  |             5 |                   1 |                         1 |                          1 |                        1 |      0.88745  |               0.895392 |      0.764706 |               0.784366 |
| Q14  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.128049 |      0.2      |               0.128049 |
| Q15  |             5 |                   1 |                         1 |                          1 |                        1 |      0.989474 |               0.986366 |      0.947368 |               0.931828 |
