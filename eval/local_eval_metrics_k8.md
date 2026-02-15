# Local RAG Evaluation Report (Deterministic)

- Model: `llama3.1:8b`
- Retrieval top-k: `8`

## Summary

- **Avg citation coverage (bullets with [x])**: 66.67%
- **% answers with any citation**: 66.67%
- **% answers with `Answer:` heading**: 100.00%
- **% answers with `Cited sources:` heading**: 66.67%
- **% answers with valid bullet count (1-5)**: 100.00%
- **Avg bullet-end citation rate**: 66.67%
- **Avg valid citation ID rate (within top-k)**: 66.67%
- **Avg overlap (0–1)**: 0.6437
- **Avg weighted overlap (IDF, 0–1)**: 0.6300
- **Avg bullets per answer**: 3.60

## Per-question metrics

| id   |   num_bullets |   citation_coverage |   bullet_count_valid_1to5 |   bullet_end_citation_rate |   valid_citation_id_rate |   avg_overlap |   avg_weighted_overlap |   min_overlap |   min_weighted_overlap |
|:-----|--------------:|--------------------:|--------------------------:|---------------------------:|-------------------------:|--------------:|-----------------------:|--------------:|-----------------------:|
| Q01  |             5 |                   1 |                         1 |                          1 |                        1 |      0.963636 |               0.958191 |      0.818182 |               0.790956 |
| Q02  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.121368 |      0.2      |               0.121368 |
| Q03  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.121368 |      0.2      |               0.121368 |
| Q04  |             5 |                   1 |                         1 |                          1 |                        1 |      0.885727 |               0.882247 |      0.75     |               0.736881 |
| Q05  |             5 |                   1 |                         1 |                          1 |                        1 |      0.95     |               0.935376 |      0.75     |               0.676878 |
| Q06  |             5 |                   1 |                         1 |                          1 |                        1 |      0.759069 |               0.747191 |      0.666667 |               0.645925 |
| Q07  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.203884 |      0.2      |               0.203884 |
| Q08  |             5 |                   1 |                         1 |                          1 |                        1 |      0.967836 |               0.973123 |      0.894737 |               0.894023 |
| Q09  |             5 |                   1 |                         1 |                          1 |                        1 |      0.967251 |               0.969996 |      0.888889 |               0.917025 |
| Q10  |             4 |                   1 |                         1 |                          1 |                        1 |      0.905232 |               0.93789  |      0.833333 |               0.903315 |
| Q11  |             5 |                   1 |                         1 |                          1 |                        1 |      0.939732 |               0.951479 |      0.894737 |               0.923853 |
| Q12  |             5 |                   1 |                         1 |                          1 |                        1 |      0.566026 |               0.566987 |      0.466667 |               0.445287 |
| Q13  |             1 |                   0 |                         1 |                          0 |                        0 |      0.2      |               0.121368 |      0.2      |               0.121368 |
| Q14  |             1 |                   0 |                         1 |                          0 |                        0 |      0        |               0        |      0        |               0        |
| Q15  |             5 |                   1 |                         1 |                          1 |                        1 |      0.950859 |               0.959933 |      0.9      |               0.906845 |
