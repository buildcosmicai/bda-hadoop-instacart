# Big Data Analytics — Hadoop & Machine Learning on the Instacart Dataset

Source code accompanying the report *Advanced Big Data Processing through MapReduce
Abstraction and HDFS Architecture* (M.Sc. Big Data Analytics, 2026).

The project builds a Hadoop-based processing pipeline and a machine learning workflow
on the [Instacart Market Basket Analysis dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis),
using the `order_products__prior.csv` file (~32.4 million records, 550.8 MB). A
Hadoop Streaming MapReduce job computes product order frequencies, and a Random Forest
classifier predicts whether a product will be reordered.

## Repository structure

| File | Purpose |
|------|---------|
| `mapper.py` | Streaming Mapper — emits `(product_id, 1)` for each order record |
| `reducer.py` | Streaming Reducer — sums counts per product to give order totals |
| `ml_analysis.py` | Preprocessing, feature engineering, Random Forest training, evaluation, and the five result charts |
| `requirements.txt` | Python dependencies for `ml_analysis.py` |

## Running the MapReduce job

With the dataset already in HDFS (e.g. under `/retail`), the product-popularity job is
run via Hadoop Streaming:

```bash
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
  -input /retail/order_products__prior.csv \
  -output /retail/product_counts \
  -mapper mapper.py \
  -reducer reducer.py \
  -file mapper.py -file reducer.py
```

The job reads ~32.4 million records and produces order totals for 49,677 unique products.

## Running the machine learning script

```bash
pip install -r requirements.txt
python ml_analysis.py
```

Set the `ARCHIVE` variable at the top of `ml_analysis.py` to the folder holding
`order_products__prior.csv` and `products.csv`. The script trains the model, prints the
accuracy and classification report, and saves five PNG charts to the working directory.

## Key results

- **Most-ordered product:** Banana (472,565 orders), with fresh produce dominating demand.
- **Model accuracy:** 66.3% — strong recall on reordered products, weaker on
  non-reordered ones owing to class imbalance.
- **Strongest predictor:** a product's historical reorder rate.

## Notes

The dataset itself is not included in this repository (it is large and publicly
available at the Kaggle link above).

---
Hans Joseph · M.Sc. Big Data Analytics, 2026
