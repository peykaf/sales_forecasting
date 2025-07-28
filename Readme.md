To run the code, make sure `Python==3.12.7` installed. then:

```
python -m venv .venv
source .venv/bin/activate
```

To install the dependencies, run

```pip install -r requirements.txt```

Explanatory Data Analysis is carried out in `eda.ipynb` and `eda.py`.


To run the model end to end, consisting of:

- Featurizing the raw data
- Train and validation split
- Fit the model
- Predict the target
- Evaluation (metrics, feature importance)

run the following command from the root directory of the project:

```python training.py```


