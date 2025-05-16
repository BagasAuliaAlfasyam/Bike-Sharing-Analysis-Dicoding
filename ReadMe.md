---

# Dicoding Dashboard

## Setup Environment

Follow these steps to set up the environment and run the Streamlit Dashboard:

### 1. Create and Activate Virtual Environment (Conda)

If you're using **Conda**, run the following commands in your terminal:

```sh
conda create --name py39 python=3.9
conda activate py39
```

### 2. Install Dependencies

Once the environment is active, install the required dependencies with:

```sh
pip install pandas numpy matplotlib seaborn streamlit plotly scikit-learn statsmodels
```

### 3. Run Streamlit

After all dependencies are installed, run Streamlit with the following command:

```sh
streamlit run Dashboard/AnalisisDataBike.py
```

## Troubleshooting

If you encounter a **ModuleNotFoundError**, make sure all dependencies are properly installed. If issues persist, try running:

```sh
pip install --upgrade pip
pip install -r requirements.txt  # If available
```

If there are other issues, check the error log and make sure you're in the correct environment by running:

```sh
conda info --envs
```

---
