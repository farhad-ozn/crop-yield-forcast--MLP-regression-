# Forecasting Yield of Crop Products using Multi Layer Perceptron

**Author:** Farhad Oghazian (April 2025)

---

## A. Preprocessing  

Preprocessing was a critical and time-intensive part of this project due to the size, resolution, and structure of the given datasets. Below is a description of preprocessing steps taken, along with their rationale and implementation strategy.  

### Loading and Merging Raw Data  
We began by loading 17 separate CSV files that included data on:  

- Crop yields and production (from FAO)  
- Monthly climate and soil features  
  - 4 CSV files for monthly SoilTMP  
  - 4 CSV files for SoilMoi (each related to a depth/level)  
  - CanopInt_inst_data, ESoil_tavg_data, Rainf_tavg_data, Snowf_tavg_data, TVeg_tavg_data, TWS_inst_data  
- Land cover data  
- Geospatial lookup table for country centroids  

Each dataset had its own format, resolution, and anomalies (e.g., Nulls, zeros) requiring alignment before merging.  

### Data Cleaning and Wrangling  
Steps included:  

- **Country lookup table:** checked for duplicates, Null values, and removed countries with 'centroid radius' = 0 (7 rows removed).  
- **Land cover:** only class 12 (Croplands) and class 14 (Cropland/Natural Vegetation Mosaics) retained.  
- **Climate & soil data:** averaged monthly columns to annual values.  
  - SoilTMP [Kelvin]  
  - SoilMoi [Kg/m²]  
  - Rainf_tavg [Kg/m²/s]  
  - TVeg_tavg [W/m²]  
  - ESoil_tavg [W/m²]  
  - CanopInt_inst [Kg/m²]  
  - TWS_inst [mm]  

These variables reflect ongoing conditions that impact crops regularly, so average yearly behavior was used.  

- **Yield and Production data:**  
  - Yield = Production / AreaHarvested.  
  - Production removed to avoid data leakage.  
  - Element column filtered to contain only yield.  
  - Flag filtering:  
    - `A` (Official), `E` (Estimated), `I` (Imputed), `X` (International) kept.  
    - `M` (Missing) removed.  
  - Dropped irrelevant columns (Domain, Element, Item code, Flag description).  

- **Country Assignment Based on Coordinates:**  
  - A custom function assigned each lat/lon to the nearest country centroid.  
  - Data grouped by **country-year** to match target yield level.  

- **Merging:**  
  - All cleaned datasets merged on `country` and `year` using outer join.  
  - Rows with null values in `Yield` and `Item` removed (3442 instances).  
  - Additional 7692 rows with missing climate/soil data removed.  
  - Outliers removed above the 99.97th percentile (e.g., extremely high yields in specific country-crop pairs).  

---

## B. Modeling  

We used a **3-layer Multilayer Perceptron (MLP)** for regression, implemented in **PyTorch**, to predict log-transformed crop yields.  

### Model Structure  

**Input → Hidden Layer 1 → Hidden Layer 2 → Output**  

- **Inputs:**  
  - Categorical:  
    - `country` (163 categories → 50-dim embedding)  
    - `Item` (102 categories → 50-dim embedding)  
  - Numerical: 8 features (e.g., Rainf_mean, SoilMoi, SoilTMP, etc.).  

- **Hidden layers:**  
  - Layer 1: 256 neurons  
  - Layer 2: 256 neurons  
  - Activation: LeakyReLU (α = 0.01)  

- **Output layer:**  
  - Single neuron for regression (`log(1 + Yield)`)  

### Training  

- Optimizer: **Adam** (with weight decay for L2 regularization)  
- Loss: MSE on log-transformed yield  
- SGD tested but less stable and more sensitive to hyperparameters.  

---

## C. Features & Labels  

- **Output variable (label):**  
  - `yield_log = log(1 + Yield)`  
  - After prediction, transformed back using `exp(yield_log) − 1` for interpretability.  

- **Categorical Features:**  
  - Country (163 categories)  
  - Item (102 categories)  
  - Encoded using **PyTorch embeddings** instead of one-hot (avoiding sparsity and improving performance).  

- **Numerical Features (8):**  

| Feature                  | Description |
|---------------------------|-------------|
| Land_cover_percent_mean  | % cropland (Class 12 + 14, MODIS) |
| CanopInt_mean            | Avg water on plant surface (kg/m²) |
| Rainf_mean               | Avg rainfall rate (kg/m²/s) |
| ESoil_mean               | Soil evaporation (W/m²) |
| Snowf_mean               | Snowfall rate (kg/m²/s) |
| TWS_mean                 | Terrestrial water storage (mm) |
| SoilMoi                  | Avg soil moisture across 4 depths (kg/m²) |
| SoilTMP                  | Avg soil temperature across 4 depths (K) |

- All numerical features standardized with `StandardScaler`.  

---

## D. Feature Selection  

- **Monthly → Yearly Aggregation**: All climate/soil data converted to annual averages.  
- **Land Cover**: Only Cropland and Cropland/Natural Vegetation retained.  
- **Multicollinearity Reduction**:  
  - Soil moisture and temperature (across depths) consolidated into two variables: SoilMoi & SoilTMP.  
  - High correlation between TVeg_mean and Rainf_mean → kept only Rainf_mean.  

---

## Key Takeaways  

- Cleaned and merged 17 datasets into a unified **country-year crop yield dataset**.  
- Applied **feature engineering, outlier removal, and aggregation** for consistency.  
- Built a **PyTorch MLP** with embeddings for categorical features.  
- Achieved stable performance with **Adam optimizer** on log-transformed yields.  

---
