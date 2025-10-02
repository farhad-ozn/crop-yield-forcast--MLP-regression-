Author: Farhad Oghazian (April 2025)
Forecasting Yield of Crop Products using Multi Layer Perceptron method
A. Preprocessing 
Preprocessing was a critical and time intensive part of this project due to the size, resolution, and structure of the given datasets. Below is a description of preprocessing steps taken, along with their rationale and implementation strategy. 
 Loading and Merging Raw Data 
We began by loading 17 separate CSV files that included data on: 
•	Crop yields and production (from FAO) 
•	Monthly climate and soil features (4 CSV file for monthly SoilTMP and 4 file for SoilMoi each file related to a depth/level , CanopInt_inst_data, ESoil_tavg_data, 
Rainf_tavg_data, Snowf_tavg_data, TVeg_tavg_data and TWS_inst_data ) 
•	Land cover data 
•	Geospatial lookup table for country centroids 
Each dataset had its own format, resolution, and anomalies (i.e, Null, zero) requiring alignment before merging. 
Data cleaning and wrangling: 
Summary statistics used to find anomalies in datasets and investigate raw data better, we took notes and proceeded as follows: 
•	In country look up table we checked for duplicates, Null values and removed countries that have 'centroid radius' = 0 (7 rows removed from lookup table) 
•	Only class 12 (Croplands) and class 14 (Cropland/Natural Vegetation Mosaics) columns were retained from the Land_cover_percent_data for analysis, as they are most relevant to agricultural land use 
• Averaging monthly columns in climate and soil datasets per year 
Based on the given units for SoilTMP [Kelvin], SoilMoi [Kg/m2], Rainf_tavg [Kg/m2S], TVeg_tavg [Watts/m2], ESoil_tavg [Watts/m2], CanopInt_inst [ Kg/m2 ], and TWS_inst [mm], and also considering their biological importance, these variables reflect ongoing conditions that impact crops regularly, so in this report average behaviour of monthly values over the year were considered.  
• Yield_and_Production_data 
The formula to compute yield is Yield= Production/AreaHarvested, so the variable Production is already included in Yield, so it is decided to remove production data from the dataset. (Including both Production and Yield as features can lead to data leakage. Production is not a true input feature but a result of the same outcome we're trying to predict). Therefore, we filtered the Element column to contain only yield. And to assess the quality of the yield data, we examined the flag descriptions provided alongside each entry. According to the FAO website, the five distinct flags in our dataset are:  
 A: Official figure  
 E:Estimated value 
 I: Imputed value 
 M: Missing value (data cannot exist, not applicable)  X: Figure from international organizations. 
The Flag M is not acceptable for our analysis. (although flags 'I' and 'E' are estimated and imputed and represent non-official data, they were kept for modelling purposes to preserve sample size). Therefore, only data points with flag 'M' were excluded from the dataset before training. 
And also irrelevant columns Domain, Element (e.g., "production"), Item code (CPC), and Flag description removed. At the end we left with a clean target variable (Yield), the key identifiers (Country, Year) and one feature (Item). 
• Country Assignment Based on Coordinates: 
Longitude and latitude and year were common keys between datasets, except for crop Yield_and_Production dataset. Assigning a country enables merging these with Yield_and_Production_data. Therefore, all the datasets can be connected to each other by using appropriate keys in lookup table.  For datasets containing latitude and longitude information, a custom function defined to assign each data point (longitute/latitude) to a country. The function first extracts centroid coordinates and country names from a lookup table to match spatial data points in to countries. Then assigns each (latitude, longitude) point to the nearest country centroid, using Euclidean distance under a flat-Earth assumption. If a point falls within the radius of one or more centroids, it is assigned to the nearest one; otherwise, it's matched to the closest centroid. And finally, latitude and longitude columns are dropped from datasets after country assignment and all their data grouped by country and year (i.e., averaged). That is because our target (Yield) is at country-year level and environmental features were previously at coordinate level, so aggregating to country-year makes them compatible before join/merging.  
• Merging  
all cleaned and aligned datasets merged on country and year using an outer join, to retain all possible country-year combinations. And final output named merged_df served as our master modelling dataset. However it needed: 
• More cleaning o checking for nulls: the rows with null value in Yield and Item column removed 
(3442 instances) o checking for nulls again: 7692 rows that were present in all climate and soil columns removed. 
o After investigating multicollinearity among features and creating new features that was described in section C, We also investigate the distribution of Yield. Its graph showed extremely right skewed distribution. Checking for possible outliers we identify a small number of crop-country pairs (29 instances) exhibited unrealistically high yield values (e.g., Watermelons in the 
Dominican Republic, Papayas in Guyana, etc.), with yields exceeding 200,000 – 300,000 kg/ha). Therefore it is decided to remove values above 99.97% percentile.  

B. Modeling
For modelling, a 3-layer Multilayer Perceptron (MLP) for regression used, and built using PyTorch. The structure includes embedding layers for categorical variables with fully connected layers for numerical inputs. The model was trained to predict the log-transformed yield (log(1 + Yield)).  
Input → hidden_layer_1 → hidden_layer_2 → Output 
 Inputs 
The model (YieldMLP) accepts numerical and categorical inputs. Categorical inputs include ‘country’ with 163 unique values (embedded to 50 dimensions) and ‘Item’ with 102 unique values (embedded to 50 dimensions). These are passed through nn.Embedding layers to learn representations. Numerical inputs are 8 features including ‘Rainf_mean’, ‘SoilMoi’, etc. (see Section C) that were concatenated into a single input tensor.  
The Dataset class (YieldDataset), wraps numeric features, categorical indices, and targets and returns a tuple: (X_num, X_cat, y) for each sample. 
 Hidden layers 
Layer_1 has 256, and layer_2 256 neurons. LeakyReLU activation (with alpha = 0.01) used on both hidden layers which by trial found to have better performance compared to ReLU on our dataset. 
 Output layer 
It is a single neuron for regression output (i.e., log(1+Yield))  
 

We used the Adam optimizer with weight decay for L2 regularization (to manage overfitting) for training. SGD did not produce satisfactory results, and its performance was very sensitive 
to hyperparameters (especially the learning rate). Adam produced much more consistent and faster convergence across different configurations in grid search. 
 
C. Features & Labels 
	Output variable: The target variable for our MLP regression is yield_log: 
this is log(1+Yield) per crop, per country, per year where Yield is the original values in the dataset (kg/ha). This log transformation helps our model to generalize better across both high and low yield of crops. Before taking Log we added +1 to Yield to prevent having zero in the logarithm. After prediction, the output transformed back to its original form using np.expm1 (i.e., exp(yield_log)−1). This ensures final results are in the original scale (kg/ha), which is required for interpretability and calculating performance metric (i.e., R2, MEA, RMSE metrics). 
	Input Features (Xi): We used a combination of categorical and numerical features as model inputs. The total feature set was carefully selected based on relevance and collinearity among features, and extracted from 17 datasets provided in the coursework. 
1. Categorical Features  
ML models require numerical inputs, but country and Item are categorical (strings). We have many categories (e.g., 163 countries, 102 crops). One solution is to use embedding-based encoding for categorical variables. Categorical features like 'country' and 'Item' are first converted into integer indices using label encoding.  
{data[col + '_idx'] = data[col].cat.codes}, This assigns each unique category an integer ID. Inside the neural network, these integer indices are passed to nn.Embedding layers, which map each index to a learnable dense vector (e.g., 50dim). The size of each learnable dense embedding vector for categorical variables was chosen as min(50, (number of categories + 1) // 2). 
{self.embeddings = nn.ModuleList([ nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in emb_dims])} 
a.	country (country name, e.g., “Sweden”) with 163 categories 
b.	Item (crop type, e.g., “Apple”) with 102 categories 
country and Item were label-encoded to integer indices using .astype('category').cat.codes. 
These were passed through PyTorch embedding layers to learn dense vector representations during training. This avoids the sparsity of one-hot encoding (i.e., another technique for handling categorical features) while the model captures relationships between similar categories. We tried one-hot encoding and did not get satisfactory results. (e.g., R2 was negative) 
2.  Numerical Features 
The following table summarizes the 8 numerical features used, along with a short description of each. Note that year only used as a data splitting criterion and not a feature. 
Feature 	Description 
Land_cover_ percent_mean 	Average % of land classified as cropland (MODIS class 12 + 14) per country and year 
CanopInt_mean 	Average water on plant surface(kg/m²) in each country and year 
Rainf_mean 	Monthly average rate of rainfall (kg/m²/s) per country and year 
ESoil_mean 	Average evaporation from soil (W/m²) per country and year 
Snowf_mean 	Monthly Average snowfall rate (kg/m²/s) per country and year 
TWS_mean* 	Monthly Average Terrestrial water storage (mm) per country and year 
SoilMoi 	Monthly Average of soil moisture over 12 months, further averaged over 4 depths per each country and year (kg/m²) 
SoilTMP 	Monthly average of soil temperature over 12 months, further averaged over 4 depths per each country and year (K) 
* TWS is expressed in millimetres of water equivalent, which represents the depth of water stored over a unit area, therefore average per country used 
It is worth mentioning that all numerical features were standardized using StandardScaler, fitted(fit.transform) on the training set and applied (.transform) to validation and test set. 
 
D. Feature selection  
Monthly to yearly aggregation 
Soil temperatures (i.e., SoilTMP at 4 different depths), moistures (i.e., SoilMoi at 4 different depths), CanopInt_inst_data, ESoil_tavg_data, Rainf_tavg_data, Snowf_tavg_data, TVeg_tavg_data and TWS_inst_data were originally monthly per latitude per longitude, we computed yearly averages across all 12 months to match the temporal resolution of  Yield_and_Production data.  
Land cover 
For the land cover percentage data, only class 12 and class 14 were included and summed up in the analysis. Out of the 17 MODIS land cover classes class 12 corresponds to Croplands and class 14 represents Cropland/Natural Vegetation Mosaics.  
Including both classes allows a more accurate representation of agricultural activity, especially in tropical regions where cropland is often mixed with natural vegetation. 
Multicollinearity reduction 
To handle redundant features, we computed pairwise Spearman correlations across all numeric features and identified highly correlated soil moisture and temperature layers (ρ > 0.9). The result showed that averaged moisture features across depths and also averaged temperatures across depths are extremely correlated. To address multicollinearity among soil-related features, we aggregated highly correlated soil moisture and soil temperature across depths into two new features (SoilMoi and SoilTMP) using their annual averages.  o SoilMoi: monthly average of all soil moistures at 4 depths per country per year 
o 	SoilTMP: monthly average of all soil temperatures at 4 depths per country per year 
These two new features preserve the underlying signal and allow us to remove 6 redundant inputs/features. After consolidating soil moisture and temperature features, we checked multicollinearity again and found one remaining high correlation pair: TVeg_mean and Rainf_mean. We confirmed the strong correlation between TVeg_mean (TVeg : Evaporation of water from plant) and Rainf_mean (Rainf : Rate of rainfall) using a Pearson correlation test (r= 0.94, p < 0.05). Therefore we statistically proved another redundancy and kept only Rainf_mean in the feature set. (It seems logical because evaporation of water from plant inherently linked to water availability -i.e., rainfall) 
  

  
