### Impact of the COVID-19 pandemic and generational heterogeneity on ecommerce shopping styles 

This repository contains the replication data and code for \
\
`Luo et al. (2022) Impact of the COVID-19 pandemic and generational geterogeneity on ecommerce shopping styles - A case study of Sacramento, California`

The paper uses kmeans clustering algorithm to group households by their shopping channel usage patterns (i.e., share of online orders) across different commodity types during the pandemic and identifies 5 shopping style groups including ECommerce Independent, Ecommerce Dependent and three mixed mode in-between. A distance-based rule is applied to identify household's shopping style pre-pandemic based on their online shopping frequency pre-pandemic and the dynamic of shopping behaviors under the impact of pandemic is further examined. In addition, generational heterogeneity is highlighted in this paper that different generation cohorts are influenced disproportionally.

The paper is submitted to [Communication in Transportation Research (COMMTR)](https://www.sciencedirect.com/journal/communications-in-transportation-research) and it is now under review.

### Code
All main results in the paper can be replicated by running the `script/ECommerceShopping.ipynb` notebook

`script/utils.py` contains functions used to cluster households by their shopping channel usage patterns pre- (*Past_Label*) and during- (*kmeans_missing*) pandemic, as well as a function (*independent_ttest*) for conducting a one-tailed two-sample t-test.

### Data
- The data used in this analysis is mostly from a retrospective revealed preference survey distributed among households in Sacramento through Qualtrics. The dataset we provide (`data/originalData/subset_data.csv`) is the raw survey data with partial but sufficient (to replicate the main results in the paper) information from the whole survey dataset.

- Records from 313 households was included.

- Each row corresponds to the record of a household and columns contain: 
  - the household's instore and online shopping frequency across different commodity types during the pandemic, and their online shopping frequency pre-pandemic 
    - `May-InStore-Trips-`
    - `May-InStore-Frequency-`
    - `May-ECommerce-Orders-`
    - `May-ECommerce-Frequency-`
    - `PastYear-ECommerce-Frequency-`
    
  - respondent's intention to use different shopping channels after the reopen of California 
    - `DoInStage23-ContactlessDoor`
    - `DoInStage23-CurbsidePickup`
    - `DoInStage23-InStorePickup`
    - `DoInStage23-VisitRetail`
    - `DoInStage23-VisitRestaurant`
  - demographic information: respondent's age, personal income, household income, household size 
    - `age`
    - `PersonalIncome`
    - `HouseholdIncome`
    - `NumberInHousehold`
  - number of kids (`kids`) in the household pre-pandemic (obtained from Sacramento Area Council of Government (SACOG) 2018 household travel survey (HTS).

### Environment
The code is written in python 3. Packages and library required include:
- `numpy` (1.18.5) 
- `pandas` (1.0.5)
- `matplotlib` (3.2.2)
- `scipy` (1.5.0)

