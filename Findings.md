WALMART SALES ANALYSIS
======================

Project Objective
-----------------
The goal of this project is to investigate the factors that influence Walmart's
weekly sales and determine how machine learning can be used to predict weekly
sales.

The analysis focuses on store-level differences, seasonal patterns, holidays,
economic conditions, environmental factors, and the predictive performance of
Linear Regression and Random Forest models.


#1. DATASET OVERVIEW
-------------------

The dataset contains weekly sales information for 45 Walmart stores.

Number of observations: 6,435
Number of stores: 45
Date range: February 5, 2010 - October 26, 2012

Variables:

Store          - Store number
Date           - Week start date
Weekly_Sales   - Weekly sales
Holiday_Flag   - Indicates whether the week is associated with a holiday
Temperature    - Regional air temperature
Fuel_Price     - Regional fuel price
CPI            - Consumer Price Index
Unemployment   - Regional unemployment rate

There were no missing values in the dataset.


#2. DATE PROCESSING
------------------

The original Date column was stored as a string in the format:

05-02-2010

The Date column was converted to a Pandas datetime using:

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)

Several additional features were extracted from Date:

Year
Month
Week
Day_of_Year

These features were created to capture seasonal and temporal patterns in
weekly sales.


#3. WEEKLY SALES DISTRIBUTION
----------------------------

Weekly sales have substantial variation across the dataset.

Mean Weekly Sales:      approximately $1.047 million
Median Weekly Sales:    approximately $961 thousand
Minimum Weekly Sales:   approximately $210 thousand
Maximum Weekly Sales:   approximately $3.819 million
Standard Deviation:     approximately $564 thousand

The large difference between minimum and maximum weekly sales suggests that
there are substantial differences in sales between stores and across time.


#4. STORE-LEVEL ANALYSIS
-----------------------

Store was found to be the strongest predictor of Weekly_Sales.

The stores with the highest average weekly sales included:

Store 20: approximately $2.108 million
Store 4:  approximately $2.095 million
Store 14: approximately $2.021 million
Store 13: approximately $2.004 million
Store 2:  approximately $1.926 million

Some of the stores with the lowest average sales included:

Store 33: approximately $260 thousand
Store 44: approximately $303 thousand
Store 5:  approximately $318 thousand
Store 38: approximately $386 thousand
Store 3:  approximately $403 thousand

This demonstrates that store-level differences are extremely important when
predicting Walmart's weekly sales.

Possible explanations include differences in store size, location, customer
base, local demand, and demographics. However, these factors are not included
in the dataset.


#5. HOLIDAY ANALYSIS
-------------------

Average weekly sales were compared between holiday and non-holiday weeks.

Non-Holiday Average Sales:
approximately $1.041 million

Holiday Average Sales:
approximately $1.123 million

Holiday weeks therefore had higher average sales than non-holiday weeks.

However, the Holiday_Flag variable does not fully capture increased sales
before holidays.


#6. OUTLIER ANALYSIS
-------------------

An outlier analysis identified 25 observations as outliers.

Several of the largest sales observations occurred immediately before
Christmas.

Examples include:

Store 14 - 2010-12-24 - $3,818,686
Store 20 - 2010-12-24 - $3,766,687
Store 10 - 2010-12-24 - $3,749,058
Store 4  - 2011-12-23 - $3,676,389
Store 13 - 2010-12-24 - $3,595,903

This was an important finding because many of these observations had:

Holiday_Flag = 0

Therefore, a week can experience extremely high sales even when it is not
classified as a holiday week.

This suggested that sales may increase during the period immediately before
a holiday.


#7. CORRELATION ANALYSIS
-----------------------

The correlations between Weekly_Sales and several continuous variables were:

Temperature:    -0.064
Fuel_Price:      0.009
CPI:            -0.073
Unemployment:   -0.106

These correlations are relatively weak.

This suggests that there is not a strong simple linear relationship between
weekly sales and these individual variables.

However, weak correlation does not necessarily mean that a variable has no
predictive value. Random Forest can identify nonlinear relationships and
interactions that simple correlation does not capture.


#8. LINEAR REGRESSION
--------------------

Linear Regression was used as a baseline model.

The model included variables such as:

Store
Year
Month
Week
Holiday_Flag
Unemployment
Day_of_Year

The Linear Regression model produced:

R²: approximately 0.117

This means the model explained only about 11.7% of the variation in weekly
sales.

The relatively low R² suggests that a simple linear relationship between the
available predictors and Weekly_Sales is not sufficient to explain the
variation in the dataset.


#9. RANDOM FOREST REGRESSION
---------------------------

A Random Forest Regressor was used to model nonlinear relationships and
interactions between the variables.

The model was configured using:

n_estimators = 200
random_state = 20

The Random Forest substantially outperformed the Linear Regression model.

Using a random 80/20 train-test split, the Random Forest achieved approximately:

R²:   0.957
MAE:  $60,608
RMSE: $114,473

This indicates that Random Forest was able to capture relationships in the
data that Linear Regression could not.


#10. FEATURE IMPORTANCE
----------------------

The Random Forest feature importance results were:

Store:          0.671380
CPI:            0.157441
Unemployment:   0.090091
Day_of_Year:    0.027019
Week:           0.025771
Temperature:    0.013437
Fuel_Price:     0.009701
Month:          0.001864
Year:           0.001757
Holiday_Flag:   0.001541

Store was by far the most important feature.

Approximately 67% of the Random Forest's feature importance was attributed
to Store.

CPI and Unemployment were the next most important variables.


#11. REMOVING STORE
------------------

Because Store was so dominant, the model was retrained without Store.

The resulting model produced:

R²:   approximately 0.117
MAE:  approximately $374,849
RMSE: approximately $516,907

This was a substantial decrease in performance.

The results demonstrate that Store captures a very large amount of the
variation in weekly sales.

Without Store, feature importance shifted toward economic and environmental
variables:

Unemployment: 28.84%
Temperature:  23.71%
CPI:          21.02%
Fuel_Price:   10.42%
Day_of_Year:   6.91%
Week:          4.81%
Year:          2.36%
Month:         1.27%
Holiday_Flag:  0.67%

This suggests that economic and environmental variables contain useful
information, but they cannot replace the predictive information provided by
store identity.


#12. HOLIDAY_LEADUP FEATURE
--------------------------

A new binary feature called holiday_leadup was created.

The feature is intended to equal:

1 = a holiday occurs within the next 7 days
0 = otherwise

The purpose of this feature is to capture increased purchasing activity
before a holiday.

This is different from Holiday_Flag because Holiday_Flag identifies the
holiday week itself, while holiday_leadup attempts to capture the period
leading up to the holiday.

The feature was added to the Random Forest model.


#13. HOLIDAY_LEADUP FEATURE IMPORTANCE
-------------------------------------

After adding holiday_leadup, the Random Forest feature importance was:

Store:           0.610275
CPI:             0.181054
Unemployment:    0.120528
Day_of_Year:     0.028169
Week:            0.026406
holiday_leadup:  0.015095
Temperature:     0.009313
Fuel_Price:      0.005879
Month:           0.001847
Holiday_Flag:    0.001247
Year:            0.000187

The holiday_leadup feature had approximately 1.51% feature importance.

Interestingly, this was substantially higher than the importance of the
original Holiday_Flag feature, which had approximately 0.12% importance.

This suggests that the timing leading up to a holiday may contain more
predictive information than simply identifying the holiday week.

However, feature importance should not be interpreted as a causal effect.


#14. TIME-BASED TRAIN/TEST SPLIT
-------------------------------

Because sales data are temporal, a time-based train/test split was also
investigated.

For example:

train = df[df["Date"] < "2012-01-01"]
test = df[df["Date"] >= "2012-01-01"]

This approach prevents future observations from being randomly placed into
the training data.

The time-based split produced:

R²:   0.7244
MAE:  $158,139.90
RMSE: $281,652.18

These results were worse than the random train/test split.

However, this is expected because the time-based split is a more difficult
prediction problem.

A random split allows the model to learn from observations occurring both
before and after individual test observations. This can produce an overly
optimistic estimate of future predictive performance.

The time-based split is therefore more appropriate if the goal is to predict
future Walmart sales.


#15. CHOOSING A TIME SPLIT
-------------------------

The dataset ends on October 26, 2012.

A potential time-based split is:

Training:
February 2010 - September 2011

Testing:
October 2011 - October 2012

Using October 1, 2011 as the cutoff is particularly useful because the test
period includes the 2011 holiday season.

This is important because the exploratory analysis found that some of the
largest sales spikes occurred immediately before Christmas.

A future analysis could compare multiple time-based cutoffs to determine
whether model performance is stable across different periods.


#16. KEY FINDINGS
----------------

1. Store is the strongest predictor of Weekly_Sales.

Store accounted for approximately 67% of Random Forest feature importance.

2. Walmart stores have very different average sales levels.

The highest-performing stores had average weekly sales above $2 million,
while some stores averaged less than $300,000.

3. Holiday weeks have higher average sales.

Holiday weeks averaged approximately $1.123 million compared with
approximately $1.041 million for non-holiday weeks.

4. Several of the largest sales observations occurred immediately before
Christmas.

This suggests strong seasonal and pre-holiday purchasing behavior.

5. Holiday_Flag alone does not fully capture holiday-related sales behavior.

Several extremely high-sales observations were classified with
Holiday_Flag = 0.

6. The holiday_leadup feature provides additional information.

It had approximately 1.51% Random Forest feature importance compared with
approximately 0.12% for Holiday_Flag.

7. Economic variables provide predictive information.

CPI and Unemployment were among the most important features in the Random
Forest model.

8. Linear Regression performed substantially worse than Random Forest.

The Linear Regression model produced an R² of approximately 0.117, while
the Random Forest produced a much higher R² under the random split.

9. Time-based evaluation produces more conservative results.

The time-based Random Forest achieved an R² of approximately 0.724 compared
with approximately 0.957 for the random split.

10. Store-level information is critical.

Removing Store caused a dramatic decline in predictive performance.


#17. LIMITATIONS
---------------

The dataset has several limitations.

First, Store is essentially an identifier for location. While it is highly
predictive, the dataset does not explain the underlying reasons why stores
have different sales levels.

Second, the dataset does not contain potentially important variables such as:

- Store size
- Population
- Demographics
- Competition
- Promotions
- Product-level sales
- Inventory
- Marketing activity

Third, Holiday_Flag is a simplified representation of holidays and does not
fully capture the timing or intensity of holiday demand.

Fourth, feature importance does not establish causation. A feature can be
useful for prediction without directly causing changes in sales.

Finally, a random train/test split can overestimate performance for temporal
data. A time-based evaluation is more appropriate for estimating future
prediction performance.


#18. FUTURE IMPROVEMENTS
-----------------------

Several improvements could be made to the analysis.

1. Add lagged sales features.

Examples:

Previous week's sales
Previous month's sales
Sales from the same week in the previous year

2. Add rolling statistics.

Examples:

4-week moving average
8-week moving average
12-week moving average

3. Add specific holiday variables.

Instead of using only Holiday_Flag, separate holidays could be represented
individually.

4. Add store characteristics.

Store size, location, demographics, and competition could improve the model.

5. Experiment with additional machine learning models.

Potential models include:

Gradient Boosting
XGBoost
LightGBM
Extra Trees
Neural Networks

6. Use time-series cross-validation.

This would provide a more robust estimate of future model performance.

7. Use permutation importance or SHAP values.

These methods could provide more detailed explanations of how individual
features affect predictions.

8. Develop store-specific models.

Because stores have substantially different sales levels, separate models
or store-specific features could potentially improve predictions.


#19. CONCLUSION
--------------

The analysis demonstrates that Walmart weekly sales are influenced by a
combination of store-level characteristics, economic conditions, and
seasonality.

The most important finding is that Store is by far the strongest predictor
of Weekly_Sales. The large differences in average sales between stores
explain why including Store dramatically improves Random Forest performance.

Economic variables such as CPI and Unemployment also provide useful
predictive information. Temperature and Fuel_Price have smaller but
measurable contributions.

The exploratory analysis identified substantial sales spikes around the
Christmas period. This motivated the creation of the holiday_leadup feature,
which captures whether a holiday is approaching within seven days. Although
holiday_leadup is not one of the strongest predictors, it provides more
information to the model than the original Holiday_Flag.

Random Forest substantially outperformed Linear Regression, demonstrating
the value of nonlinear machine learning methods for this dataset.

Finally, the time-based evaluation produced a lower but more realistic
estimate of predictive performance than the random train/test split. For a
real-world sales forecasting application, preventing future information from
entering the training data is important.

Overall, the analysis suggests that Walmart sales prediction benefits most
from understanding differences between stores, seasonal patterns, and
economic conditions, while additional temporal and store-level features
could further improve the model.
