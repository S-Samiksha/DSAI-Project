# DSAI Project: Brazillian Olist E-commerce Database
Hello! We are [Sankar Samiksha](https://github.com/S-Samiksha), [Fathima](https://github.com/fath3725) and [Gideon](https://github.com/gmanik30) from Nanyang Technological University. We were tasked to do a project in Data Science. We chose the Brazillian Olist E-commerce Database. 
| Work Done | Files to look at |
| --- | --- |
| Exploratory Data Analysis |(In order) Review Status, Delivery Time, Product Type |
| Machine Learning | Training |

# About this project
**The Research Question:** 
How different variables such as **actual delivery time**, **the difference between actual and estimated wait time**, **freight value**, **payment value**, **payment installment** affect the **review score** in each of the different **product type categories**, `Houseware`, `auto`, `furniture decor`, `computer accessories`, `health beauty`, `sports leisure`? 


# About the Database
![image](https://user-images.githubusercontent.com/71448008/112981086-44770880-918d-11eb-91f0-996f72c4ddde.png)
Taken from: https://www.kaggle.com/olistbr/brazilian-ecommerce?select=olist_products_dataset.csv 

All CSVs that are being used for this project are uploaded. 

# Packages To Install
 1. Graphviz

# Data Extraction, Curation, Preparation & Cleaning
 1. Merging Datasets
 2. Filtering reviews based on order status
 3. Reclassifying review score
 4. Splitting the dataset into 6 product categories
 5. Removing duplicates and null values
 6. Balancing review score

The above is done throughout the notebooks and not in any specific notebook.

# Exploratory Data Analysis (EDA)
There are 4 parts to EDA:
 1. Part 1_Review Status
 2. Part 2_Delivery Time
 3. Part 3_Product Type
 4. Part 4_Payment Mode 

Each of these parts explain the CSVs provided by the Kaggle Website. Additionally, it helps to sieve out variables that affect review score such that it can be later used for multi variate analysis and thereafter machine learning. 

The first part explores the response variable which is the `review score`. The reason why `review score` was chosen as the response variable is because review score is a major contributor to perception about the sellers. When this perception is positive, consumers are more likely to purchase from that particular seller. 
This is substantiated by a research conducted by Profitero: 
<br>
[Assessing the Impact of Ratings and Reviews on eCommerce Performance.pdf by Keith Anderson](http://insights.profitero.com/rs/476-BCC-343/images/Assessing%20the%20Impact%20of%20Ratings%20and%20Reviews%20on%20eCommerce%20Performance.pdf)

Part 1 also analyses how to clean the data with respect to review score. One such example is class balancing. This is further explained in the python notebook. 

Parts 2 to 4 explore the predictors that will be used in the multivariate analysis and machine learning. Additionally, parts 2 to 4 provides insights into how to clean data and prepare it for the multi variate analysis and machine learning process in this project. 

# Multi Variate Analysis 
For the machine learning, we initially used a Decision Tree. The classification accuracy was above 65% for most product types. False positive rate was below 50% for most product types. The classification accuracy could have been better and the false positive rate was too high.

Hence, we used a Random Forest. The classification accuracy increases to almost 80% for most product types. Additionally, the false positive rates drop below 30%. Hence, Random Forest was a better suited technique. We tested and found out that the following hyperparameters provide the best result (without overfitting): max_depth = , n_estimators =  .

# Results


Thank you for dropping by.
