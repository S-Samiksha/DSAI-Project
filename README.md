# DSAI Project: Brazillian Olist E-commerce Database
Hello! We are [Sankar Samiksha](https://github.com/S-Samiksha), Fathima and Gideon from Nanyang Technological University. We were tasked to do a project in Data Science. We chose the Brazillian Olist E-commerce Database. 
| Work Done | Files to look at |
| --- | --- |
| Single Variate Analysis |(In order) Review Status, Delivery Time, Product Type |
| Machine Learning | Training |

# About this project
**The Research Question:** 
How different variables such as **actual delivery time**, **the difference between actual and estimated wait time**, **order status**, **freight value**, **payment value**, **payment installment** affect the **review score** in each of the different **product type categories**, `Houseware`, `auto`, `furniture decor`, `computer accessories`, `health beauty`, `sports leisure`? 


# About the Database
![image](https://user-images.githubusercontent.com/71448008/112981086-44770880-918d-11eb-91f0-996f72c4ddde.png)
Taken from: https://www.kaggle.com/olistbr/brazilian-ecommerce?select=olist_products_dataset.csv 

All CSVs that are being used for this project are uploaded. 

# Single Variate Analysis
There are 4 parts to Single Variate Analysis:
Part 1: Review Status 
Part 2: Delivery Time 
Part 3: Product Type 
Part 4: Payment Mode 

Each of these parts explain the CSVs provided by the Kaggle Website. Additionally, it helps to seive out variables that affect review score such that it can be later used for multi variate analysis and thereafter machine learning. 

The first part explores the response variable which is the `review score`. The reason why `review score` was chosen as the response variable is because review score is a major contributor to perception about the sellers. When this perception is positive, consumers are more likely to purchase from that particular seller. 
This is substantiated by a research conducted by Profitero: 
<br>
[Assessing the Impact of Ratings and Reviews on eCommerce Performance.pdf by Keith Anderson](http://insights.profitero.com/rs/476-BCC-343/images/Assessing%20the%20Impact%20of%20Ratings%20and%20Reviews%20on%20eCommerce%20Performance.pdf)
Part 1 also analysizes how to clean the data with respect to review score. One such example is class balancing. This is further explained in the python notebook. 


Parts 2 to 4 explore the predictors that will be used in the multivariate analysis and machine learning. Additionally, part 2 to 4 single variate analysis provides insights into how to clean data and prepare it for the multi variate and machine learning process in this project. 


# Multi Variate Analysis 
