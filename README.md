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
 5. Part 5_Multi Variate Analysis

Each of these parts explain the CSVs provided by the Kaggle Website. Additionally, it helps to sieve out variables that affect review score such that it can be later used for multi variate analysis and thereafter machine learning. 

The first part explores the response variable which is the `review score`. The reason why `review score` was chosen as the response variable is because review score is a major contributor to perception about the sellers. When this perception is positive, consumers are more likely to purchase from that particular seller. 
This is substantiated by a research conducted by Profitero: 
<br>
[Assessing the Impact of Ratings and Reviews on eCommerce Performance.pdf by Keith Anderson](http://insights.profitero.com/rs/476-BCC-343/images/Assessing%20the%20Impact%20of%20Ratings%20and%20Reviews%20on%20eCommerce%20Performance.pdf)

Part 1 also analyses how to clean the data with respect to review score. One such example is class balancing. This is further explained in the python notebook. 

Parts 2 to 4 explore the predictors that will be used in the multivariate analysis and machine learning. Additionally, parts 2 to 4 provides insights into how to clean data and prepare it for the multi variate analysis and machine learning process in this project. 

Part 5 explores the predictors with respect to their product category.

# Multi Variate Analysis 
A decision tree (part 6) is a basic machine learning tool that did not provide a very high classification accuracy(65% to 75%), true positive(60% to 85%) and true negative(50% to 60%). There was a large distribution of these values across product types too. Every time the notebook is run, the train and test classification accuracy would differ from a range of 1% to 10%. Additionally, false positive was above 40% for most product types across multiple runs of the ipynb. This shows that the machine learning tool was not the best.

Another option was to run random forest. A random forest uses a 'forest', a multitude of decisions trees that help to classify the data points into the different review scores. The reason why random forest works so well is that "A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models." [1]

There is low correlation between the trees and the trees help to cover each others errors to enable the most accurate classification. Hence we see that the random forest does enable better classification in part 7. Classification accuracy increase above 80% and false positive rates fell below 30%. Train and test were similar in terms of classification accuracy.

False positives are generally bad given our reseach question. This is because sellers want to adjust their actual delivery time, the difference between actual and estimated wait time, freight value, payment value, payment installment such that they obtain the best review score 1 which is translated to 3 to 5 review score. If they adjust these factors and get a false positive, then they may predict a high review score but end up getting a low one.

Hence, although the classification accuracy is relatively better, an additional step must be done to reduce the false positive. Tuning of hyperparameters must be done to achieve the highest possible classification accuracy, true positive and true negative and lowest possible false positive and false negative. Hyperparameters are used to make the random forest. There are many hyperparameters but for the scope of this project max_depth and n_estimators were chosen. max_depth is the maximum depth each decision tree goes in the 'forest' of trees. n_estimators is the number of trees in the forest. [2]

To find the best hyperparemeters a Grid Search is done.[3][4][5][6][7][8]

Grid Search will then run a range for the hyperparameters as seen in part 7. The definition provided by Sci-Kit is "The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid." Cross validation is defined as "Cross validation is a statistical method used to estimate the performance (or accuracy) of machine learning models." [9]

Best score is the "Mean cross-validated score of the best_estimator". Since cv = 5, the train and test split will occur five times for each hyperparameter. The best score is calculated for each try and for each combination of hyperparameters. The best score will return the average of the best hyperparameter combination in those 5 tries.

This function will return the best hyperparameters max_depth and n_estimators. These are then run into the random forest in this very ipynb. Seen from above, classification accuracy, true positive and true negative are above 95% and false positive and false negative are below 1% for all product types. Train and test data are very similar. Hence, there is no overfitting as explained earlier.

This is a comparison of the different machine learning results:
{"A?":"B","a":5,"d":"B","h":"www.canva.com","c":"DAEXkuQnfLk","i":"8zl65Zp3it1-dczTYypLkQ","b":1619165828930,"A":[{"A?":"I","A":199.64137862848582,"B":183.58251021499086,"D":1552.8349795700185,"C":829.678979422435,"a":{"B":{"A":{"A":"MAEcfpSar_0","B":1},"B":{"A":-1.1368683772161603e-13,"B":-1.1368683772161603e-13,"D":1552.8349795700187,"C":829.6789794224352}}}}],"B":1920,"C":1080}

# Conclusion
From our machine learning we learnt the following:
<br>
We learnt that sellers should liaise with a delivery team that is able deliver faster to get lower actual times and lower actual minus estimated time to increase their review score. Similarly, to attain lower freight value ,sellers can order in bulk or find companies that allow lower freight value. 
To achieve lower payment value, sellers should Create deals with credit card companies or banks to enable the lowest payment value. To achieve lower payment installments sellers can create better and a more variety of payment installment plans. 


# Reference 
Olist, “Brazilian E-Commerce Public Dataset by Olist,” Kaggle, 29-Nov-2018. [Online]. Available: https://www.kaggle.com/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv. [Accessed: 23-Apr-2021].
L. Breiman and A. Cutler, “Random Forests Leo Breiman and Adele Cutler,” Random forests - classification description. [Online]. Available: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm. [Accessed: 23-Apr-2021].
K. Anderson, D. Reifenberger, and T. O' Neil, “Assessing the Impact of Ratings and Reviews on eCommerce Performance,” https://www.profitero.com/. [Online]. Available: http://insights.profitero.com/rs/476-BCC-343/images/Assessing%20the%20Impact%20of%20Ratings%20and%20Reviews%20on%20eCommerce%20Performance.pdf. [Accessed: 23-Apr-2021].
“eCommerce is Set to Increase 39 Percent by 2022 in Brazil, Reaching Nearly R$150bn - Press Releases: FIS,” FIS Global. [Online]. Available: https://www.fisglobal.com/en/about-us/media-room/press-release/2018/ecommerce-is-set-to-increase-39-percent-by-2022-in-brazil-reaching-nearly-r150bn. [Accessed: 23-Apr-2021].
B. L. F. @biancamvickers, “Brazilian E-commerce Market 2016 Highlights,” PagBrasil, 17-Jul-2019. [Online]. Available: https://www.pagbrasil.com/insights/brazilian-e-commerce-market-2016-highlights/. [Accessed: 23-Apr-2021].
H. M. -, By, -, Hussain MujtabaHussain is a computer science engineer who specializes in the field of Machine Learning.He is a freelance programmer and fancies trekking, H. Mujtaba, Hussain is a computer science engineer who specializes in the field of Machine Learning.He is a freelance programmer and fancies trekking, and P. enter your name here, “What is Cross Validation in Machine learning? Types of Cross Validation,” GreatLearning Blog: Free Resources what Matters to shape your Career!, 24-Sep-2020. [Online]. Available: https://www.mygreatlearning.com/blog/cross-validation/. [Accessed: 21-Apr-2021].
abuabu 54777 silver badges1616 bronze badges, Mischa LisovyiMischa Lisovyi 2, and Vivek KumarVivek Kumar 28.8k66 gold badges7575 silver badges109109 bronze badges, “Interpreting sklearns' GridSearchCV best score,” Stack Overflow, 01-Feb-1967. [Online]. Available: https://stackoverflow.com/questions/50232599/interpreting-sklearns-gridsearchcv-best-score. [Accessed: 21-Apr-2021].
R. Joseph, “Grid Search for model tuning,” Medium, 29-Dec-2018. [Online]. Available: https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e. [Accessed: 21-Apr-2021].
Shahul ES Freelance Data Scientist | Kaggle Master Data science professional with a strong end to end data science/machine learning and deep learning (NLP) skills. Experienced working in a Data Science/ML Engineer role in multiple startups. K, S. ES, Freelance Data Scientist | Kaggle Master Data science professional with a strong end to end data science/machine learning and deep learning (NLP) skills. Experienced working in a Data Science/ML Engineer role in multiple startups. Kaggle Kernels Master ra, and F. me on, “Hyperparameter Tuning in Python: a Complete Guide 2021,” neptune.ai, 19-Mar-2021. [Online]. Available: https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020#:~:text=Hyperparameter%20tuning%20is%20the%20process,maximum%20performance%20out%20of%20models. [Accessed: 21-Apr-2021].
J. Brownlee, “Hyperparameter Optimization With Random Search and Grid Search,” Machine Learning Mastery, 18-Sep-2020. [Online]. Available: https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/. [Accessed: 21-Apr-2021].
“sklearn.model_selection.GridSearchCV¶,” scikit. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. [Accessed: 21-Apr-2021].
M. Sharma, “Grid Search for Hyperparameter Tuning,” Medium, 21-Mar-2020. [Online]. Available: https://towardsdatascience.com/grid-search-for-hyperparameter-tuning-9f63945e8fec. [Accessed: 21-Apr-2021].
R. Meinert, “Optimizing Hyperparameters in Random Forest Classification,” Medium, 07-Jun-2019. [Online]. Available: https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6. [Accessed: 21-Apr-2021].
“sklearn.ensemble.randomforestclassifier¶.” [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. [Accessed: 21-Apr-2021]. 




Thank you for dropping by.
