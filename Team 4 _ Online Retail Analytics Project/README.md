# INFO7374-DigitalMarketingAnalytics


### Online Retail Analytics 

webapp: https://tkmaxx.herokuapp.com/

Project Final Document: https://docs.google.com/document/d/1jf4OKJ9X-Iot08j9lNOgUb6IBWwkj8ydd0Nxym1KV9c/edit?usp=sharing

## Customer Segmentation

Why: As every customer is unique and can be targeted in different ways. The Customer segmentation plays an important role in this case.

How: So we have created the customer segmentation as Low, Mid and High clusers based on the overall value of how recent the customer is, what is the frequency and how much revenue is earned from the particular customer.
This is calculated by using RFM model.

## Cross Selling

Why: Cross selling is the process of recommending customers to buy related or complementary items to the items already have bought. It has applications in areas such ecommerce, product placement, fraud detection etc.

How: Using Market Basket Analysis, we can identify items which are frequently bought together from a list of transactions. We used FP growth algorithm for  extracting frequent itemsets.

## Customer Lifetime Value

Why: In the current world, almost every other retailer promotes its subscription and this is further used to understand the customer lifetime.  

How: In the project, we have used XGBClassifier model to predict the customer lifetime cluster which is categorized as LowCLV, MidCLV and HighCLV.

## Next Purchase Day


Why:Our objective is to analyze when our customer will purchase products in the future so for such customers we can build strategy and can come up with tactical solutions


How: For predicting next purchase day of our customers we developed KNN model and predicted our results in the form of classes where 
      
      class 0 represents Customers who Will Purchase in more than 50 days
      class 1 represents Customers who  Will Purchase in 21-49 days
      class 2 represents Customers who  Will Purchase in 0-20 days
      


