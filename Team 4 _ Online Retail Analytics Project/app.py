#Loading neccesary packages
import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
#import joblib
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

@st.cache(allow_output_mutation=True)
def load_data(dataset):
	df = pd.read_csv(dataset,encoding= 'unicode_escape')
	return df

def main():
    """Online Retail Analytics ML App"""
    
    st.title("Online Retail Analytics")
    #st.subheader("Streamlit ML App")
    
    activities = ['EDA','Prediction','About']
    choices = st.sidebar.selectbox("Select Activities",activities)
    data = load_data('data/dataset.csv')
    
    if choices == 'EDA':
        st.header("Exploratory Data Analysis")
        choice1 = st.sidebar.selectbox("Choose One:",["Show top 5 rows of data","Show Summary of Dataset","Customer Retention Rate","User Type Revenue"])
                                                      
        if choice1=="Show top 5 rows of data":
            st.write("Top 5 rows of data")
            st.dataframe(data.head(5))
        
        if choice1=="Show Summary of Dataset":
            st.write("Summary of Dataset")
            st.write(data.describe())
            
        if choice1=="Customer Retention Rate":
            st.write("Customer Retention Rate")
            ret_rate = pd.read_csv(os.getcwd()+"/data/retention_rate.csv")
            
            # Replace using dictionary
            ret_rate['InvoiceYearMonth'] = ret_rate['InvoiceYearMonth'].replace({
            201102:"Feb",
            201103:"Mar",
            201104:"Apr",
            201105:"May",
            201106:"June",
            201107:"July",
            201108:"Aug",
            201109:"Sep",
            201110:"Oct",
            201111:"Nov",
            201112:"Dec",
            })
            
            plt.plot( 'InvoiceYearMonth', 'TotalUserCount', data=ret_rate, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

            plt.plot( 'InvoiceYearMonth', 'RetainedUserCount', data=ret_rate,marker='o', markerfacecolor='red', markersize=12, color='tomato', linewidth=4)

            plt.legend()
            st.pyplot()
                
                        
        if choice1=="User Type Revenue":
            st.write("User Type Revenue")
            user_revenue = pd.read_csv(os.getcwd()+"/data/User_Type_Revenue.csv")
            user_revenue = user_revenue.drop(columns=['Unnamed: 0'])

#             user_revenue['InvoiceYearMonth'] = user_revenue['InvoiceYearMonth'].replace({
#                 201012:"Dec 2010",
#                 201101:"Jan 2011",
#                 201102:"Feb 2011",
#                 201103:"Mar 2011",
#                 201104:"Apr 2011",
#                 201105:"May 2011",
#                 201106:"June 2011",
#                 201107:"July 2011",
#                 201108:"Aug 2011",
#                 201109:"Sep 2011",
#                 201110:"Oct 2011",
#                 201111:"Nov 2011",
#                 201112:"Dec 2011",
#             })

            viz = sns.barplot(x="InvoiceYearMonth",y="Revenue", data = user_revenue,
                            hue='UserType')

            plt.xticks(rotation=-45)
            viz.set(ylabel='Revenue')
            plt.show()
            st.pyplot()
    
    if choices == 'Prediction':
        st.header("Prediction Analytics")
        choice = st.sidebar.selectbox("Choose One:",["Customer Segmentation","Cross Selling","Customer Lifetime Value","Next Purchase Day"])
        if choice =="Customer Segmentation":
            st.subheader("Customer Segmentation")
            st.write("Classifying Customers based on RFM Model")
            customer_segmentation = pd.read_csv(os.getcwd()+"/data/Customer_Segmentation.csv")
            customer_segmentation['CustomerID'] = customer_segmentation['CustomerID'].astype(int)
            customerID = st.selectbox('CustomerID',customer_segmentation['CustomerID'].head(100))
            if st.button("Submit"):
                selected_customer = customer_segmentation.loc[customer_segmentation['CustomerID']==customerID]
                st.write(selected_customer[['Segment','Recency','Frequency','Revenue']])
          
            
        if choice =="Customer Lifetime Value":
            st.subheader("Customer Lifetime Value")
            st.write("Predicting LTV using XGBoost classifier")
            clv = pd.read_csv(os.getcwd()+"/data/CLV.csv")
            clv['CustomerID'] = clv['CustomerID'].astype(int)
            customerID = st.selectbox('CustomerID',clv['CustomerID'].head(100))
            if st.button("Submit"):
                selected_customer = clv.loc[clv['CustomerID']==customerID]
                st.write(selected_customer['Customer_Lifetime_value'])   
        
        if choice =="Next Purchase Day":
            st.subheader("Next Purchase Day")
            st.write("Predict Next Purchase Day using KNN")
            st.write("NextPurchaseDayClass=0:Customer Will Purchase in more than 50 days")
            st.write("NextPurchaseDayClass=1:Customer Will Purchase in 21-49 days")
            st.write("NextPurchaseDayClass=2:Customer Will Purchase in 0-20 days")
            Next_pday=pd.read_csv(os.getcwd()+"/data/Next.csv",encoding= 'unicode_escape',index_col=False)
            Next_pday['Customer_Id']=Next_pday['Customer_Id'].astype('str')
            Customer=st.selectbox("Select CustomerID:",Next_pday['Customer_Id'])
            if st.button("Submit"):
                 st.write(Next_pday.loc[Next_pday['Customer_Id']==Customer,'NextPurchaseDayClass'])
        
        if choice =="Cross Selling":
            st.subheader("Cross Selling")
            st.write("Market Basket Analysis using FP Growth")
            #Reading Data From Web
            #data = load_data('data/dataset.csv')
            #input1= st.selectbox("Select min support",[0.1,0.2,0.3,0.4,0.5])
            #if st.button("Submit"):'
            #country = st.sidebar.selectbox("Choose a country",data['Country'].unique())
            #if st.sidebar.button("Submit1"):
                #st.write("You selected: ",country)

                #min_support=input1
                #st.write("You selected this option ",input1)

            #Cleaning
            data['Description'] = data['Description'].str.strip()
            data.dropna(axis=0, subset=['InvoiceNo'], inplace=True) 
            data['InvoiceNo'] = data['InvoiceNo'].astype('str')
            data = data[~data['InvoiceNo'].str.contains('C')] 
            data.head()

            #Separating transactions for Country
            basket = (data[data['Country'] =='Germany']
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))

            #converting to 1 and  0
            def encoder(x):
                if x <= 0:
                    return 0
                if x >= 1:
                    return 1

            basket = basket.applymap(encoder)
            basket.drop('POSTAGE', inplace=True, axis=1, errors='ignore')

            #Generatig frequent itemsets
            itemsets = fpgrowth(basket, min_support=0.07, use_colnames=True)

            #generating rules
            rules = association_rules(itemsets, metric="lift", min_threshold=1)

            #rules=rules[['antecedents','consequents']]
            #df=rules
            #df.columns = ['Input', 'Output']

            choice = st.selectbox("Choose One:",rules['antecedents'].head(100))
            if st.button("Submit"):
                output= rules.loc[rules['antecedents'] == choice]
                st.write(output[['consequents']])
                
                
    if choices == 'About':
        st.subheader("About")
        st.write("TK Maxx is a subsidiary of the American apparel and home goods company TJX Companies and offers customers across various  countries great values on brand name apparel and more, including high-end designer goods and juniors.")
        st.write("TK Maxx wants to analyse the customer transactions at their stores over an 8 month period to understand their behavior and make some predictions about their customer behavior. They also plan to use this data to cross-sell products which are frequently bought together.")
        st.write("We are building a web app to analyse the key metrics using various algorithms to segment customers,  predict the lifetime value of each customer, product recommendation and predict the next purchase date.")
        st.write("A marketing analyst at TK Maxx can use these insights to develop strategies like targeting users, identifying important customers , predict the amount to be spent on acquiring or retaining customers  and offer personalised recommendations")
    
if __name__ == '__main__':
    main()


            