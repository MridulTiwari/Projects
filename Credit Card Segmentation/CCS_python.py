#!/usr/bin/env python
# coding: utf-8

# # Buisness Problem: Credit Card Segmentation

# In[1]:


# importing all the necesary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas_profiling # For Overview of data-summary statistics with plots


# These are most common library used use which ever necessary and explore it more

# In[2]:


# ! pip install --user pandas_profiling


# In[3]:


# !pip install --user joblib==0.14.1


# In[4]:


df = pd.read_csv('credit-card-data.csv')


# In[136]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


##make the column name in lower case so that it is easy to type
df.columns=df.columns.str.lower()


# In[12]:


df.describe().T


# before starting and preprocess to have a glance over your data set generally profiling wil be used
# 

# In[13]:


import pandas_profiling
pandas_profiling.ProfileReport(df)


# In[14]:


##Check for the outlier in column ehich have null values& make the function to check the outlier
def boxplot(value):
    return value.plot.box()


# In[15]:


boxplot(df['credit_limit'])


# In[16]:


boxplot(df['minimum_payments'])


# In[17]:


plt.plot(df['credit_limit'])


# In[18]:


plt.plot(df['minimum_payments'])


# In[19]:


df.head()


# In[20]:


##imputation of na value


# In[21]:


df['credit_limit'].median()


# In[22]:


df['credit_limit'].fillna(df['credit_limit'].median(),inplace=True)


# In[23]:


df.isna().sum()


# In[24]:


df['minimum_payments'].median()


# In[25]:


df['minimum_payments'].fillna(df['minimum_payments'].median(),inplace =True)


# In[26]:


df.isna().sum()


# In[27]:


##Visulaisation


# In[28]:


def boxplot(value):
    return value.plot.box()


# In[29]:


boxplot(df['balance']) ##We can see there are many outlier


# In[30]:


plt.plot(df['balance'])


# In[31]:


boxplot(df['balance_frequency'])


# In[32]:


plt.plot(df['balance_frequency'])


# In[33]:


boxplot(df['purchases'])


# In[34]:


plt.plot(df['purchases'])


# In[35]:


boxplot(df['oneoff_purchases'])


# In[36]:


plt.plot(df['oneoff_purchases'])


# In[37]:


boxplot(df['installments_purchases'])


# In[38]:


plt.plot(df['installments_purchases'])


# In[39]:


boxplot(df['cash_advance'])


# In[40]:


plt.plot(df['cash_advance'])


# In[41]:


boxplot(df['purchases_frequency']) ## In this variable there is no outlier


# In[42]:


plt.plot(df['purchases_frequency'])


# In[43]:


boxplot(df['oneoff_purchases_frequency'])


# In[44]:


plt.plot(df['oneoff_purchases_frequency'])


# In[45]:


boxplot(df['purchases_installments_frequency'])


# In[46]:


plt.plot(df['purchases_installments_frequency'])


# In[47]:


boxplot(df['cash_advance_frequency'])


# In[48]:


plt.plot(df['cash_advance_trx'])


# In[49]:


boxplot(df['purchases_trx'])


# In[50]:


plt.plot(df['purchases_trx'])


# In[51]:


boxplot(df['credit_limit'])


# In[52]:


plt.plot(df['credit_limit'])


# In[53]:


boxplot(df['payments'])


# In[54]:


plt.plot(df['payments'])


# In[55]:


boxplot(df['minimum_payments'])


# In[56]:


plt.plot(df['minimum_payments'])


# In[57]:


boxplot(df['prc_full_payment'])


# In[58]:


plt.plot(df['prc_full_payment'])


# In[59]:


boxplot(df['tenure'])


# In[60]:


plt.plot(df['tenure'])


# # Building KPI(Key Performance Indicator)

# In[61]:


df['monthly_avg_purchase']=df['purchases']/df['tenure']
df['monthly_avg_purchase'].head()


# In[62]:


df['monthly_cash_advance']=df['cash_advance']/df['tenure']
df['monthly_cash_advance'].head()


# In[63]:


df['oneoff_purchases'].value_counts() ##record of oneoff_purchase equal to zero most frequent


# # Puchase Type

# In[64]:


#for customers who do only oneoff_purchases
df[(df['oneoff_purchases']>0) & (df['installments_purchases']==0)].shape


# In[65]:


#For customers who do only installment purchases
df[(df['oneoff_purchases']==0) & (df['installments_purchases']>0)].shape


# In[66]:


#For the customers who do both one-off purchases and installment purchases
df[(df['oneoff_purchases']>0) & (df['installments_purchases']>0)].shape


# In[67]:


#For the customers neither do one-off purchases nor installment purchases
df[(df['oneoff_purchases']==0) & (df['installments_purchases']==0)].shape

Insights and  observation
There are four type of customer in the entire dataset on the basis of transaction 
1.Customer who do only oneoff_purchase transactions
2.Customer who do only installments purchase Transactions
3.Customer who do both oneoff and installments transactions
4.Customer who neither do oneff and nor installments transactions 
# In[68]:


#Writing a function for all the transaction types
def transaction(df):  
    if (df['oneoff_purchases']>0) & (df['installments_purchases']==0):
        return 'one_off'
    if (df['oneoff_purchases']==0) & (df['installments_purchases']>0):
        return 'installment'
    if (df['oneoff_purchases']>0) & (df['installments_purchases']>0):
         return 'both'
    if (df['oneoff_purchases']==0) & (df['installments_purchases']==0):
        return 'none'


# In[69]:


#Creating label type for the transaction types
df['transaction_type']=df.apply(transaction,axis=1)
df['transaction_type'].value_counts()


# In[70]:


df['transaction_type'].value_counts().plot.bar()


# In[71]:


#Finding the limit usage for each customer
df['limit_usage']=df.apply(lambda x: x['balance']/x['credit_limit'], axis=1)
df['limit_usage'].head()


# In[72]:


#finding Payment to minimum payments Ratio
df['payment_minpay']=df.apply(lambda x:x['payments']/x['minimum_payments'],axis=1)
df['payment_minpay'].describe().T


# In[73]:


df.info()


# In[74]:


##log transformation
transform=df.drop(['cust_id','transaction_type'],axis=1).applymap(lambda x: np.log(x+1))


# In[75]:


transform.describe().T


# In[76]:


col=['balance','purchases','cash_advance','tenure','payments','minimum_payments','prc_full_payment','credit_limit']
transform_pre=transform[[x for x in transform.columns if x not in col ]]


# In[77]:


transform_pre.describe().T


# # Finding the insights frm the data
# 

# In[78]:



df[df['transaction_type']=='n']


# In[79]:


# Average payment_minpayment ratio for each purchse type.
x=df.groupby('transaction_type').apply(lambda x: np.mean(x['payment_minpay']))
type(x)
x.values


# In[80]:


df.groupby('transaction_type').apply(lambda x: np.mean(x['payment_minpay'])).plot.barh()
plt.title('Average minimum payment ratio for each purchase type')
plt.xlabel('minimum payment')


# 
# Customers who make transactions in installments are paying the amount regularly

# In[81]:



df.groupby('transaction_type').apply(lambda x: np.mean(x['monthly_cash_advance'])).plot.barh()
plt.title('Average cash advance for each purchase type')
plt.xlabel('monthly cash advance')

2.Customers neither make a transaction in one_off payments nor installments are having high monthly cash advances
# In[82]:


df.groupby('transaction_type').apply(lambda x: np.mean(x['limit_usage'])).plot.barh()
plt.title('Average limit usage for each purchase type')
plt.xlabel('limit usage')


3.Less limit usage gives high credit score and the good score is with the customers who make transactions in installments
# # Dataset preparation set for model

# In[83]:


df_original=pd.concat([df,pd.get_dummies(df['transaction_type'])],axis=1)


# In[84]:


df_original.describe().T


# In[85]:


df.head()


# In[86]:


transform_pre['transaction_type']=df.loc[:,'transaction_type']


# In[87]:


transform_pre.head()


# In[88]:


df_dummy=pd.concat([transform_pre,pd.get_dummies(transform_pre['transaction_type'])],axis=1)
df_dummy.head()


# In[89]:


df_dummy=df_dummy.drop(['transaction_type'],axis=1)


# In[90]:


df_dummy.head()


# In[91]:


df_dummy.describe().T


# Finding the correlation among the variables in dataset

# In[92]:


sns.heatmap(df_dummy.corr())


# # Observations:
# The variables available for the model selection are very high in this dataset and this leads to dimensionality curse. In order to reduce the high dimensionality curse we use Principal Component Analysis technique, but before we use this we must make sure that the data available in the dataset have no weightage issues. So we use standard scaler technique if there are any weightage issues among the variables of the dataset.
choosing pca model
# In[93]:


from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()
df_scaled=sc.fit_transform(df_dummy)


# In[94]:


from sklearn.decomposition import PCA
var_ratio={}
for n in range(4,15):
    pc=PCA(n_components=n)
    df_pca=pc.fit(df_scaled)
    var_ratio[n]=sum(df_pca.explained_variance_ratio_)


# In[95]:


type(df_pca)


# In[96]:


var_ratio

Observation: From the above variance ratio we can see that the maximum variance is available when the number of components are 5. Hence we choose n_components as 5 to reduce the dimensionality in the datset
# In[97]:


pd.Series(var_ratio).plot()


# In[98]:


df_scaled.shape


# In[99]:


pc_final=PCA(n_components=5).fit(df_scaled)
reduced_df=pc_final.fit_transform(df_scaled)


# In[100]:


df1=pd.DataFrame(reduced_df)
df1.head()


# In[101]:


df1.shape


# In[102]:


col_list=df_dummy.columns
col_list


# In[103]:


pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(5)],index=col_list)


# In[104]:


# Factor Analysis : variance explained by each component- 
pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(5)])


# # Model Selection

# In[105]:


from sklearn.cluster import KMeans
km_4=KMeans(n_clusters=4,random_state=42)
km_4.fit(reduced_df)
km_4.labels_


# In[106]:


pd.Series(km_4.labels_).value_counts()


# In[107]:


color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in km_4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_df[:,0],reduced_df[:,1],c=label_color,cmap='Spectral',alpha=0.5)
plt.title('Clustering when number of components=4')


# In[108]:


df_pair_plot=pd.DataFrame(reduced_df,columns=['PC_' +str(i) for i in range(5)])
df_pair_plot['Cluster']=km_4.labels_
#pairwise relationship of components on the data
sns.pairplot(df_pair_plot,hue='Cluster', palette= 'Dark2', diag_kind='kde',height=2)


# # Observations:
# From the above graphs we can conclude that the only PC_0 and PC_1 are identifiable clusters and hence we go with further analysis by increasding the number of clusters value to identify more number of insights about the customers present in the dataset.# 

# In[109]:


# Key performace variable selection . here i am dropping varibales which are used in derving new KPI
col_kpi=['purchases_trx','monthly_avg_purchase','monthly_cash_advance','limit_usage','cash_advance_trx',
         'payment_minpay','both','installment','one_off','none','credit_limit']


# In[110]:


transform_pre.describe().T


# In[111]:


df_original.columns


# In[112]:


# Conactenating labels found through Kmeans with data 
cluster_df_4=pd.concat([df_original[col_kpi],pd.Series(km_4.labels_,name='Cluster_4')],axis=1)


# In[113]:


# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_4=cluster_df_4.groupby('Cluster_4').apply(lambda x: x[col_kpi].mean()).T
cluster_4


# In[114]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['monthly_cash_advance',:].values)
credit_score=(cluster_4.loc['limit_usage',:].values)
purchase= np.log(cluster_4.loc['monthly_avg_purchase',:].values)
payment=cluster_4.loc['payment_minpay',:].values
installment=cluster_4.loc['installment',:].values
one_off=cluster_4.loc['one_off',:].values


bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='one_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()

Observations: From the above graph we can see that the four clusters have been categorised perfectly so that the difference in each cluster can be understood
# In[115]:


# Percentage of each cluster in the total customer base
s=cluster_df_4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
print(s)

per=pd.Series((s.values.astype('float')/ cluster_df_4.shape[0])*100,name='Percentage')
print ("Cluster -4 ")
print (pd.concat([pd.Series(s.values,name='Size'),per],axis=1))


# # Exploring the insights if the number of cluster=5

# In[116]:


#kmeans with 5 clusters
km_5=KMeans(n_clusters=5,random_state=42)
km_5=km_5.fit(reduced_df)
km_5.labels_


# In[117]:


plt.figure(figsize=(7,7))
plt.scatter(reduced_df[:,0],reduced_df[:,1],c=km_5.labels_,cmap='Spectral',alpha=0.5)
plt.xlabel('PC_0')
plt.ylabel('PC_1')


# In[118]:


cluster_df_5=pd.concat([df_original[col_kpi],pd.Series(km_5.labels_,name='Cluster_5')],axis=1)


# In[119]:


# Finding Mean of features for each cluster
five_cluster=cluster_df_5.groupby('Cluster_5').apply(lambda x: x[col_kpi].mean()).T

five_cluster


# In[120]:


s1=cluster_df_5.groupby('Cluster_5').apply(lambda x: x['Cluster_5'].value_counts())
print(s1)


# In[121]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(five_cluster.columns))

cash_advance=np.log(five_cluster.loc['monthly_cash_advance',:].values)
credit_score=(five_cluster.loc['limit_usage',:].values)
purchase= np.log(five_cluster.loc['monthly_avg_purchase',:].values)
payment=five_cluster.loc['payment_minpay',:].values
installment=five_cluster.loc['installment',:].values
one_off=five_cluster.loc['one_off',:].values

bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='one_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3','Cl-4'))

Observations: From the above graph, we can't come to a particular conclusion regarding the behaviour of customer groups, because cluster 2 is having highest average purchases in the transactions, but at the same time cluster1 has highest cash advance and second highest purchases.
# In[122]:


# percentage of each cluster
print("Cluster-5")
per_5=pd.Series((s1.values.astype('float')/ cluster_df_5.shape[0])*100,name='Percentage')
print(pd.concat([pd.Series(s1.values,name='Size'),per_5],axis=1))


# # Exploring the insights when number of clusters=6¶# 

# In[123]:


km_6=KMeans(n_clusters=6).fit(reduced_df)
km_6.labels_


# In[124]:


color_map={0:'r',1:'b',2:'g',3:'c',4:'m',5:'k'}
label_color=[color_map[l] for l in km_6.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_df[:,0],reduced_df[:,1],c=label_color,cmap='Spectral',alpha=0.5)


# In[125]:


cluster_df_6=pd.concat([df_original[col_kpi],pd.Series(km_6.labels_,name='Cluster_6')],axis=1)


# In[126]:


six_cluster=cluster_df_6.groupby('Cluster_6').apply(lambda x: x[col_kpi].mean()).T
six_cluster


# In[127]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(six_cluster.columns))

cash_advance=np.log(six_cluster.loc['monthly_cash_advance',:].values)
credit_score=(six_cluster.loc['limit_usage',:].values)
purchase= np.log(six_cluster.loc['monthly_avg_purchase',:].values)
payment=six_cluster.loc['payment_minpay',:].values
installment=six_cluster.loc['installment',:].values
one_off=six_cluster.loc['one_off',:].values

bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='one_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3','Cl-4','Cl-5'))

Observations: From the above graph we can see that cluster 2 and cluster 4 have similar behavior regarding the parameters, hence distinguishing between the clusters is hard when we have the number of clusters as 6
# In[128]:


cash_advance=np.log(six_cluster.iloc[2,:].values)
credit_score=list(six_cluster.iloc[3,:].values)
print(cash_advance)
print(credit_score)


# # Metrics for the KMeans Model

# In[129]:


from sklearn.metrics import calinski_harabasz_score,silhouette_score
score={}
score_c={}
for n in range(3,10):
    km_score=KMeans(n_clusters=n)
    km_score.fit(reduced_df)
    score_c[n]=calinski_harabasz_score(reduced_df,km_score.labels_)
    score[n]=silhouette_score(reduced_df,km_score.labels_)


# In[130]:


print(score)


# In[131]:


pd.Series(score).plot()
plt.title('silhouette_score')


# In[132]:


pd.Series(score_c).plot()
plt.title('calinski_harabasz_score')


# In[133]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(reduced_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Observations:¶
# From all the above graphs we can conclude the performance of the KMeans Model regarding the explanation of data distribution and measure of spread is highest when we consider the number of cluster as four.

# # Final KMeans Model¶# 

# In[134]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['monthly_cash_advance',:].values)
credit_score=(cluster_4.loc['limit_usage',:].values)
purchase= np.log(cluster_4.loc['monthly_avg_purchase',:].values)
payment=cluster_4.loc['payment_minpay',:].values
installment=cluster_4.loc['installment',:].values
one_off=cluster_4.loc['one_off',:].values


bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='one_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()


# # 
# Marketing Strategies
# cluster 0:
# Customers belong to this cluster must be the primary focus regarding the marketing strategy because the customers under this cluster are making frequent purchases and also paying the dues on time thus maintaining good credit score. Customers in this cluster must be given with good reward points and provided with increased credit limit or the premium credit cards with some exciting offers make them do more transactions in the future.
# 
# cluster 1:
# Customers who fall under this category of cluster are having the best credit card and also paying the dues on time without defaults. Hence these group of customers must rewarded with reward points and thus make them do more transactions in future.
# 
# cluster 2:
# 
# Customers belong to this category of cluster having the highest cash advance and poor credit score yet these customers pay the due amounts of the installments on time.Hence these customers may be given with the loan amounts at less interest charges, thus help the banks providing continuous services to these group of customers in future
# 
# cluster3:
# Customers belong to this cluster has the least minimum payment ratio and always does the one off payment transactions, hence no bank offers can excite these kind of cutomers. The marketing to this group of customers is hard and when the usage  is minimum, this group can be ignored from the marketing strategy.Further the customers falling under this category can be rejected from issuing the credit cards in future.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




