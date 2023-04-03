#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np


# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['image.cmap'] = 'viridis'


# In[51]:


path = '/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/RFM_ht_data.csv'


# In[52]:


df = pd.read_csv(path)


# In[53]:


df.head(5)


# In[59]:


df['CustomerCode'] = df['CustomerCode'].apply(str)
df['InvoiceNo'] = df['InvoiceNo'].apply(str)


# In[55]:


df.groupby('CustomerCode').agg({'InvoiceNo':'count'}).sort_values('InvoiceNo', ascending=False)


# In[60]:


df.dtypes


# In[61]:


df.shape


# In[62]:


df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)


# In[63]:


df.dtypes


# In[9]:


df.InvoiceDate.min()


# In[10]:


df.InvoiceDate.max()


# In[ ]:


mindate = df.sort_values('InvoiceDate').drop_duplicates()


# In[64]:


df.InvoiceDate.describe()


# In[65]:


last_date = df.InvoiceDate.max()
last_date


# In[66]:


rfmTable = df.groupby('CustomerCode').agg({'InvoiceDate': lambda x: (last_date - x.max()).days, # Recency #Количество дней с последнего заказа
                                        'InvoiceNo': lambda x: len(x),      # Frequency #Количество заказов
                                        'Amount': lambda x: x.sum()}) # Monetary Value #Общая сумма по всем заказам

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'Amount': 'monetary_value'}, inplace=True)


# In[69]:


rfmTable.head()


# In[70]:


rfmTable.shape[0]


# In[71]:


df.CustomerCode.nunique()


# In[72]:


quantiles = rfmTable.quantile(q=[0.25, 0.5, 0.75])


# In[73]:


quantiles


# In[74]:


rfmSegmentation = rfmTable


# In[75]:


def RClass(value,parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 1
    elif value <= quantiles_table[parameter_name][0.50]:
        return 2
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 3
    else:
        return 4


def FMClass(value, parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 4
    elif value <= quantiles_table[parameter_name][0.50]:
        return 3
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 2
    else:
        return 1


# In[76]:


rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles))

rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles))

rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles))

rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str)                             + rfmSegmentation.F_Quartile.map(str)                             + rfmSegmentation.M_Quartile.map(str)


# In[97]:


rfmSegmentation.head(5)


# In[104]:


rfm_new = rfmSegmentation.reset_index()
rfm_new


# In[100]:


rfm_new.dtypes


# In[103]:


rfm_new['CustomerCode'] = rfm_new['CustomerCode'].apply(str)


# In[107]:


rfm_new.groupby('RFMClass').agg({'CustomerCode':'nunique'}).sort_values('CustomerCode', ascending=True)


# In[ ]:





# In[102]:


rfmSegmentation.groupby('RFMClass').agg({'CustomerCode':'nunique'})


# In[93]:


first_order_date = df.groupby('CustomerCode')['InvoiceDate'].min().reset_index(name='first_order_date')


# In[78]:


pd.crosstab(index = rfmSegmentation.R_Quartile, columns = rfmSegmentation.M_Quartile)


# In[79]:


rfm_table = rfmSegmentation.pivot_table(
                        index='R_Quartile', 
                        columns='F_Quartile', 
                        values='monetary_value', 
                        aggfunc=np.median).applymap(int)
sns.heatmap(rfm_table, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=4.15, annot_kws={"size": 10},yticklabels=4);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




