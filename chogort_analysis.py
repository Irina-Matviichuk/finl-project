#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


path = '/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/KC_case_data .csv'


# In[3]:


df = pd.read_csv(path)


# In[4]:


df


# In[5]:


df['date'] = pd.to_datetime(df['date'])


# In[6]:


df.dtypes


# In[7]:


df.shape


# In[8]:


df_feb = df.query('date>="2020-02-01" and date<="2020-02-28"').groupby('date', as_index=False).agg({'device_id':'nunique'})


# In[9]:


df_feb.head(5)


# In[10]:


df['month'] = df['date'].dt.month


# In[11]:


df.head(5)


# In[12]:


df_feb_2 = df.groupby('month', as_index=False).agg({'device_id':'nunique'})


# In[13]:


df_feb_2.head(5)


# In[14]:


df_jan = df.query('month=="1" and event=="app_install"')    .groupby('month', as_index=False).agg({'device_id':'nunique'})


# In[15]:


df_jan


# In[16]:


# создаём таблицу только с установками
cog = df.query('event == "app_install"')


# In[17]:


cog=cog[['date','device_id']]


# In[18]:


# проверяем, нет ли дуюлей по инсталам
cog.groupby('device_id', as_index=False)     .agg({'date':'count'})     .sort_values('date', ascending=True)


# In[19]:


cog=cog.rename(columns={'date':'Когорта'})


# In[20]:


# смердживаем когорты по установке с исходной таблицей
df=df.merge(cog, how='left', on='device_id')


# In[21]:


df.head(5)


# In[22]:


#mindate = df.sort_values('date').drop_duplicates('device_id')


# In[23]:


#mindate.head(5)


# In[24]:


#mindate = mindate[['device_id', 'date']]


# In[25]:


#mindate = mindate.rename(columns = {'date':'Когорта'})


# In[26]:


#df = df.merge(mindate, how = 'left', on = 'device_id')


# In[27]:


#df.head(5)


# In[28]:


# создаём таблицу по покупкам
purchase= df.query('event=="purchase"')


# In[29]:


purchase.head(5)


# In[30]:


# оставляем только 2 колонки
purchase=purchase[['date','device_id']]


# In[31]:


# оставляем только первые покупки
purchase=purchase.sort_values('date').drop_duplicates('device_id')


# In[32]:


purchase=purchase.rename(columns=({'date':'Первая_покупка'}))
purchase


# In[33]:


# смердживаем с главной таблицей
df=df.merge(purchase,how='left',on='device_id')


# In[34]:


df.head(5)


# In[35]:


# считаем разницу между первой покупкой и когортой
df['daydiff']=((df['Первая_покупка']-df['Когорта'])).dt.days


# In[36]:


df.head(5)


# In[37]:


# делаем метку, если datediff<=7
df['conversion']= np.where(df['daydiff']<=7, 'yes', 'no')
df


# In[38]:


# создаём таблицу с численностью когорт
final1=df.groupby('Когорта', as_index=False)     .agg({'device_id':'nunique'})


# In[39]:


final1.head(5)


# In[40]:


# создаём таблицу с кол-вом сконвертировавшихся юзеров по когортам
final2=df.query('conversion=="yes"').groupby('Когорта', as_index=False)     .agg({'device_id':'nunique'})


# In[41]:


final2.head(5)


# In[42]:


# объединяем таблицы
final1=final1.merge(final2, how='left',on='Когорта')


# In[43]:


final1.head(5)


# In[44]:


#считаем конверсию
final1['CR']=(final1.device_id_y/final1.device_id_x*100).round(1)


# In[45]:


final1.head(5)


# In[46]:


# сортируем по убыванию
final1=final1.sort_values('CR', ascending=False)


# In[47]:


final1.head(5)


# In[ ]:





# In[48]:


# создаём таблицу с численностью когорт
final3=df.groupby(['Когорта', 'utm_source'], as_index=False)     .agg({'device_id':'nunique'})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


#С какого платного маркетингового канала пришло больше всего новых пользователей? 


# In[50]:


df.head(5)


# In[51]:


df_purchase= df.query('event=="purchase"')
df_purchase.head(5)


# In[52]:


# оставляем только 2 колонки
df_purchase=df_purchase[['date','device_id']]


# In[53]:


# проверяем, нет ли дуюлей по инсталам
df_purchase.groupby('device_id', as_index=False)     .agg({'date':'count'})     .sort_values('date', ascending=True)


# In[54]:


#Для того, чтобы выделить группу тех, кому нужно и не нужно регистрироваться, добавьте колонку с датой регистрации. 
#Если дата регистрации < даты совершения события, то пользователь уже зарегистрирован. 


# In[55]:


df.head(5)


# In[56]:


# создаём таблицу только с регистрациями
reg = df.query('event == "register"')


# In[57]:


reg=reg[['date','device_id']]
reg.head(5)


# In[58]:


# оставляем только первые reg
reg=reg.sort_values('date').drop_duplicates('device_id')


# In[59]:


reg=reg.rename(columns=({'date':'Reg_Date'}))
reg


# In[60]:


# смердживаем с главной таблицей
df=df.merge(reg,how='left',on='device_id')
df.head(5)


# In[61]:


df_reg = df.query('Reg_Date<date')
df_reg.head(5)


# In[ ]:





# In[ ]:





# In[62]:


# создаём таблицу только с поиском
search = df.query('event == "search"')


# In[63]:


search=search[['date','device_id']]


# In[64]:


search=search.rename(columns={'date':'Поиск'})


# In[65]:


# смердживаем когорты по поиску с исходной таблицей
df=df.merge(search, how='left', on='device_id')


# In[66]:


df.head(5)


# In[67]:


# создаём таблицу с численностью когорт
final1=df.groupby('Когорта', as_index=False)     .agg({'device_id':'nunique'})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


df_ch = df.groupby('utm_source').agg({'device_id':'nunique'}).sort_values('device_id', ascending=False)


# In[69]:


df_ch.head(5)


# In[ ]:





# In[ ]:




