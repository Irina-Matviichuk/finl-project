#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.integrate 
import scipy
from scipy import stats
from scipy.stats import iqr


# In[4]:


path = '/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/hw_bootstrap.csv'


# In[5]:


df = pd.read_csv(path, sep = ';')
df.head(5)


# In[6]:


df = df.drop(columns=['Unnamed: 0'])
df.head(5)


# In[7]:


df['value'] = df['value'].apply(lambda x: float (x.split()[0].replace(",", ".")))


# In[8]:


df['value'] = pd.to_numeric(df['value'])


# In[9]:


df['value'] = df['value'].astype(float)


# In[10]:


df.head(5)


# In[11]:


df.experimentVariant.nunique


# In[12]:


df.dtypes


# In[13]:


df.shape


# In[14]:


df.isna().sum()


# In[15]:


plt.figure(figsize=(6, 6))
df_control_hist = df.query('experimentVariant == "Control"').value.hist()


# In[16]:


plt.figure(figsize=(6, 6))
df_test_hist = df.query('experimentVariant == "Treatment"').value.hist()


# In[17]:


df_control = df.query('experimentVariant == "Control"')
df_test = df.query('experimentVariant == "Treatment"')


# In[18]:


sns.boxplot(y='value', data = df_test)


# Нужно избавиться от выбросов в тестовой группе

# In[20]:


Q1 = df_test.value.quantile(q=0.25)
Q1


# In[21]:


Q3 = df_test.value.quantile(q=0.75)
Q3


# In[22]:


IQR = Q3 - Q1
IQR


# In[23]:


df_IQR = df_test[(df_test['value'] > (Q1-1.5*IQR)) & (df_test['value'] < (Q3+1.5*IQR))]
df_IQR.boxplot(column="value")


# In[24]:


import scipy.integrate as integrate
import scipy
from scipy import stats


# In[25]:


scipy.stats.normaltest(df_control.value)


# In[26]:


scipy.stats.normaltest(df_test.value)


# t-test

# Формулируем H0 - отсутствие различий между средним value в тестовой и контрольной группах 
# 
# Формулируем H1 - наличие различий между средним value в тестовой и контрольной группах 

# In[49]:


scipy.stats.ttest_ind(df_control.value, df_IQR.value) #данные без выбросов


# In[50]:


scipy.stats.ttest_ind(df_control.value, df_test.value) #данные с выбросами


# In[51]:


sr_test_IQR = df_IQR.value.mean()
sr_test_IQR


# In[30]:


sr_control = df_control.value.mean()
sr_control


# In[52]:


sr_test = df_test.value.mean()
sr_test


# Поскольку p-value > 0.05, - принимаем H0 - средние value в тестовой и контрольной группах не отличаются
# 
# Однако, возникают сомнения в корректности изспользования t-test'a, поскольку:
# - число наблюдений довольно велико (1000)
# - не соблюдается требование к нормальности распределения данных в обеих группах (не повод не использовать t-test)
# - наличие выбросов в данных
# 
# Помним, что если распределение признака отличается от нормального,
# можно использовать непараметрический аналог – U-критерий Манна-Уитни.

# U-тест Mann-Whitney

# Этот критерий менее чувствителен к экстремальным отклонениям от нормальности и наличию выбросов.
# 
# Формулируем H0 - распределение value в тестовой и контрольной группах не отличаются
# 
# Формулируем H1 - распределение value в тестовой и контрольной группах отличаются

# In[36]:


scipy.stats.mannwhitneyu(df_test.value, df_control.value)


# Поскольку p-value > 0.05, - принимаем H0 - распределение value в тестовой и контрольной группах не отличаются

# BOOTSTRAP

# Бутстрап позволяет многократно извлекать подвыборки из выборки, полученной в рамках экспериментва.
# 
# В полученных подвыборках считаются статистики (среднее, медиана и т.п.).
# 
# Из статистик можно получить ее распределение и взять доверительный интервал.
# 
# ЦПТ, например, не позволяет строить доверительные интервал для медианы, а бутстрэп это может сделать

# In[43]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

plt.style.use('ggplot')


# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа

# In[44]:


def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[45]:


booted_data_mean = get_bootstrap(df_control.value, 
                                 df_test.value, 
                                 boot_it = 1000, 
                                 statistic = np.mean)


# In[46]:


booted_data_mean["p_value"] # альфа


# In[47]:


booted_data_mean["quants"] # ДИ


# Поскольку p-value > 0.05 и в доверительный интервал не входит 0, - принимаем H0 - средние значения value в тестовой и контрольной группах равны

# Мы получили синхронные выводы по трем разным тестам. Соответственно, делаем вывод, что средние value в тестовой и контрольной группах значимо не отличаются. 
# 
# При этом если получаем не синхронные выводы по разным тестам, нужно принять решение, 
# на какой тест опираться.
# Так, бутстрап хорошо работает с ненормальными распределениями.
# При этом Манн-Уитни и Бутстрап проверяют разные гипотезы. 
# 
# 
# Финализировав вывод, я бы сказала, что средние значения value в тестовой и контрольной группах
# не отличаются, при этом большие выбросы сильно искажают нам среднее значение.

# In[ ]:




