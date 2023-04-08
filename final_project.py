#!/usr/bin/env python
# coding: utf-8

# # Проект: вариант 1

# Представьте, что вы работаете в компании, которая разрабатывает мобильные игры. К вам пришел менеджер с рядом задач по исследованию нескольких аспектов мобильного приложения:

# # Задание 1

# Retention – один из самых важных показателей в компании. Ваша задача – написать функцию, которая будет считать retention игроков (по дням от даты регистрации игрока).

# In[2]:


#Импортируем необходимые библиотеки
    
import pandas as pd
import numpy as np
import scipy.stats as ss

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from scipy.stats import norm, shapiro
from scipy import stats
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm
plt.style.use('ggplot')
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Считываем данные
reg_data = pd.read_csv('~/shared/problem1-reg_data.csv', sep=';')
auth_data = pd.read_csv('~/shared/problem1-auth_data.csv', sep =';')


# In[4]:


# Смотрим на структуру данных в таблицах
auth_data.head()


# In[5]:


reg_data.head()


# In[6]:


# Проверяем тип данных в таблицах
reg_data.dtypes


# In[7]:


auth_data.dtypes


# In[8]:


# Смотрим на размеры таблиц
auth_data.shape


# In[9]:


reg_data.shape


# In[10]:


# Проверим на наличие пропущенных значений
auth_data.isna().sum()


# In[11]:


reg_data.isna().sum()


# In[12]:


# Переводим в формат даты
reg_data['reg_date'] = pd.to_datetime(reg_data['reg_ts'], unit='s').dt.normalize()
auth_data['auth_date'] = pd.to_datetime(auth_data['auth_ts'], unit='s').dt.normalize()


# In[13]:


# Смотрим на данные в таблицах
reg_data.head()


# In[14]:


auth_data.head()


# In[15]:


reg_data.dtypes


# In[16]:


auth_data.dtypes


# In[17]:


# Проверим датасет с регистрациями на уникальность User ID
reg_data.uid.nunique()


# Размер сходится с размером изначального датасета, соответственно User ID уникальны

# In[18]:


#Посмотрим на описательную статистику для даты регистрации
reg_data.reg_date.describe(datetime_is_numeric=True)


# In[19]:


# Напишем функцию для расчета Retention игроков по дням с момента регистрации для заданного промежутка времени

def day_retention(start_date, end_date, df_reg=reg_data, df_auth=auth_data): 
    
    df_reg  = df_reg.query("@start_date <= reg_date <= @end_date")   
    df_auth = df_auth.query("@start_date <= auth_date <= @end_date")
    # Cмержим датасеты
    df_merged = df_auth.merge(df_reg, on='uid')[['uid', 'reg_date', 'auth_date']]  
    df_merged['days'] = (df_merged.auth_date - df_merged.reg_date).dt.days           
    df_merged.reg_date = df_merged.reg_date.astype('str')
    df_merged['reg_date'] = df_merged['reg_date'].apply(lambda x: x[:10])
    # Добавим столбец для сводной таблицы
    df_merged['retention'] = 1
    
    # Создадим сводную таблицу
    df_pivot_table = pd.pivot_table(df_merged,
                                     index='reg_date',
                                     columns='days',
                                     values='retention',
                                     aggfunc=pd.Series.sum
                                    )   
    # Вычислим ретеншн
    pivot_table_ret = df_pivot_table.div(df_pivot_table[0], axis=0)
    pivot_table_ret = pivot_table_ret.drop([0], axis=1)
    
    # Отформатируем финальную таблицу
    pivot_table_ret = (pivot_table_ret
            .style
            .set_caption('Retention by cohort')  # добавим подпись
            .background_gradient(cmap='Purples')  # раскрасим ячейки
            .highlight_null('white')  # белый фон для значений NaN
            .format("{:.2%}", na_rep=""))  # числа отформатируем как проценты, NaN заменим на пустоту
    
    return pivot_table_ret    


# In[20]:


#Рассчитываем Retention для периода
day_retention("2020-09-01", "2020-09-15")


# # Задание 2

# Имеются результаты A/B теста, в котором двум группам пользователей предлагались различные наборы акционных предложений. Известно, что ARPU в тестовой группе выше на 5%, чем в контрольной. При этом в контрольной группе 1928 игроков из 202103 оказались платящими, а в тестовой – 1805 из 202667.
# 
# Какой набор предложений можно считать лучшим? Какие метрики стоит проанализировать для принятия правильного решения и как?

# In[21]:


# Считываем данные
df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/Проект_1_Задание_2.csv', sep=';')


# In[22]:


df.head()


# Для начала посмотрим на основные метрики для контрольной и тестовых групп

# In[23]:


df_group_a = df.query('testgroup=="a"')
df_group_b = df.query('testgroup=="b"')


# In[24]:


df_group_a.shape


# In[25]:


payers_a=df_group_a.query('revenue> 0')
payers_b=df_group_b.query('revenue> 0')


# In[26]:


payers_a.head()


# In[27]:


payers_a.shape


# Посчитаем конверсию в плательщика в группах a и b

# In[28]:


cr_group_a = round(payers_a.shape[0]/df_group_a.shape[0]*100, 2)
cr_group_a


# In[29]:


cr_group_b = round(payers_b.shape[0]/df_group_b.shape[0]*100, 2)
cr_group_b


# Видим, что конверсия в плательщика в группе a больше, чем конверсия в плательщика в группе b

# Посчитаем минимальный, максимальный, средний и медианный чеки в группах a и b

# In[30]:


payers_a.revenue.min()


# In[31]:


payers_b.revenue.min()


# In[32]:


payers_a.revenue.max()


# In[33]:


payers_b.revenue.max()


# In[34]:


payers_a.revenue.mean().round(2)


# In[35]:


payers_b.revenue.mean().round(2)


# In[36]:


payers_a.revenue.median()


# In[37]:


payers_b.revenue.median()


# Посчитаем ARPU и ARPPU для обоих групп

# In[38]:


total_rev_a = payers_a.revenue.sum()
total_rev_a


# In[39]:


arpu_a =round(total_rev_a/df_group_a.shape[0],2)
arpu_a


# In[40]:


total_rev_b = payers_b.revenue.sum()
total_rev_b


# In[41]:


arpu_b =round(total_rev_b/df_group_b.shape[0],2)
arpu_b


# In[42]:


arppu_a=round(total_rev_a/payers_a.shape[0], 2)
arppu_a


# In[43]:


arppu_b=round(total_rev_b/payers_b.shape[0], 2)
arppu_b


# Теперь посмотрим на форму распределения в обеих группах (проверим нормальность распределения)

# In[44]:


#группа a
sns.histplot(df_group_a.revenue, kde = True, bins=10)


# In[45]:


#группа b
sns.histplot(df_group_b.revenue, kde = True, bins=10)


# In[46]:


import scipy.integrate as integrate
import scipy
from scipy import stats


# In[47]:


scipy.stats.normaltest(df_group_a.revenue)


# In[48]:


stats.shapiro(df_group_a.revenue.sample(1000, random_state=15))


# In[49]:


scipy.stats.normaltest(df_group_b.revenue)


# In[50]:


stats.shapiro(df_group_b.revenue.sample(1000, random_state=15))


# Видим, что распределение выборок в обеих группах не является нормальным, для дальнейшего анализа воспользуемся методом bootstrap.
# 
# Сформулируем гипотезы:
# 
# Н0 - статистически значимая разница в средних значениях в обоих группах отсутствует (при p > 0.05)
# 
# Н1 - статистически значимой разницы в средних значениях в обоих группах нет (при p < 0.05)

# In[51]:


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


# Проанализируем данные по всем пользователям

# In[52]:


booted_data_mean = get_bootstrap(df_group_a.revenue, 
                                 df_group_b.revenue, 
                                 boot_it = 1000, 
                                 statistic = np.mean)


# In[53]:


booted_data_mean["p_value"] # альфа


# In[54]:


booted_data_mean["quants"] # ДИ


# Проанализируем данные по платящим пользователям

# In[55]:


booted_data_mean_payers = get_bootstrap(payers_a.revenue, 
                                 payers_b.revenue, 
                                 boot_it = 1000, 
                                 statistic = np.mean)


# In[56]:


booted_data_mean_payers["p_value"] # альфа


# In[57]:


booted_data_mean_payers["quants"] # ДИ


# Как мы видим, p-value > 0.05 для всех пользователей (и для платящих пользователей в частности), при этом 0 входит в доверительный интервал.
# Таким образом, мы не можем отклонить H0 (статистически значимая разница в средних значениях в обоих группах отсутствует). То есть, различий между средними в группах нет.

# Проанализировав метрики в обеих группах, можем сделать вывод, что между группами есть разница в количестве платящих пользователей, минимальном и максимальном чеке.

# Таким образом, мы можем сделать финальный вывод, что ни одно из акционных предложений статистически значимо не повлияло на выручку компании.

# # Задание 3

# 3.1 В игре Plants & Gardens каждый месяц проводятся тематические события, ограниченные по времени. В них игроки могут получить уникальные предметы для сада и персонажей, дополнительные монеты или бонусы. Для получения награды требуется пройти ряд уровней за определенное время. С помощью каких метрик можно оценить результаты последнего прошедшего события?
# 

# При любом анализе метрик важно определить срезы (сегменты), в разрезе которых будем анализировать метрики.
# Я бы анализировала метрики по таким сегментам, как:
# 1. Новые пользователи / Пользователи, повторно установившие игру
# 2. Пол (м/ж/na/uncategorized)
# 3. Возраст
# 4. Гео
# 
# Опционально:
# 4. Уровни игры
# 5. Модели девайсов
# 6. Операторы
# 
# Метрики для анализа:
# - DAU
# - MAU
# - EAU (Event Active Users)
# - ARPU
# - ARPpU
# - Avg.Длина сессии
# - Avg.Количество сессий за день/период
# - Avg.Количество возвратов в игру в течение дня/периода
# - Avg.Количество пройденных уровней за день/период
# - Последний пройденный уровень за определенное время
# - CR
# - Retention 1d, 3d, 7d, 30d
# - ROAS 7d, 14d

# 3.2 Предположим, в другом событии мы усложнили механику событий так, что при каждой неудачной попытке выполнения уровня игрок будет откатываться на несколько уровней назад. Изменится ли набор метрик оценки результата? Если да, то как?

# При усложнении механики событий и введении системы откатов дополнительно следует проанализировать:
# 1. Churn rate - процент отвалившихся игроков
# 2. Уровень, на котором большинство игроков отваливаются (возможно слишком сложный)
# 3. Среднее количество откатов
# 4. Процент игроков, которые не откатились ни разу за период
