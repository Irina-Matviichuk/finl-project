#!/usr/bin/env python
# coding: utf-8

# In[247]:


# загрузка библиотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[248]:


# считываем данные


# In[249]:


path1 = ('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/assessments.csv')


# In[250]:


df_assessments = pd.read_csv(path1)


# In[251]:


df_assessments.head(5)


# In[252]:


path2 = ('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/courses.csv')


# In[253]:


df_courses = pd.read_csv(path2)


# In[254]:


df_courses.head(5)


# In[255]:


path3 = ('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/studentAssessment.csv')


# In[256]:


df_studentAssessment = pd.read_csv(path3)


# In[257]:


df_studentAssessment.head(5)


# In[258]:


path4 = ('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-matvijchuk-28/studentRegistration.csv')


# In[259]:


df_studentRegistration = pd.read_csv(path4)


# In[260]:


df_studentRegistration.head(5)


# In[261]:


df_assessments.nunique()


# In[262]:


#видим 7 уникальных предметов, 4 семестра, 206 уникальных испытаний с 3 типами оценок, дата испытания, вес испытания


# In[263]:


df_assessments.code_presentation.unique() #семестры


# In[264]:


df_assessments.assessment_type.unique() #тип теста


# In[265]:


df_assessments.code_module.unique() #список предметов


# In[266]:


#выберем все типы теста - Экзамен


# In[267]:


df_exam = df_assessments.query('assessment_type == "Exam"')
df_exam.head(5)


# In[268]:


df_exam.shape #всего 24 экзамена


# In[269]:


df_studentAssessment.id_student.nunique()  #уникальных студентов


# In[270]:


#Объединим датафреймы df_studentAssessment и df_assessments по полю id_assessment


# In[271]:


df_student_merge = df_studentAssessment.merge(df_assessments, on='id_assessment')
df_student_merge.head(5)


# In[272]:


#видим, какие тесты по какому предмету сдавал каждый студент, в каком семестре, дата теста и когда именно,


# In[273]:


df_student_exam = df_student_merge.query('assessment_type == "Exam"')
df_student_exam.head(5)
#отфильтровали экзамены в новый датафрэйм


# In[274]:


df_student_exam.id_student.nunique() #уникальных студентов сдавали экзамены


# In[275]:


df_student_exam.id_assessment.nunique() #количество уникальных экзаменов


# In[276]:


df_student_exam.code_module.nunique() #количество уникальных предметов


# In[277]:


df_student_exam.code_presentation.nunique() #количество уникальных семестров


# In[278]:


#1.Сколько студентов успешно сдали только один курс?
#(Успешная сдача — это зачёт по курсу на экзамене) (7 баллов).


# In[279]:


#Курсом будем считать предмет (code_module).
#Курс считается завершенным, если экзамен по нему успешно сдан (score не меньше 40)


# In[280]:


df_student_exam.query('score >= 40')     .groupby('id_student', as_index = False)     .agg({'code_module': 'count'})     .code_module.value_counts()


# In[281]:


#Итого, 3802 студента успешно сдали один курс


# In[ ]:





# In[282]:


#2. Выяви самый сложный и самый простой экзамен:
#найди курсы и экзамены в рамках курса, которые обладают самой низкой и самой высокой завершаемостью. (5 баллов)


# In[283]:


# Сначала найдём количество экзаменов, которые завершились успешно


# In[284]:


successful_exams = df_student_merge.query('assessment_type == "Exam" & score >= 40')     .groupby(['code_module', 'id_assessment', 'code_presentation'], as_index=False)     .agg({'assessment_type' : 'count'})
successful_exams


# In[285]:


# Найдём кол-во всех попыток сдать экзамен


# In[286]:


all_exams = df_student_merge.query('assessment_type =="Exam"')     .groupby(['code_module', 'id_assessment', 'code_presentation'], as_index=False)     .agg({'assessment_type' : 'count'})
all_exams


# In[287]:


all_exams['completion'] = ((successful_exams.assessment_type / all_exams.assessment_type) * 100).round(1)
all_exams


# In[288]:


all_exams.sort_values('completion')


# In[289]:


#Самый сложный экзамен - 25340 по предмету DDD в семестре 2013B


# In[290]:


#Самый простой экзамен - 25361 по предмету DDD в семестре 2014B


# In[291]:


all_module = df_student_merge.query('assessment_type =="Exam"')     .groupby('code_module', as_index=False)     .agg({'assessment_type' : 'count'})
all_module


# In[292]:


all_module['completion'] = ((successful_exams.assessment_type / all_module.assessment_type) * 100).round(1)
all_module


# In[293]:


#Курс DDD немного сложнее курса CCC


# In[ ]:





# In[294]:


#3. По каждому предмету определи средний срок сдачи экзаменов
#(под сдачей понимаем последнее успешное прохождение экзамена студентом). (5 баллов) 


# In[295]:


df_student_merge.query('assessment_type =="Exam"')     .groupby('code_module', as_index=False)     .agg({'date_submitted': 'mean'}).round(2)


# In[ ]:





# In[296]:


#4. Выяви самые популярные предметы (ТОП-3) по количеству регистраций на них.
#А также предметы с самым большим оттоком (ТОП-3). (8 баллов)


# In[297]:


top_registration = df_studentRegistration.groupby('code_module', as_index=False)     .agg({'id_student':'count'})     .sort_values('id_student', ascending=False)
top_registration.head(3)


# In[298]:


#Самые популярные курсы - BBB, FFF, DDD


# In[299]:


top_unregistration = df_studentRegistration.query('date_unregistration !="NaN"')     .groupby('code_module', as_index=False)     .agg({'id_student':'count'})     .sort_values('id_student', ascending=False)
top_unregistration.head(3)


# In[300]:


df_reg = top_registration.merge(top_unregistration, on='code_module')
df_reg.head(5)


# In[301]:


df_reg['churnrate'] = (df_reg.id_student_y / df_reg.id_student_x).round(2)
df_reg.sort_values('churnrate')


# In[302]:


#Курсы с самым большим оттоком - CCC, DDD, FFF 


# In[303]:


#5. Используя pandas, в период с начала 2013 по конец 2014 выяви семестр с самой низкой
#завершаемостью курсов и самыми долгими средними сроками сдачи курсов.  (10 баллов)


# In[304]:


all_exams.sort_values('completion')


# In[305]:


#Семестр с самой низкой завершаемостью курсов - 2013B


# In[306]:


df_student_merge.query('assessment_type =="Exam"')     .groupby('code_presentation', as_index=False)     .agg({'date_submitted': 'mean'})     .sort_values('date_submitted')     .round(2)


# In[307]:


# Семестр с самым долгим средним сроком сдачи курсов - 2014J


# In[ ]:





# In[343]:


#6. RFM анализ


# In[384]:


df_student_exam.head(5)


# In[385]:


#R - среднее время сдачи одного экзамена


# In[386]:


df_rfm = df_student_exam.groupby('id_student')     .agg({'date_submitted': 'mean'})     .reset_index()     .rename(columns={'date_submitted': 'r_score'})
df_rfm.head(5)


# In[387]:


#F - завершаемость курсов
# *завершаемость = кол-во успешных экзаменов / кол-во всех попыток сдать экзамен


# In[390]:


success_attempt = df_student_exam.query('score >= 40')     .groupby('id_student', as_index=False)     .agg({'code_module' : 'count'})     .sort_values('id_student')
success_attempt.head(5)


# In[391]:


all_attempts = df_student_exam.groupby('id_student', as_index=False)     .agg({'code_module' : 'count'})     .sort_values('id_student')
all_attempts.head(5)


# In[392]:


df_F = success_attempt.merge(all_attempts, how='inner', on='id_student')
df_F.head(5)


# In[393]:


df_F['f_score'] = df_F.code_module_x / df_F.code_module_y
df_F.head(5)


# In[394]:


df_rfm = df_rfm.merge(df_F, on='id_student', how='inner')
df_rfm.head(5)


# In[395]:


df_rfm = df_rfm.drop(columns={'code_module_x', 'code_module_y'}, axis=1)  #удалили ненужные колонки
df_rfm.head(5)


# In[396]:


# M - среднее количество баллов, получаемое за экзамен


# In[397]:


df_M = df_student_exam.groupby('id_student')     .agg({'score': 'mean'})     .reset_index()     .rename(columns={'score': 'm_score'})
df_M.head(5)


# In[398]:


df_rfm = df_rfm.merge(df_M, on='id_student', how='inner')
df_rfm.head(5)


# In[399]:


#посчитаем квантили для r_score, f_score, m_score


# In[400]:


quintiles = df_rfm[['r_score', 'f_score', 'm_score']].quantile([.2, .4, .6, .8])
quintiles


# In[401]:


# функция для r_score
def func_r_score(x):
# чем меньше число дней тем метрика лучше, поэтому в обратном порядке
    if x <= 231:
        return 5
    elif x <= 237:
        return 4
    elif x <= 242:
        return 3
    elif x <= 243.5:
        return 2
    else:
        return 1


# In[402]:


# функция для m_score  для m_score логично первый порог взять 40 - порог сдачи экзамена, 84 - это отлично
# итоговые границы: 40, 62, 84
def func_m_score(x):
    if x <= 40:
        return 1
    elif x <= 62:
        return 2
    elif x <= 84:
        return 3
    else:
        return 4


# In[403]:


# функция для f_score
def func_f_score(x):
    d = {0 : 1, 1: 2, 2: 3}
    return x + 1


# In[404]:


# применим к df_rfm функции для r_score, m_score, f_score и создадим колонку rfm_score


# In[405]:


df_rfm['r_score'] = df_rfm.r_score.apply(func_r_score)
df_rfm['m_score'] = df_rfm.m_score.apply(func_m_score)
df_rfm['f_score'] = df_rfm.f_score.apply(func_f_score)


# In[428]:


df_rfm['rfm_score'] = df_rfm.r_score * 100 + df_rfm.f_score * 10 + df_rfm.m_score
df_rfm


# In[437]:


df_rfm.groupby('rfm_score')     .agg({'id_student': 'count'})     .rename(columns={'id_student': 'count'})     .reset_index().sort_values('count', ascending = False)


# In[ ]:





# In[ ]:





# In[ ]:




