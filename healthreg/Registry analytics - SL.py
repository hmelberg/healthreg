# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:54:36 2017

@author: sandresl_adm
"""
#%% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from collections import Counter
import re
import datetime as dt 
import seaborn as sns
%matplotlib inline 
sns.set_style('darkgrid')
#%% read
try:
    df = pd.read_csv("C:/Users/sandresl_adm/Google Drive/sandre/takeda/data/NPR 2015 oppdatert rensket.csv", encoding = "latin-1")
except:
    df = pd.read_csv("C:/Users/sandresl/Google Drive/sandre/takeda/data/NPR 2015 oppdatert rensket.csv", encoding = "latin-1")

#%% Update to english variable names
a = df.head(111)
new = ['Unnamed: 0', 'pid', 'k50h', 'k51h', 'k52h', 't81h', 'k50b', 'k51b',
       'k52b', 't81b', 'year', 'in_date', 'male', 'agegr', 'los', 'kilde',
       'region', 'institution_code', 'institution', 'hospital_code',
       'hospital', 'drg', 'procedure_codes', 'k50', 'k51', 'k52', 'disease']

old = ['Unnamed: 0', 'lopenr', 'k50h', 'k51h', 'k52h', 't81h', 'k50b', 'k51b',
       'k52b', 't81b', 'aar', 'inn_mnd', 'male', 'aldgrp', 'liggetid', 'kilde',
       'region', 'institusjon', 'institusjon_ny', 'behandlingssted',
       'behandlingssted_ny', 'drg', 'alt', 'k50', 'k51', 'k52', 'disease']

new_labels = dict(zip(old, new))
df = df.rename(columns=new_labels)

#%% Bio dummies
bio_list = [("4AA33", "vedoli"), ("4AB02", "infli"), ("4AB04", "adali"), ("4AB06", "goli"), ('4AC05', 'usteki'), ('4AA24', 'natali')]
'''

def make_dummies(df, var_list):
    for var in var_list:
        df[var[1]] = df.alt.procedure_codes.contains(var[0])

make_dummies(df,bio_list)
'''
def make_dummy_text(df, var_list, varname):
    for var in var_list:
        df[var[1]] = df.procedure_codes.str.contains(var[0])
        df[var[1]] = np.where(df[var[1]] == True, var[1], '')
        name_list = [var[1] for var in var_list]
    df[varname] = df[name_list].sum(axis=1)
    

make_dummy_text(df, bio_list, 'bio')

# entyvio and inflixi simultaniously = entyvio. 19 infusions
df['bio'] = df['bio'].replace('vedoliinfli', 'vedoli')
df['bio'] = df['bio'].replace('vedoliinfli', 'vedoli')
df['bio'] = df['bio'].replace('infliadali', 'infli')
df['bio'] = df['bio'].replace('infligoli', 'infli')
df['bio'] = df['bio'].replace('adaligoli', 'adali')
df['bio'] = df['bio'].replace('adaliusteki', 'adali')
df['bio'] = df['bio'].replace('vedoliadali', 'vedoli')
df['bio'] = df['bio'].replace('infliusteki', 'infli')

df['bio'] = df['bio'].replace('inflicertoli', 'infli')
df['bio'] = df['bio'].replace('golicertoli', 'goli')
df['bio'] = df['bio'].replace('golicertoli', 'goli')


df.bio.value_counts()
#%% id in index and columns
df = df.set_index('pid')
df.index.name = 'pid_index'
df['pid'] = df.index.values
#%% diagnose vars
region_code = {7: 'hso', 4: 'hm', 3: 'hv', 5: 'hn', 6 : 'privat'}
df.region = df.region.replace(region_code)

df['in_date'] = pd.to_datetime(df.in_date)
df['diagnose_date'] = df[(df.k50 == 1) | (df.k51 == 1)].groupby(['pid'])['in_date'].min()

#df = df[df.inn_mnd >= df.diagnose_mnd]
df['days_after_diagnose'] = df.in_date - df.diagnose_date
df.days_after_diagnose = df.days_after_diagnose.dt.days
df['diagnose_year'] = df.diagnose_date.dt.year
df = df[df.days_after_diagnose >-1]

df['diagnose_age'] = df[df.in_date == df.diagnose_date].groupby('pid').agegr.min()
df['diagnose_age'] = (df.diagnose_age-1)*5+2

df['los'] = np.where(df.los == 99999, 50,df.los)
df['nregion'] = df[df.region != 'privat'].groupby('pid')['region'].nunique()

df['diagnose_year'] = np.where(df.diagnose_year == 2007,2008,df['diagnose_year'])

region_name = {'hso': 'South-East', 'hm': 'Central', 'hv': 'West', 'hn': 'North'}
df.region = df.region.replace(region_name)

#%%
dff = df.copy()
#df = dff.copy()
#%%
def new_ids(df,cohort_years,return_as = 'set',pid = 'pid', year_col = 'year',  years_down = 3):
    """
    bappbipp
    """
    cohort = []
    for year in cohort_years:
        ins = set(df[df[year_col] == year][pid])
        befores =  set(df[df[year_col].isin(range(year-years_down,year))][pid])
        new = ins.difference(befores)
        cohort.append(b)
    if return_as=='list':
        print('List might cointain duplicates')
        return cohort
    elif return_as=='set':
        cohort = [item for sublist in cohort for item in sublist]
        return cohort
    elif return_as =='dict':
        cohort = dict(zip(cohort_years, cohort))
        return cohort
    #return print('Please specify how the cohort should på returned ("list","set" or "dict")')

a = df[(df.k50==1)|(df.k51 ==1)]
b = new_ids(a,range(2011,2016), return_as = 'list')
len(b)

# If return as list:
"{}".format(len(b)-len(set(b)))+ ' is number of people classified as new patients twice'
#%%

def first_event(self,pid, date_col, return_as = 'series'):
    """
    Select people from dataframe
    
    Dataframe columns:
        pid: Unique patient identifier (string)
        date_col: Datetime variable (string)
        Return_as: series or dict (string)
        
    Example:
        pd.DataFrame.first_event('pid','in_date', 'dict')
        
    Returns:
        First observation per patient as series or dict
        
    """
    time = self.sort_values([pid, date_col]).groupby(pid)[date_col].first()

    if return_as == 'dict':
        time = dict(time)
    return time
   

pd.DataFrame.first_event = first_event

folk = df.first_event('pid','in_date', 'dict')


def make_cohort(self,pid, date_col,cohort_years):
    """
    Select people from dataframe
    """
    self = self[self[date_col].dt.day =< self[date_col].dt.day + 377]

    self['diagnose_date'] = self.sort_values([pid, date_col]).groupby(pid)[date_col].first()
    self['days_after_diagnose'] = self.in_date - self.diagnose_date
    a = self[self.days_after_diagnose < 377]

    cohort = set(a[a.dt.year.isin(cohort_years)].index)
    return cohort

df['days_between'] = df.sort_values(['pid', 'in_date']).groupby('pid')['in_date'].diff()

df2 = df[df.bio != '']
df2['days_between'] = df2.sort_values(['pid', 'in_date']).groupby('pid')['in_date'].diff()

df2.days_between = df2.days_between.dt.days

df3 = df2[df2.bio == 'infli']

df3.days_between.value_counts(normalize = True)[0:10]
8.plot.bar()
#%%
len(a.dt.year.isin([2010,2011]))
df.pid.nunique()   
a = df.sort_values(['pid','in_date']).groupby('pid')['in_date'].first()

df.diagnose_date.dt.year

df['diagnose_date'] = df[(df.k50 == 1) | (df.k51 == 1)].groupby(['pid'])['in_date'].min()

#within one year
a = df[(df.k50==1)|(df.k51 ==1)]
a = a[a.days_after_diagnose < 377]

b = a.groupby('pid').size()
b = b[b>1]

df = df[df.pid.isin(b.index)]

unit = 'month'
"df.in_date.dt.{unit}"

eval("df.in_date.dt.year")

years = 2010,2011
list(years)
map(np.unique(),df[df['year'].isin(range(2010,2012))]['pid'].unique())

len(set(df[df['year'].isin(range(2010,2012))]['pid']))
8



def not_in_but_around(df,year,pid = 'pid', year_col = 'year', return_as = 'count',years_down = 1, years_up = 1):
    """
    bappbipp
    """
    ins = set(df[df[year_col] == year][pid])
    befores =  set(df[df[year_col].isin(range(year-years_down,year))][pid])
    afters =  set(df[df[year_col].isin(range(year+1,year+years_up+1))][pid])
    both = befores.intersection(afters)
    missing = both.difference(ins)
    if return_as == 'count':
        return len(missing)
    elif return_as == 'pct':
        return len(missing)/len(ins)
        
    return missing
    

ppl2010 = not_in_but_around(df,2012,pid = 'pid', year_col = 'year', return_as = 'count',years_down = 2, years_up = 2)

ppl2010


#%%
def get_ids(df, in_years, groupby, year_col = 'year', pid = 'pid', return_as = 'count'):
    """
    mippmapp
    """
    if not isinstance(groupby, list):
        groupby = list([groupby])

    ids = df[df[year_col].isin(in_years)].groupby(groupby)[pid].unique()
    if return_as == 'count':
        ids = df[df[year_col].isin(in_years)].groupby(groupby)[pid].nunique()
    elif return_as == 'dict':
        ids = dict(ids)
    elif return_as == 'pct':
        ids = df[df[year_col].isin(in_years)].groupby(groupby)[pid].nunique()  
        ids = ids/ids.sum()
    return ids

ss = get_ids(df, in_years = [2011,2013], groupby = 'institution', return_as = 'pct')

ss

#%%
def create_set_persons(pharma, aar):
    persons = set(df[df.aar == aar][df.alt.str.contains(pharma)]['lopenr'].unique().tolist())
    return persons

#inflix_persons_2010 = create_set_persons('4AB02', 2012)

def create_many_sets(pharmas, aar):
    p = {}
    for year in aar:
        for pharma in pharmas:
            #persons = set(df[df.aar == aar][df.alt.str.contains(pharma)]['lopenr'].unique().tolist())
            p[(pharma, year)] = create_set_persons(pharma, year)
    return p

#a = create_set_persons2(pharmas = ['4AB02', '4AB04'], aar = [2011,2012])
#a['4AB04',2011]
def create_set_intersection(pharmas, aar):
    p = {}
    for year in aar:
        for pharma in pharmas:
            p[(pharma, year)] = create_set_persons(pharma, year)
    
    persons = p[pharma, year]     

    for setp in p:
        persons = persons.intersection(p[setp])
    return persons
    
#%%
def variations(df, var_list, geo):
    """
    Returns a dataframe with the frequency of persons with a given list of events for the units defined by geo (eg. hospitals, municipalities)
    """
    make_one_dum(df, var_list, dum_name = 'tmptmp')
    a = df.groupby(['institution', 'year', 'id','disease'])['bio'].max()
    a = a.groupby(level = ['institution', 'year','disease']).sum()
    b = df.groupby([geo, 'year','disease'])['id'].nunique()
    c = a / b
    c = c.unstack(level = 'year')
    c = c[~(c==0)]
    c = c.dropna()
    #b = b.sort_values('2014_left', ascending = False)
    collist = c.columns.tolist()
    for col in collist:
        c["rank" + str(col)] = c[col].rank(ascending = False)

    c = c.join(b.unstack('year'), lsuffix='_pct', rsuffix='_obs')
    return c
 
d = variations(df, bio_list, geo = 'institution')

#%%

def agg_regions(df, variables, years, agg_years=False):
   a = {}
   p = {}
   n = {}
   for year in years:
    
       tmp = df[df.diagnose_date.dt.year == year]
       print(year)
       for disease in ['uc', 'cd']:
           tmp2 = tmp[tmp.disease == disease]
           for region in regions:
               tmp3 = tmp2[tmp2.region == region]
               n[(disease, year, region)] = tmp3.pid.nunique()
               for days in range(1,2000,31):
                   b = tmp3[tmp3.days_after_diagnose < days].groupby('pid')[variables].max().sum()
                   a[(disease, year, region, days)] = b
                   p[(disease, year, region, days)] = (b/n[(disease, year, region)]) * 100
                   n[(disease, year, region, days)] = n[(disease, year, region)]
                   
   a = pd.DataFrame(a).T
   p = pd.DataFrame(p).T
   n = pd.DataFrame(n, index = range(1)).T
   
   if agg_years is True:
       a = a.groupby(level = [0,2,3]).sum()
       n = n.groupby(level = [0,2,3]).sum()
       a.index.names = ['disease', 'region', 'day']
       n.index.names = ['disease', 'region', 'day']
       p = a.divide(n.iloc[:,0], axis = 'index')
       
   return a,p,n

