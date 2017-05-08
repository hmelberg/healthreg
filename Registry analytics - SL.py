# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:54:36 2017

@author: sandresl_adm
"""
#%%
dff = df.copy()

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

df.in_date.dt.day


a = df.sort_values(['pid', 'in_date']).groupby('pid')['in_date'].first()

set(a.index)

df['days_between'] = df.groupby('pid')['days_after_diagnose'].apply(pd.rolling_mean,2, min_periods=1)  
df['days_between'] = df.groupby('pid')['days_after_diagnose'].apply(pd.rolling_mean,2, min_periods=1)  

df['days_between'] = df.sort_values(['pid', 'in_date']).groupby('pid')['in_date'].diff()


df2 = df[df.days_between > ]

df[]


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



def not_in_but_around(df, year):
    ins = df[df.aar == year]['lopenr'].unique().tolist()
    ins = set(ins)
    befores =  df[df.aar < year]['lopenr'].unique().tolist()
    befores = set(befores)
    afters =  df[df.aar > year]['lopenr'].unique().tolist()
    afters = set(afters)
    both = befores.intersection(afters)
    missing = both.difference(ins)
    return missing

    
def get_ids(df, in_years, sub):
    ids = df[df.aar.isin(in_years)].groupby([sub])['lopenr'].nunique().tolist()
    ids = set(ids)
    return ids

ss = get_ids(df, in_years = [2011,2013], sub = 'institusjon')
len(ss)
def not_in_but_around(df, year):
    ins = df[df.aar == year]['lopenr'].unique().tolist()
    ins = set(ins)
    befores =  df[df.aar < year]['lopenr'].unique().tolist()
    befores = set(befores)
    afters =  df[df.aar > year]['lopenr'].unique().tolist()
    afters = set(afters)
    both = befores.intersection(afters)
    missing = both.difference(ins)
    return missing
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