# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:51:36 2017

@author: hmelberg
"""

        
#%% testing
#%%
a=event_aggregator(
        df=df,
        pidcol='pid', 
        datecol='in_date', 
        eventcol=['bio', 'procedure_codes'],
        keep_events=None,
        exclude_events=None,
        old_sep=[' ',','],
        new_sep=',',
        episode_sep='_',
        query='male==0',
        out='list')

#%%
a = get_filelist('C:/dat/knp/annual/hdf/')
select_files(a, ends_with='csv')

get_filelist()  


file='C:/dat/knp/annual/hdf/npr.h5'
file_keys = {file: ['npr2008','npr2009','npr2010','npr2011','npr2012', 'npr2013', 'npr2014']}



ibd = ['K500','K501','K508','K509','K510', 'K512','K513','K514', 'K515','K518', 'K519']

cirrhosis = ['K703', ]

k70x = select_from_list(icdtopid.keys(), starts_with='K70')


allids=set()

def union_of_ids_from_dict(icds, icd_to_pid):    
    allids = {ids  
              for icd in icds 
              for ids in icd_to_pid[icd]}
    return allids

ids_k70x = union_of_ids_from_dict(k70x, icdtopid)




ids={}
for file, keys in file_keys.items():
    for key in keys:
        ids[(file, key)] = allids
    
len(allids)

df = pd.read_feather('C:/dat/knp/annual/ibd.feather')

df.head()
df.columns
df.query("icd10main1 == 'K500'")

# speed tests
df = pd.read_csv('C:/dat/knp/annual/npr2004.csv', encoding='latin-1')


usecols=['pasient', 'hdiag', 'bdiag1', 'bdiag2', 'bdiag3']
dtypecol = {'pasient': int, 'hdiag': str, 'bdiag1':str, 'bdiag2':str, 'bdiag3':str}
dtypecol = {'pasient': int, 'hdiag': 'category', 'bdiag1': 'category', 'bdiag2': 'category', 'bdiag3': 'category'}

import datetime

a = datetime.datetime.now()
df = pd.read_csv('C:/dat/knp/annual/npr2004.csv', 
                 encoding='latin-1',
                 usecols=usecols,
                 dtype=dtypecol)
b = datetime.datetime.now()
print(b-a)

df=

a = datetime.datetime.now()
df = pd.read_hdf('C:/dat/knp/annual/hdf/npr.h5', 
                 key='npr2004')
b = datetime.datetime.now()
print(b-a)

df.columns
df.dtypes
from fastparquet import write
write('C:/dat/knp/annual/hdf/npr2004.parq', df)


a = datetime.datetime.now()
from fastparquet import ParquetFile
pf = ParquetFile('C:/dat/knp/annual/hdf/npr2004.parq')
df = pf.to_pandas()
b = datetime.datetime.now()
print(b-a)

pf = ParquetFile('C:/dat/knp/annual/hdf/npr2004.parq')
df3 = pf.to_pandas(usecols, filters=[('hdiag', '==' ['K500', 'K501', 'K502'])])
df3 = pf.to_pandas(['col1', 'col2'], filters=[('col3', 'in' [1, 2, 3, 4])])

a = datetime.datetime.now()
df = pd.read_feather('C:/dat/knp/annual/npr2004.feather')
df.columns

b = datetime.datetime.now()
print(b-a)

bcolz_dir = "movielens-denorm.bcolz"
if os.path.exists(bcolz_dir):
    import shutil
    shutil.rmtree(bcolz_dir)
    
import bcolz  

bdf = bcolz.ctable.fromdataframe(df, rootdir='C:/dat/knp/annual/bcolz/npr.bcolz')



df.reset_index().to_feather('C:/dat/knp/annual/npr2004.feather')
a = datetime.datetime.now()
df = pd.read_csv('C:/dat/knp/annual/npr2004.csv', 
                 encoding='latin-1',
                 usecols=usecols,
                 dtype=dtypecol)
b = datetime.datetime.now()
print(b-a)


df.columns
df.hdiag.str.contains('K500')

df.query("hdiag=='K500' | bdiag1=='K500' | bdiag2=='K500' ")
len(df)


['K500' in str(icd) for icd in df.hdiag.tolist()]

df.query("hdiag == ['K500', 'K501']")


# Grab DataFrame rows where column has certain values
valuelist = ['K500', 'K501', 'K502']
df = df[df.column.isin(valuelist)]


df['hdiag2']=df.hdiag.astype('category')
df.hdiag2.str.contains('K500')
df.query("hdiag==['K500', 'K501', 'K502', 'K503'] or (bdiag1==['K500', 'K501', 'K502', 'K503']) or bdiag2==['K500', 'K501', 'K502', 'K503'] or bdiag3==['K500', 'K501', 'K502', 'K503']")
a = df.groupby('hdiag').get_group('K500')

df[df.hdiag == 'K500']

df.hdiag2.str.contains('K500|K501|K502|K503|K509')

df=read_hdf_using_ids(file_keys,ids=ids_k70x)

df['all_diag'] = df.hdiag + ',' +  df.bdiag1 + ',' + df.bdiag2 + ',' + df.bdiag3

df.all_diag.str.contains('K500|K501|K502|K503')

keys = ['npr2010', 'npr2011']
ids = {(file, key): brain for key in keys}
ids.keys()
icd_to_pid['K500']

df.head()
df = df.reset_index()
#df.to_feather('C:/dat/knp/annual/ibd.feather')
df.bydel2

df.dtypes

# rem: may be better to conovert to categorical?
def mixed_to_str(df):
    for column in df.columns:
        if df[column].dtype == np.object:
            print(column)
            df[column]= df[column].map(str)
    return df
df = mixed_to_str(df)

a = datetime.datetime.now()
def mixed_to_category(df):
    for column in df.columns:
        if df[column].dtype == np.object:
            print(column)
            df[column]= df[column].astype('category')
    return df
a = datetime.datetime.now()
df = mixed_to_str(df)
b = datetime.datetime.now()
print(b-a)

df.dtypes

df.to_hdf('C:/dat/knp/annual/hdf/n2014.h5', key='npr20xx', format='table',append=True, data_columns = ['pid','drg'], complevel=3)
df.columns

df['bydel2'].dtype == np.object
df.dtypes['bydel2']
import numpy as np

df=pd.read_hdf('C:/dat/knp/annual/hdf/npr.h5', key='npr1992', start=0, stop=2)

a=datetime.datetime.now()
df2 = pd.read_hdf('C:/dat/knp/annual/hdf/n2014.h5', key='npr20xx', where='f_aar==2000')
b=datetime.datetime.now()
print(b-a)

df.to_feather('C:/dat/knp/annual/hdf/npr20xx.feather')

df3 = pd.read_feather('C:/dat/knp/annual/hdf/npr20xx.feather')

df=pd.read_hdf('C:/dat/knp/annual/hdf/npr.h5', start=0, stop=2)

new_to_old = {'pid': ['id', 'pid'], 
          'icd10main1': ['tilstand_1_1s'],
          'icd10main2': ['tilstand_1_2'],
          'drg':['drg'],
          'icd10main_all' : ['icdmain'],
          'icd10bi_all' : ['icdbi'],
          'birthyear' : ['fodselsar'],
          'male' :  ['kjonn']}
schema=new_to_old

df = read_ids_hdf(file_keys=file_keys,
                 ids=ids, 
                 schema=new_to_old, 
                 id_col='pid', 
                 columns=False, 
                 select=None, 
                 dtype=None)

df= df.reset_index()
df.to_feather("C:/dat/knp/annual/k70x.feather")
del df['index']
df.columns
df.icd10main1.value_counts()
df2.tail()
df.head()
c = df.male
c.columns=['old_male', 'male']
c
del df['male']
df2=pd.concat([df,c], axis=1)

df = df.append(c)

df.utdato