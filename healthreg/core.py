import pandas as pd
import numpy as np
import os
import re
import pickle

#%% helper functions
def _tolist(string_or_list):
    if isinstance(string_or_list, list):
        return string_or_list
    else:
        return [string_or_list]
    
def _invert(schema):
    inverted = dict( (v,k) for k in schema 
                       for v in schema[k] )
    return inverted

def _totuple(str_or_list):
    """
    converts a string or a list into a tuple
        tuplify('npr20')
        tuplify(['results', 'resultater'])
        
    note: may seems unnecessary, but it is suprisingly helpful
        allows input to be single str or list and then changed
    """
    
    if type(str_or_list) == str:
        result = tuple([str_or_list])
    if type(str_or_list) == list:
        result = tuple(str_or_list)
    return result

def _check_if_path(path):
    """ gets the current path if no path is specified and 
        makes a list of it if only one path is specified"""

    if not path:
        path = [os.getcwd()]
    if type(path) == str:
        path = [path]
    return path

#%%
def extract_filename(file, ignore_path=True, ignore_format=True):
    """ Extracts filename from a string
        
    >>> extract_filename('C:/dat/meps/annual/meps2014.csv')
    meps2004
    """
    tmp = file    
    if ignore_path:
        tmp = file.split('/')[-1]
    if ignore_format:
        tmp = tmp.split('.')[0]
    return tmp

#%%
def get_vars(files=None):
    """
    returns the variable names in a file or list of files
    
    Parameters
    ----------
    
    files    : str or list
        filename or list of filenames
    
    Returns
    -------
    dict
    
    A dictionary with the filenames as keys and the columnnames as values
    """
    
    files=_tolist(files)        
    variables = {}

    for i, file in enumerate(files):
        tmp = pd.read_csv(file, nrows = 5, header = 0) 
        variables[file] = tmp.columns.tolist()
    return variables

def get_dtypes(files=None):
    files=_tolist(files)
    dtypes = {}
    for i, file in enumerate(files):
        tmp = pd.read_csv(file, nrows = 5, header = 0) 
        variables[file] = tmp.columns.tolist()
        datatypes[file] = tmp.dtypes.tolist()
    return datatypes

def get_nfirst(files=None, n=5):
    files=_tolist(files)
    nfirst = {}
    datatypes = {}
    for i, file in enumerate(files):
        nfirst[file] = pd.read_csv(file, nrows=n, header=0)
    return nfirst

# Rewrite this (Very inefficient and may cause errors if n is larger than rows in file)        
def get_nlast(files=None, n=5):
    files=_tolist(files)
    top = {}
    datatypes = {}
    for i, file in enumerate(files):
        tmp = pd.read_csv(file, header=0) #what if no header
        nlast[file] = tmp.tail(5)
    return nlast

#%%
def explore(files=None, rows=5):
    files=_tolist(files)
    nfirst = {}
    variables = {}
    datatypes = {}
    for i, file in enumerate(files):
        nfirst[file] = pd.read_csv(file, nrows=rows, header=0)
        variables[file] = nfirst[file].columns.tolist()
        datatypes[file] = nfirst[file].dtypes.tolist()
    return nfirst, variables, datatypes



#%%
def get_filelist(path=None, starts_with=None, ends_with=None, contains=None, 
              subpaths=False, 
              strip_folders=False, 
              strip_file_formats=False,
              regexp=None, 
              only_filename=False):
    """ Returns a list of files in a path 
    
    
    parameters
    ----------
        path (str or list) : the directory to search (str) 
                             or a list of directories to search

        starts_with (str or list): only includes files that starts with a
            given string (or list of strings). Default: None.
<<<<<<< HEAD
            
        ends_with (str or list): only includes files that ends with a
            given string (or list of strings). Default: None.
            
=======
            
        ends_with (str or list): only includes files that ends with a
            given string (or list of strings). Default: None.
            
>>>>>>> 4dbadd8d1cde5a9480ed3a848e08b9cb9b90080e
        contains (str or list): only includes files that contains a
            given string (or list of strings). Default: None.
        
        subpaths (bool) : Includes all subdirectories if True. Dafault: False
    
        example
        -------
        files = get_filelist(path = 'C:/dat/meps/annual/', starts_with='meps', ends_with='csv')

    """
    
    # Note: may be extended to include list of paths, and/or all subpaths
    
    path = _check_if_path(path)
    all_files = []

    for folder in path:
        print(folder)
        # include subdirectories if subpaths is True            
        if subpaths: 
            files = [x[0] for x in os.walk(folder)]
        else:
            files = os.listdir(folder)
        
        # include only files that satisfy given criteria
        files = select_files(files=files, 
                             starts_with=starts_with, 
                             ends_with=ends_with, 
                             contains=contains, 
                             regexp=regexp, 
                             only_filename=only_filename)
        
        # make list of all files
        full_files = [folder+file for file in files]
        print(full_files)
        all_files.append(full_files)
        print(all_files)
    return all_files[0] #check if this works with multiple dirs
    
#%%
def select_from_list(lst, 
                 starts_with=None, 
                 ends_with=None, 
                 contains=None, 
                 regexp=None):
    """
    Selects some elements from a list of strings
    
    Example
    In a list of many many meps files for several years, get only files
    from year 2000:
        
    >>>files = select_from_list(files, starts_with="meps20", only_filename=True)
    >>>k73x = select_from_list(icdtopid.keys, starts_with='K73')
    
    """
    # hmm not happy with this ... not clear if it is and or or when multiple conditions are specified
    
   
    if starts_with:
        selected = [element for element in lst if element.startswith(starts_with)]
    
    if ends_with:
        selected = [element for element in lst if element.endswith(ends_with)]
        
    if contains:
        selected = [element for element in lst if contains in element]
        
    if regexp:
        regxpr=re.complie(regexpr)
        selected = [element for element in lst if regxpr.search(element) is not None]  
           
    return selected

#%%
def select_from_filelist(files, 
                 starts_with=None, 
                 ends_with=None, 
                 contains=None, 
                 regexp=None, 
                 ignore_path=True, 
                 ignore_format=True,
                 only_filename=True):
    """
    Selects some elements from a list of strings
    
    Example
    In a list of many many meps files for several years, get only files
    from year 2000:
        
    >>>files = select_from_list(files, starts_with="meps20", only_filename=True)
    >>>k73x = select_from_list(icdtopid.keys, starts_with='K73')
    
    """
    
    for file in files:
        if starts_with:
            beginnings = _totuple(starts_with)
            files = [file for file in files 
                     if extract_filename(file=file).startswith(beginnings)]
        if ends_with:
            endings = _totuple(ends_with)
            files = [file for file in files if file.endswith(endings)]
        if contains:
            files = [file for file in files if contains in file]
        if regexp:
            regxpr=re.complie(regexpr)
            files = [file for file in files if regxpr.search(file) is not None]             
    return files
      
<<<<<<< HEAD


#%%
def ids_from_csv(files, 
                 id_col='pid', 
                 schema={'pid': ['pid']}, 
                 find=None, 
                 query=None, 
                 dtype=None,
                 **kwargs):
    """
    Get set of ids from rows that satisfy some conditions in a list of csv files
            
    Parameters
    ----------
        files: (list) List of files to include
        
        id_col: (string) Column name containing the ids
        
        schema: (dictionary) 
            A mapping of desired column names (keys) to the equivalent and 
            possible diverse column names in the various files. 
=======
>>>>>>> 4dbadd8d1cde5a9480ed3a848e08b9cb9b90080e

            Example: The columns with id and year information may have 
            different names in different files and we want to label 
            all the id columns "pid" and similarly for 'year':
                
                schema={'pid'  : ['id', 'ID', 'person'], 
                        'year' : ['jahre', 'year', 'yyyy']}
                
        find: (dictionary) 
            Key is column name to search, 
            Value is list of strings to search for
            
            Example: find = {'icd_main': ['K50, K51']}
        
        query: (string) 
            Text specifying a query. 
            Example: query = "age > 18"
        
                                      
    Example
    -------
    Get ids for indidivuals who have K50 or K51 in the column icd_main:
    
    >>>ibd_codes = {'icd_main': ['K50, K51']}       
    >>>ids_from_csv(df, id_col='pid' find=ibd_codes)
    

    Returns
    -------
        dictionary with files as keys and a set of ids as values
    """
    
    old_to_new = _invert(schema)
    original_cols= old_to_new.keys()
    
    
    oldvars = {var for varlist in schema.values() for var in varlist}
    
    files=_tolist(files)
    ids={}
    
    for file in files:
        header = pd.read_csv(file, nrows=0)
        header = set(header.columns)
        usecols = header.intersection(oldvars)
        df = pd.read_csv(file, usecols=usecols, dtype=dtype)
        df = df.rename(columns=old_to_new)                     
        idset = ids_from_df(df=df, id_col=id_col, find=find, query=query, **kwargs)
        ids[file] = idset   
        
    return ids
#%%
def ids_from_csv(files, 
                 id_col='pid', 
                 schema={'pid': ['pid']}, 
                 find=None, 
                 query=None, 
                 dtype=None,
                 **kwargs):
    """
    Get set of ids from rows that satisfy some conditions in a list of csv files
            
    Parameters
    ----------
        files: (list) List of files to include
        
        id_col: (string) Column name containing the ids
        
        schema: (dictionary) 
            A mapping of desired column names (keys) to the equivalent and 
            possible diverse column names in the various files. 

<<<<<<< HEAD
=======
            Example: The columns with id and year information may have 
            different names in different files and we want to label 
            all the id columns "pid" and similarly for 'year':
                
                schema={'pid'  : ['id', 'ID', 'person'], 
                        'year' : ['jahre', 'year', 'yyyy']}
                
        find: (dictionary) 
            Key is column name to search, 
            Value is list of strings to search for
            
            Example: find = {'icd_main': ['K50, K51']}
        
        query: (string) 
            Text specifying a query. 
            Example: query = "age > 18"
        
                                      
    Example
    -------
    Get ids for indidivuals who have K50 or K51 in the column icd_main:
    
    >>>ibd_codes = {'icd_main': ['K50, K51']}       
    >>>ids_from_csv(df, id_col='pid' find=ibd_codes)
    

    Returns
    -------
        dictionary with files as keys and a set of ids as values
    """
    
    old_to_new = _invert(schema)
    original_cols= old_to_new.keys()
    
    
    oldvars = {var for varlist in schema.values() for var in varlist}
    
    files=_tolist(files)
    ids={}
    
    for file in files:
        header = pd.read_csv(file, nrows=0)
        header = set(header.columns)
        usecols = header.intersection(oldvars)
        df = pd.read_csv(file, usecols=usecols, dtype=dtype)
        df = df.rename(columns=old_to_new)                     
        idset = ids_from_df(df=df, id_col=id_col, find=find, query=query, **kwargs)
        ids[file] = idset   
        
    return ids
#%%

>>>>>>> 4dbadd8d1cde5a9480ed3a848e08b9cb9b90080e
def ids_from_df(df, id_col='pid', find=None, query=None, **kwargs):
    """
        Gets the ids that satisfy some conditions
                
        Parameters
        ----------
            df: (dataframe) Dataframe
            id_col: (string) name of column with the ids
            find: (dictionary) Key is column name to search, value is list of strings to search for
            query: (string) String specifying a selection query. Example: "year > 2003"
            out: (string, 'set') Format to return the ids
            
        Example
        -------
        Get all ids for peole who have K50 or K51 in the column icd_main:
        
        >>>contains = {'icd_main': ['K50, K51']}       
        >>>get_ids(df, id_col='pid', contains=None, query=None, out='set', **kwargs)
        
        Returns
        -------
            Set
    """
    
    if query:
        df = df.query(query)
        
    if find:
        #boolean array, starintg point: all false, no rows are included
        combined=np.array([False] * len(df))
        
        for var, searchlist in find.items():
            searchstr = "|".join(searchlist)
            true_if_found = df[var].str.contains(searchstr, na=False)
            combined = np.logical_or(combined, np.array(true_if_found))
        df=df[combined]
        
    ids = set(df[id_col])
         
    return ids

#%%
def read_csv_using_ids(files, 
                 ids, 
                 schema=None, 
                 id_col='pid', 
                 columns=None, 
                 select=None, 
                 dtype=None, 
                 **kwargs):
    """
    Aggregate selected rows and columns from a list of csv files to a single dataframe

    files: list
        list of csv files to read
        
    ids: dict
        a dict with the ids of the rows to be included
           
    schema : dict (with lists of columns as values)
        keys are the desired column names that corresponod to the list of columns names in values
        example: 
            A schema to so specify that files named 'person' 'number' and 'id' in the different csv files contain the same information and should be labelled 'pid:
            {'pid' : ['person', number', 'id']}
                    
    id_col= str (default is 'pid')
        the column name that containts information about id (column names from the schema)
        
    columns= List
        the columns in the files to be read (column names from the schema)
  
    select: string
        additional query to limit the result
        example: query='female==1'
         
    dtype=dictionary of dtypes (column names from the schema)
        speficy the dtypes of the columns
    
    
    
    """
    
    files=_tolist(files)
    dfs=[]
    
    # allow user to input a list of ids (instead of dicts with 
    # separate ids sets for every (file, key) combination
    #
    # if it is a list, every items read use the same ids
    # useful shortcut when ids are connected and unique across files
    
    if isinstance(ids, list):
        for file, keys in file_keys.items():
            for key in keys:
                ids[(file, key)] = ids
    
    if schema:
        oldvars = {var for varlist in schema.values() for var in varlist}
        old_to_new = dict( (v,k) for k in schema for v in schema[k] )
    
    for file in files:
        header = pd.read_csv(file, nrows=0)
        header = set(header.columns)
        
        # is columns are specified, use this, if not, use all columns 
        if columns:        
            oldvars = {var for k, varlist in schema.items() 
                        if k in columns 
                        for var in varlist} 
            
            use_columns = header.intersection(oldvars)
        else:
            use_columns = header
        
        df = pd.read_csv(file, usecols=use_columns)
        
        if schema:
            df = df.rename(columns=old_to_new)
        
        df = df[df[id_col].isin(ids[file])]
     
        if select:
            df = df.query(select)
        dfs.append(df)
    dfs=pd.concat(dfs)
    return dfs

#%%

def read_hdf_using_ids(file_keys,
                 ids, 
                 schema=None, 
                 id_col='pid', 
                 columns=None, 
                 select=None, 
                 dtype=None, 
                 **kwargs):
    """
    Aggregate selected rows and columns from a tables in a hdf datastore to a single dataframe

    path: dict
        a dictionary of paths with a list of keys for the files to be read
        example:  
            path_keys = {'Q:/mepsdata/annual/hdf/meps.h5' :['meps1992', 'meps1993']}
        
    ids: dict
        a dict with the ids of the rows to be included in each file
        the key in the dictionary is a tuple with path and key to the table
        example: ids = {('Q:/mepsdata/annual/hdf/meps.h5', 'meps1992'): [28, 35]}
           
    schema : dict (with lists of columns as values)
        keys are the desired column names that corresponod to the list of columns names in values
        example: 
            A schema to so specify that files named 'person' 'number' and 'id' in the different csv files contain the same information and should be labelled 'pid:
            {'pid' : ['person', number', 'id']}
                    
    id_col= str (default is 'pid')
        the column name that containts information about id (column names from the schema)
        
    columns= List
        the columns in the files to be read (column names from the schema)
  
    select: string
        additional query to limit the result
        example: query='female==1'
         
    dtype=dictionary of dtypes (column names from the schema)
        speficy the dtypes of the columns
    """
    
    dfs=[]
    
    # allow user to input a list of ids (instead of dicts with separate ids 
    # for every (file, key) combination
    # if it is a list, every file select for these ids
    #
    
    if (isinstance(ids, list)) or (isinstance(ids, set)):
        ids_new={}
        for file, keys in file_keys.items():
            for key in keys:
                ids_new[(file, key)] = ids
        ids = ids_new
        
    if schema:
        oldvars = {var for varlist in schema.values() for var in varlist}
        old_to_new = dict( (v,k) for k in schema for v in schema[k] )
    
    for file, keys in file_keys.items():
        for key in keys:
            header = pd.read_hdf(file, key, start=0, stop=1)
            header = set(header.columns)
            
            if columns:        
                oldvars = {var for k, varlist in schema.items() 
                            if k in columns 
                            for var in varlist} 
                
                use_columns = header.intersection(oldvars)
            else:
                use_columns = header
                
            store= pd.HDFStore(file)
            #hmm, should not use pid here, use old pid (since pid may not exist)
            #better, but not perfect since there may two is columns in some files?
            #to solve this: need "true" file schema. one for each file
            #or make user input a pid dictionary?
            
            #local_pid_col = set(old_to_new['pid']).intersection(header)
                        
            all_ids = store.select_column(key, id_col)
            selected_ids = ids[(file, key)]
            idarray = all_ids.isin(selected_ids)
            
            df = pd.read_hdf(file, key, columns=use_columns, where=idarray)
            
            if schema:
                df = df.rename(columns=old_to_new)
                     
            if select:
                df = df.query(select)
            dfs.append(df)
            
    dfs=pd.concat(dfs)
    return dfs



def find_ids_and_read_hdq(
        file_keys,
        id_col='pid', 
        schema={'pid': ['pid']}, 
        find=None,
        query=None, 
        dtype=None,
        **kwargs):
<<<<<<< HEAD



#%%
def event_aggregator(
        df,
        pidcol='pid', 
        datecol='in_date', 
        eventcol=['bio', 'procedure_codes'],
        keep_events=None,
        exclude_events=None,
        reduce_event_detail=None,
        old_sep=[' ',','],
        new_sep=',',
        episode_sep=',',
        query=None,
        out='set'):
    """ Lists all events associated with a person in chronological order
    
    """
    
    old_sep = [sep.replace(' ','\s') 
                            for sep in old_sep]
    sep_re = '|'.join(old_sep)
    
    if query:
        df=df.query(query) # careful ... this modifies the original dataframe ... or
        
    if isinstance(eventcol,str):
        eventcol=[eventcol]
    
    df=df.set_index(pidcol)
    
    #necessary if user include columns with numeric (not text) event descriptors
    cols = [df[col].astype(str).values for col in eventcol[1:]]
        
    grouped = (df[eventcol[0]]
        .astype(str)
        .str.cat(cols, sep=new_sep)
        .add(episode_sep)
        .groupby(level=0)
        .sum()
        .str.replace(fr'({sep_re}){{1,}}', new_sep) # replace all old seperators with new
        .str.replace(fr'({new_sep}){{2,}}]', new_sep) # eliminate all cases of double seperators
        .str.replace(fr'(^{new_sep})|({new_sep}$)','') # eliminate seperators from beginning and end
        )
    
    # note: ineffienct keep_event algorithm, may be improved
    if (keep_events!=None) | (exclude_events != None):
        event_set = [set(x.split(new_sep)) for x in grouped.values]
        if keep_event:
            all_events = set.union(*event_set)
            exclude_events=all_events - set(keep_events)    
        #many options, looping (inefficient), regex (long!)
        #try: expand, drop, reintegrate (probem: memory since scales to individual with most events!)
        expanded_df = grouped.str.split(',', expand=True)
        expanded_df = grouped.replace(list(exclude_events), value=np.nan)
        #reintegrate
        cols = [expanded_df[col].values 
                for col in expanded_df.columns]
        grouped = dfexpanded[cols[0]].str.cat(cols[1:], sep=new_sep)
    
    if reduce_event_detail:
        for reg, rep in reduce_event_detail.items():
            grouped = grouped.str.replace(reg, rep)
            
    if out=='set':
        event_set = [set(x.split(new_sep)) for x in grouped.values]
        event_set_dict = dict(zip(grouped.index,event_set))
        return event_set_dict
    
    elif out=='list':
        return grouped.to_dict()
    
    elif out=='series':
        return grouped
    
    elif out=='expanded':
        return grouped.str.split(',', expand=True)
    
    elif out=='no repeat, no sep list':
        grouped = (grouped
                   .str.replace(new_sep, '')
                   .str.replace(r'(\w)\1{%d,}'%(1), r'\1')
                   )
        
    else:
        print('Error: Out method {out} is not a valid out argument')
        return


=======
        
#%% testing
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
>>>>>>> 4dbadd8d1cde5a9480ed3a848e08b9cb9b90080e
#%%
def first_event(self, id_col, date_col, groupby = [], return_as = 'series'):
    """
        Returns time of the first observation for the person
        
        PARAMETERS
        ----------
            id_col: (string) Patient identifier
            date_col: (string) 
            groupby: (list)
            return_as: ('series' or 'dict', default: 'series')
            
        
        EXAMPLE
        -------
            first_event(id_col = 'pid', date_col='diagnose_date', groupby = ['disease'], return_as='dict')
            
            df['cohort'] = df.first_event(id_col = 'pid', date_col='diagnose_date', return_as='dict')
        
        Returns
            Pandas dataframe or a dictionary
            
        
    """
    groupby.append(id_col)
    first = self.sort_values([id_col, date_col]).groupby(groupby)[date_col].first()
    if return_as=='dict':
        first = first.to_dict()
    
    return first