# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:17:33 2018

@author: hmelberg
"""
import uuid
import numpy as np
import pandas as pd
import re
import os
from itertools import chain
from itertools import zip_longest


#%%
def listify(string_or_list):
    """
    returns a list if the input is a string, if not: returns the input as it was

    :param string_or_list (str or any):

    :return:  a list if the input is a string, if not: returns the input as it was

    note
        allows user to use a string as an argument when the only one element needs to be specified
        example: cols='icd10' is allowed instead of cols=['icd10']
        cols='icd10' is transformed to cols=['icd10'] by this function

    """
    if isinstance(string_or_list, str):
        string_or_list = [string_or_list]
    return string_or_list

#%%
def sample_persons(df, pid='pid', n=100):
    """
    take a sample of all observations for some (randomly selected) individuals
    
    sample: number of individuals to sample
            if less than 1: fraction of individuals to sample
    
    example
    
    sample_df=sample_persons(df, sample=100)
    """
    ids=df.pid.unique()
    
    #sample less than zero means taking fractional sample
    if sample<1:
        sample=int(sample*len(ids))
    new_ids=np.random.choice(ids, size=sample)
    new_sample = df[df.pid.isin(new_ids)]
    return new_sample

#%%
def first_event(df, codes, cols, pid = 'pid', date='in_date', sep=None, groupby=None):
    """
        Returns time of the first observation for the person based on certain 
        
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
            
            first_event(df=df, codes=['k50*','k51*'], cols='icd', date='date')
        
        Returns
            Pandas series or a dictionary
            
        
    """
    codes=listify(codes)
    cols=listify(cols)
    
    expanded_cols = expand_columns(df=df, cols=cols)
    expanded_codes=expand_codes(df=df,codes=codes,cols=cols, sep=sep)
    
    rows_with_codes=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subdf=df[rows_with_codes]
    
    #groupby.extent(pid)
    first_date=subdf[[pid, date]].groupby(pid, sort=False)[date].min()
    
    return first_date
#%%
def get_ids(df, codes, cols, groupby, pid='pid', out=None, sep=None):
    codes=listify(codes)
    groupby=listify(groupby)
    
    codes=expand_codes(df=df,codes=codes,cols=cols,sep=sep)
    
    rows_with_codes=get_rows(df=df,codes=codes, cols=cols,sep=sep)
    #grouped_ids = df[rows_with_codes].groupby([pid, groupby]).count()
    grouped_ids = df[rows_with_codes].groupby(groupby)[pid].unique()
    grouped_ids=grouped_ids.apply(set)
    
    return grouped_ids

#%%
def unique_codes(df,
           cols,
           sep=None,
           strip=True,
           name=None):        
        
    """
    Get unique set of values from columns in a dataframe 
    
    parameters
        df: dataframe
        cols: columns with  content used to create unique values
        sep: if there are multiple codes in cells, the separator has to 
             be specified. Example: sep =',' or sep=';'
             
    
    note
        Each column may have multiple values (if sep is specified)
        Star notation is allowed to describe columns eg: col='year*'
        In large dataframes this function may take some time, 

    
    examples
        
        to get all unique values in all columns that start with 'drg':
            drg_codeds=unique_codes(df=df, cols=['drg*'])
        
        all unique atc codes from a column with many comma separated codes
             atc_codes==unique_codes(df=ibd, cols=['atc'], sep=',')
    
    worry
        numeric columns/content
    """
 
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
        
    unique_terms=set(pd.unique(df[cols].values.ravel('K')))
    #unique_terms={str(term) for term in unique_terms}
    
    if sep:
        compound_terms = {term for term in unique_terms if sep in str(term)}
        single_uniques = {term for term in unique_terms if sep not in str(term)}
        split_compounds = [term.split(sep) for term in compound_terms]
        split_uniques = {term.strip()
                    for sublist in split_compounds
                    for term in sublist
                    }
        unique_terms = single_uniques | split_uniques
        
    if strip:
        unique_terms = list({str(term).strip() for term in unique_terms})
    
    return unique_terms

#%%
def events(df,
          codes, 
          cols=['icd'], 
          pid='pid', 
          pre_query=None, 
          post_query=None, 
          sep=',',
          out='df'):
    
    """
    Get all events for people who have a specific code/diagnosis
    
    parameters
        df: dataframe og all events for all patients
        codes: the codes that identify the patients of interest
                star notation is allowed
                example: icd codes for crohn's disease: K50*
        cols: the column(s) where the code(s) can be found
                star notation is allowed
        pid: the column with the id of the individuals
        sep: the seperator used between codes if multiple codes exist in a column
        out: if the output should be the (sub)dataframe or a list of ids

    example
        get all events only for those who are registered with an ibd diagosis
        
        ibd = events(df=df,
              codes=['K50*', 'K51*'], 
              cols=['icd*'], 
              pid='pid')   
    """
    
    codes=listify(codes)
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
    
    codes=expand_codes(df=df, codes=codes, cols=cols, sep=sep)
    
    with_codes = get_rows(df=df, codes=codes, cols=cols, sep=sep)
    
    pids = df[with_codes][pid].unique()
    
    if out=='df':
        return df[df[pid].isin(pids)]
    
    elif out=='pids':
        return pids
    else:
        print(f"Error: {out} is not a valid 'out' argument")
        return
      
#%%
def stringify(df,
              codes, 
              cols, 
              pid='pid', 
              start_time='in_date',
              replace=None,
              end_time=None, 
              sep=None,
              new_sep=None,
              single_value_columns=None,
              out='series'):
    
    codes=listify(codes)
    cols=listify(cols)
    
    single_cols = infer_single_cols(df=df, 
                                   cols=cols, 
                                   sep=sep, 
                                   check_all=True)
    
    multiple_cols = [set(cols) - set(single_cols)]
    
    expanded_codes=expand_codes(df=df, cols=cols, codes=codes, sep=sep)
    #use expanded codes?, not codes as argument. 
    # why? because a code may be in a compount col and not in single cols
    
    if single_cols:
        single_events=stringify_singles(df=df,
                                      codes=expanded_codes, 
                                      cols=single_cols, 
                                      pid=pid, 
                                      start_time=start_time,
                                      replace=replace,
                                      end_time=end_time, 
                                      out='df')
        all_events=single_events

    
    if multiple_cols:
        multiple_events=stringify_multiples(df=df,
                                      codes=expanded_codes, 
                                      cols=multiple_cols, 
                                      pid=pid, 
                                      start_time=start_time,
                                      replace=replace,
                                      end_time=end_time, 
                                      out='df')
        all_events=multiple_events
    
    if single_cols and multiple_cols:
        all_events=pd.concat([multiple_events, single_events])
    
    
    if out=='series':
        events_by_id = all_events.sort_values([pid, start_time]).groupby(pid)['events'].sum()
        return events_by_id
    elif out =='df':
        return all_events


#%%
def expand_codes(df,
          codes=None, 
          cols=None,
          sep=None):
    """
    Returns all the unique terms in the columns that match codes
    
    example
        get all atc codes that are related to steroids in the atc column:
            codes= ['H02*', 'J01*', 'L04AB02', 'L04AB04']
            codes=expand_codes(df=df, codes=['H02*'], cols='atc')
    
    """
    if isinstance(codes, dict):
        codes=list(codes.keys())
        
    codes=listify(codes)
    cols=listify(cols)

        
    #expand only if there is something to expand
    if '*' not in ''.join(codes):
        return codes # returns like this may not pass a smell test?
    
    else:
        matches=set()
        
        unique_words=unique_codes(df=df, cols=cols, sep=sep) 
        
        for find in codes:    
            start_words=end_words=contain_words=set(unique_words)
            
            if find.count('*')==1:
                startswith, endswith = find.split('*')
                 
                if startswith:
                    start_words = {word for word in unique_words if str(word).startswith(startswith)}
                if endswith:
                    end_words = {word for word in unique_words if str(word).endswith(endswith)}
               
                intersecting_words = start_words & end_words
            else:
                intersecting_words = {find}
            matches.update(intersecting_words)
        
    return list(matches)

#%%    
def get_rows(df, 
             codes, 
             cols,  
             sep=None):
    """
    Returns a boolean array that is true for the rows where column(s) contain the code(s)
    
    example
        get all drg codes starting with 'D':
            
        d_codes = get_rows(df=df, codes='D*', cols=['drg'])
    """
    
    codes=listify(codes)
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
    
    expanded_codes = expand_codes(df=df, 
                                     codes=codes, 
                                     cols=cols,
                                     sep=sep)
            
    # if compound words in a cell    
    if sep:
        expanded_codes_regex = '|'.join(expanded_codes)
        b = np.full(len(df),False)
        for col in cols:
            a = df[col].str.contains(expanded_codes_regex,na=False).values
            b = b|a
    # if single value cells only
    else:
        b=df[cols].isin(expanded_codes).any(axis=1).values
    
    return b

#%%
def get_some_id(df, 
             codes, 
             cols,
             xid,
             sep=None):
    """
    help function for all get functions that gets ids based on certain filering criteria
    
    x is the column with the info to be collected (pid, uuid, event_id)
    
    
    """
    
    codes=listify(codes)
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
    
    expanded_codes = expand_codes(df=df, 
                                     codes=codes, 
                                     cols=cols,
                                     sep=sep)
            
    # if compound words in a cell    
    if sep:
        expanded_codes_regex = '|'.join(expanded_codes)
        b = np.full(len(df),False)
        for col in cols:
            a = df[col].str.contains(expanded_codes_regex,na=False).values
            b = b|a
    # if single value cells only
    else:
        b=df[cols].isin(expanded_codes).any(axis=1).values
    
    pids=set(df[b][xid].unique())
    
    return pids

#%%
def get_uuid(df, codes, cols, uuid='uuid', sep=None):
    """
    Returns a set pids who have the given codes in the cols
    """
    
    uuids=get_some_id(df=df, codes=codes, some_id=uuid, sep=sep)
    
    return uuids

#%% 
def make_uuid(df, name='uuid'):
    """
    Creates a list of uuids with the same length as the dataframe
    """
    
    uuids = [uuid.uuid4().hex for _ in range(len(df))]
    return uuids
    
#%%
def get_pids(df, 
             codes, 
             cols,
             pid='pid',
             sep=None):
    """
    Returns a set pids who have the given codes in the cols
    
    example
        get all drg codes starting with 'D':
            
        get pids for all individuals who have icd codes starting with 'C509':
            
        c509 = get_pids(df=df, codes='C509', cols=['icdmain', 'icdbi'], pid='pid')
    """
    pids=get_some_id(df=df, codes=codes, some_id=pid, sep=sep)
    return uuids 

#%%
def count_persons(df, codes=None, cols=None, pid='pid', sep=None, normalize=False, dropna=False):
    """
    Count number of individuals who are registered with some codes
    
    args:
        codes
        cols
        pid
        sep
        normalize
        dropna
    
    examples
        count_persons(df=df, codes={'4AB*':'a'}, cols='ncmpalt', sep=',', pid='pid')
        count_persons(df=df.ncmpalt, sep=',', pid='pid')
    """

    codes, cols, old_codes, replace = fix_args(df=df, codes=codes, cols=cols, sep=sep)
    
    rows=get_rows(df=df,codes=codes, cols=cols, sep=sep)
    subset=df[rows].set_index(pid)
    #subset=df[rows]
    
    
    code_df=extract_codes(df=subset, codes=replace, cols=cols, sep=sep)
    labels=list(code_df.columns)  
    
    counted=pd.Series(index=labels)
    
    for label in labels:
        counted[label]=code_df[code_df[label]==1].index.nunique()
    
    if not dropna:
        counted['NaN']=df[pid].nunique()-counted.sum()
        
    if normalize:
        counted=counted/counted.sum()
    return counted

    
#%%
def get_mask(df, 
             codes, 
             cols, 
             sep=None):
    
    codes=listify(codes)
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
    
    expanded_codes = expand_codes(df=df, 
                                     codes=codes, 
                                     cols=cols,
                                     sep=sep)
            
    # if compound words in a cell    
    if sep:
        expanded_codes_regex = '|'.join(expanded_codes)
        b = pd.DataFrame()
        for col in cols:
            b[col] = df[col].str.contains(expanded_codes_regex,na=False).values
            
    # if single value cells only
    else:
        b=df[cols].isin(expanded_codes)
    
    return b


#%%
def expand_columns(df, cols):
        
    cols=listify(cols)
    expanded_cols = cols.copy() #nb copy(), bug if not! 
    
    for col in cols:        
        if '*' in col:
            print(col)
            startstr,endstr = col.split('*')
            if startstr:
                add_cols = list(df.columns[df.columns.str.startswith(startstr)])
            if endstr:
                add_cols = list(df.columns[df.columns.str.endswith(endstr)])
            if startstr and endstr:
                #col with single letter not included, start means one or more of something
                #beginnig is not also end (here!)
                start_and_end = (df.columns.str.startswith(startstr) 
                                & 
                                df.columns.str.endswith(endstr))
                add_cols = list(df.columns[start_and_end])
        else:
            add_cols=[col]
                
        expanded_cols.remove(col)
        expanded_cols.extend(add_cols)
    return expanded_cols

#%%
def infer_single_value_columns(df, 
                               cols, 
                               sep, 
                               n=100, 
                               check_all=False):
    single_value_columns=[]
    multiple_value_columns=[]
    

    for col in cols:
        if (df[col].head(100).str.contains(sep).any()) or (df[col].tail(100).str.contains(sep).any()):
               multiple_value_columns.append(col)
        else: 
            single_value_columns.append(col)
    
    if check_all:
        for col in single_value_columns:
            if df[col].str.contains(sep).any():
                multipe_value_columns.append(col)
                single_value_columns.remove(col)
    return single_value_columns


#%%
def stringify_multiples(df,
              codes, 
              cols,
              replace=None,
              pid='pid', 
              start_time='in_date',
              end_time=None,
              period_length=None,
              sep=None,
              new_sep=None,
              single_value_columns=None,
              out='series'):
    
    codes=listify(codes)
    cols=listify(cols)
    
    expanded_codes=expand_codes(df=df, codes=codes, cols=cols, sep=sep)
    
    cols=expand_columns(df=df, cols=cols)
    
    mask=get_mask(df=df, codes=codes, cols=cols, sep=sep)
    
    subset=df[mask.any(axis=1).values]
    
    relevant = subset[cols]
    
        
    events=relevant
    
    merged=events.fillna('').astype(str).sum(axis=1)
     
        
    expanded_codes = [fr"""\b{code.strip()}\b""" for code in expanded_codes]
    expanded_codes_regex = '|'.join(expanded_codes)
    
    relevant_codes_only=merged.str.findall(expanded_codes_regex)
    
    relevant_codes_only= relevant_codes_only.apply(lambda x: ','.join(x))
    

    if replace:
        relevant_codes_only=relevant_codes_only.astype(str).replace(replace)   
    
    
    subset['events'] = relevant_codes_only
    
    if period_length: 
        subset['time_period'] = subset[start_time] - subset['first_event_date']
        
        subset['time_period'] = subset['time_period'].dt.days.div(period_length).astype(int)
        
        events_by_id_time = (subset
                                 .sort_values([pid, 'time_period', start_time])
                                 .groupby([pid, 'time_period'])['events'].sum()
                                 )
        
        events_by_id_time.unstack('time_period').fillna('-').apply(lambda x: ','.join(x), axis=1)
        
    
    if out=='series':
        events_by_id = subset.sort_values([pid, start_time]).groupby(pid)['events'].sum()
        return events_by_id
    elif out =='df':
        return subset

#%%
def first_event(df, codes, cols='icd', pid='pid', sep=None, date_col='in_date'):
    """
    Get the date of the first ocurrance of the code for each individual
    
    example
        date of first registered event with an ibd code for each individual:
            first_ibd=first_event(df=df, pid='id', codes=['K50*', 'K51*'], cols='icd', date_col='innDato') 
    
    note
    allows star notation in codes and columns:
        codes = 'K50*'
        
    Returns a series where id is the index and the first date is the column
    
    todo: 
        include groupby
    """
    
    codes=listify(codes)
    cols=listify(cols)
    
    codes=expand_codes(df=df, codes=codes, cols=cols, sep=sep)
    cols=expand_columns(df=df, cols=cols)
    
    mask=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    first_date=df[mask].groupby(pid, sort=False)[date_col].min()
    return first_date



#%% stringify for singles
def stringify_singles(df,
              codes, 
              cols, 
              pid='pid', 
              replace=None,
              start_time='in_date',
              end_time=None,
              time_period=None,
              sep=None,
              new_sep=None,
              delete_repeats=False,
              out='series'):

    codes=listify(codes)
    cols=listify(cols)
    
    cols=expand_columns(df=df, cols=cols)
    
    if len(codes)==1 and codes[0]=='infer':
        expanded_codes = df.apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False)[10]
        #also infer replacement
    expanded_codes=expand_codes(df=df, codes=codes, cols=cols, sep=None)
    
    rows=get_rows(df=df, codes=codes, cols=cols, sep=None)

    subset=df[rows]
    
    relevant_codes = subset[cols].isin(expanded_codes)
    
    relevant = subset[cols][relevant_codes]
    
    if replace:
        relevant=relevant.replace(replace)
    
    events=relevant
    
    if len(cols)>1:   
        events=relevant.iloc[:,0]
        for col in relevant.columns[1:]:
            events=events.str.cat(relevant[col], sep=',')
      
    subset['events'] = events
   
    if out=='series':
        events_by_id = subset.sort_values([pid, start_time]).groupby(pid)['events'].sum()
        if time_period:
            df['time_after']=df[start_date] - df[first_date]
            df['time_category']=df['time_after'].div(time_length).round()
            
                            
        return events_by_id
    elif out =='df':
        return subset
   
#%%    
def stringify_durations(df,
              codes, 
              col='atc', 
              pid='pid', 
              replace=None,
              start='in_date',
              end=None,
              first_date=None,
              length_col='ddd',
              step_length=120,
              dataset_end_date=None,
              censored=None,
              out='df',
              all_codes=None,
              only_use_whole_period=False):
    """
    codes={
        'L04A*' : 'i',
        'L04AB*' : 'a',
                'H02*' : 'c'}
    pr=pr.set_index('pid_index')
    pr['first_date'] = pr.groupby('pid')['date'].min()
    events=stringify_durations(df=pr, codes=codes, start='date', first_date='first_date', dataset_end_date="01-01-2018")
    events=stringify_durations(df=df, codes=codes, col='ncmpalt', start='start_date', first_date='first', dataset_end_date="01-01-2018")

    df2.columns    
    """
    
    if isinstance(codes, dict):
        replace=codes.copy()
        codes=list(codes.keys())
    
    
    if len(codes)==1 and codes[0]=='infer':
        codes = list(df[col].value_counts(ascending=False)[:10].index)
        replace = {code:chr(97+i) for i, code in enumerate(codes)}
        #also infer replacement
    
    # expand codes if star notation is used            
    expanded_codes=expand_codes(df=df,codes=codes, cols=col,sep=None)
    
    
    subset=df[df[col].isin(expanded_codes)]
    subset=subset.sort_values([pid, start])

    subset=subset.set_index(pid, drop=False)
    subset.index.name='pid_index'
    
    if replace:
        replace=expand_replace(df=df,replace=replace, cols=col)
        subset[col]=subset[col].map(replace)
    else:
        replace={code:code for code in expanded_codes}
    
        
    # inclusive end points? 4 days in ddd. so 1. january to 4th or 5th as end?
    # also: decimals: 4.3 ddd = 1. january to 4th or 5th or 6th?    
    subset['end']=subset[start] + pd.to_timedelta(subset[length_col].fillna(0).add(0.5).round(0), unit='D')
    
    if first_date:
        if first_date in df.columns:
            subset['first_event'] = subset[first_date]
        else:
            subset['first_event'] = pd.to_datetime(first_date)     
    else:
        subset['first_event'] = subset.groupby(pid)[start].min()  
    #frist is perhaps faster, but what if it is missing, use min()?
    #silently drop nans or raise warning, or error?
    
    subset['start_position'] = (subset[start] - subset['first_event']).dt.days.div(step_length).astype(int)
        
    subset['end_position'] = (subset['end'] - subset['first_event']).dt.days.div(step_length).add(0.5).astype(int)
    #make adding optional depeding of you want to measuree any use is part og the period or use in whole period
    
    if dataset_end_date:
        subset['dataset_end_position'] = (pd.to_datetime(dataset_end_date) - subset['first_event']).dt.days.div(step_length).add(0.5).astype(int)
    else:
        max_date = subset['end'].max()
        subset['dataset_end_position']= (max_date - subset['first_event']).dt.days.div(step_length).astype(int)
    
    if censored:
        censored_bool=df[censored].notnull()
        subset.loc[censored_bool, 'dataset_end_position']= (subset[censored] - subset['first_event']).dt.days.div(step_length).astype(int)
        
        
    def make_string(events2):

        n_events=events2['dataset_end_position'].max()
        #may extend byond this i.e. to observation end
        event_list = ['-'] * (n_events)    
        from_to_positions = tuple(zip(events2['start_position'].tolist(), events2['end_position'].tolist()))
               
        for pos in from_to_positions:
            event_list[pos[0]:pos[1]]=code
        event_string = "".join(event_list)
        return event_string
    
    string_df=pd.DataFrame(index=subset[pid].unique()) 
    
    for code in set(replace.values()):
        code_df=subset[subset[col].isin([code])]
        stringified=code_df.groupby(pid, sort=False).apply(make_string)
        string_df[code]=stringified
    return string_df


#%%
def stringify_events(df,
              codes='infer', 
              cols='procedure_codes',
              sep=None,
              new_sep=None,
              na_rep=None,
              pid='pid',
              start_date=None,
              end_date=None,
              first_date=None,
              replace=None,
              event_date='in_date',
              step_length=120,
              out='df',
              all_codes=None,
              convert_dates=False,
              data_end_date=None,
              censored=None):
    """
    df.columns
    
    codes={
        '4AB02' : 'i',
        '4AB04' : 'a'}
    
    df=df.set_index('pid', drop=False)
    df['first_date'] = s1.groupby('pid').start_date.min()
  
    
    stringify_events(df,
              codes, 
              cols='ncmpalt',
              sep=',',
              new_sep=None,
              na_rep=None,
              pid='pid',
              start_date='start_date',
              end_date=None,
              first_date='first_date',
              replace=None,
              event_date='start_date',
              step_length=120,
              out='df',
              all_codes=None,
              convert_dates=False,
              data_end_date=None,
              censored=None)
    

    """
    
    if len(codes)==1 and codes[0]=='infer':
        codes = list(df[cols].value_counts(ascending=False)[:10].index)
        # may need to break up this before counting ...
        replace = {code:chr(97+i) for i, code in enumerate(codes)}
        codes=replace
    

    codes, cols, old_codes, replace = fix_args(df=df, 
                                          codes=codes, 
                                          cols=cols, 
                                          sep=sep)
               
       
    rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    
    subset=df[rows]
    
    extracted_codes=extract_codes(df=subset, 
                                     codes=replace, 
                                     cols=cols, 
                                     sep=sep, 
                                     new_sep=new_sep, 
                                     merge=False, 
                                     out='text')

    for code in extracted_codes.columns:
        subset[code]=extracted_codes[code].values        

    if convert_dates:
        if event_date:
            subset[event_date] = pd.to_datetime(subset[event_date])
    
    subset=subset.sort_values([pid, event_date])
    subset=subset.set_index(pid, drop=False)
    subset.index.name='pid_index'        
    # inclusive end points? 4 days in ddd. so 1. january to 4th or 5th as end?
    # also: decimals: 4.3 ddd = 1. january to 4th or 5th or 6th?    
    if first_date:
        if first_date in df.columns:
            subset['first_event'] = subset[first_date]
        else:
            subset['first_event'] = pd.to_datetime(first_date)    
    else:
        subset['first_event'] = subset.groupby(pid)[event_date].first()    

    subset['position']=(subset[event_date]-subset['first_event']).dt.days.div(step_length).dropna().astype(int)    
      
    if data_end_date:
        max_date = pd.to_datetime(data_end_date)
    else:
        max_date = subset[event_date].max()
    
    subset['end']=max_date
    
    # censored, for instance death, then specify column with date of death
    if censored:
        censored_bool=df[censored].notnull()
        subset.loc[censored_bool, 'end']= subset[censored]
        
    subset['end_position'] = (subset['end'] - subset['first_event']).dt.days.div(step_length).astype(int)
    #make adding optional depeding of you want to measuree any use is part og the period or use in whole period
    
        
    def make_string(events2):

        n_events=events2['end_position'].max()+1
        event_list = ['-'] * (n_events)
               
        for pos in events2.position.values: 
            event_list[pos]=code
        event_string = "".join(event_list)
        return event_string
    
    string_df=pd.DataFrame(index=subset[pid].unique()) 
        
    for code in set(replace.values()):
        code_df=subset[subset[code].isin([code])]
        stringified=code_df.groupby(pid, sort=False).apply(make_string)
        string_df[code]=stringified
    return string_df


#%%
def interleave_strings(df,cols=None, sep=" ", nan='-', agg=False):
    """
    Interleaves strings in two or more columns
    
    parameters    
        cols : list of columns with strings to be interleaved
        nan : value to be used in place of missing values
        sep : seperator to be used between time periods
        agg : numeric, used to indicate aggregation of time scale
                default is 1
        
    background
        to identify treatment patters, first stringify each treatment, 
        then aggregate the different treatments to one string
        each "cell" in the string (separated by sep) represent one time unit
        the time unit can be further aggregated to reduce the level of detail
    
    example output (one such row for each person)
        a---s, a---, ai-s, a---, ----
        
        Interpretation: A person with event a and s in first time perod, then a only in second,
        the a, i and s in the third, a only in fourth and no events in the last
    
    purpose
        examine typical treatment patterns and correlations
        use regex or other string operations on this to get statistcs
        (time on first line of treatment, number of switches, stops)
        
    """
    # if cols is not specified, use all columns in dataframe
    if not cols:
        cols=list(df.columns)
        
    if agg:
        for col in cols:
            df[col]=df[col].fillna('-')
            #find event symbol, imply check if all are missing, no events
            try:
                char=df[col].str.cat().strip('-')[0]
            except:
                df[col] = (col.str.len()/agg) *'-'
                
            missing = '-'*len(cols)        
            def aggregator(text, agg):
                missing = '-'*agg
                units = (text[i:i+agg] for i in range(0, len(text), agg))
                new_aggregated=('-' if unit==missing else char for unit in units)
                new_str="".join(new_aggregated)
                return new_str
        df[col]=df[col].apply(aggregator, agg=agg)            
    
    if sep:
        interleaved = df[cols].fillna('-').apply(
                (lambda x: ",".join(
                        "".join(i)
                        for i in zip_longest(*x, fillvalue='-'))), 
                    axis=1)
    else:
        interleaved = df[cols].fillna('-').apply(
                (lambda x: "".join(chain(*zip_longest(*x, fillvalue='-')))),
                axis=1)
                        
    return interleaved


#%%
def overlay_strings(df,cols=None, sep=",", nan='-', collisions='x', interleaved=False):
    """
    overlays strings from two or more columns
    
    note
        most useful when aggregating a string for events that usually do not happen in the same time frame
        
    parameters    
        cols : list of columns with strings to be interleaved
        nan : value to be used in place of missing values
        collisions: value to be usef if ther is a collision between events in a position
        
        
    background
        to identify treatment patters, first stringify each treatment, 
        then aggregate the different treatments to one string
        each "cell" in the string (separated by sep) represent one time unit
        the time unit can be further aggregated to reduce the level of detail
    
    example output (one such row for each person)
        asaaa--s--aa-s-a
        
        Interpretation: A person with event a and s in first time perod, then a only in second,
        the a, i and s in the third, a only in fourth and no events in the last
    
    purpose
        examine typical treatment patterns and correlations
        use regex or other string operations on this to get statistcs
        (time on first line of treatment, number of switches, stops)
    
    todo
        more advanced handeling of collisions
            - special symbols for different types of collisions
            - warnings (and keep/give info on amount and type of collisions)
            
    """
    # if cols is not specified, use all columns in dataframe
    if not cols:
        cols=list(df.columns)
               
    interleaved = df[cols].fillna('-').apply(
                (lambda x: "".join(chain(*zip_longest(*x, fillvalue='-')))),
                axis=1)
    step_length=len(cols)
    
    def event_or_collision(events):
        try:
            char=events.strip('-')[0]
        except:
            char='-'
        n=len(set(events).remove('-'))
        if n>1:
            char='x'
        return char
    
    def overlay_individuals(events):
        
        units = (events[i:i+step_length] for i in range(0, len(events), step_length))
        
        new_aggregated=(event_or_collision(unit) for unit in units)
        new_str="".join(new_aggregated)
        return new_str
       
    interleaved.apply(overlay_individuals)
    
                        
    return interleaved

#%%      
def shorten(events, agg=3, missing='-'):
    """
    create a new and shorter string with a longer time step
    
    parameters
        events: (str) string of events that will be aggregated
        agg: (int) the level of aggregation (2=double the step_length, 3=triple)
    """
    try:
        char=events.strip('-')[0]
    except:
        char='-'
    units = (events[i:i+agg] for i in range(0, len(text), agg))
    new_aggregated=('-' if unit==missing else char for unit in units)
    new_str="".join(new_aggregated)
    return new_str


#%%
def shorten_interleaved(text, agg=3, sep=',', missing='-'):  
    """
    text="a-si,a--i,a-s-,--si,---i,--s-"
    
    shorten_interleaved(c, agg=2)
    """     
    units=text.split(sep)
    ncodes=len(units[0])
    nunits=len(units)
    
    unitlist=[units[i:i+agg] for i in range(0, nunits, agg)]
    charlist = ["".join(aggunit) for aggunit in unitlist]
    unique_char = ["".join(set(chain(chars))) for chars in charlist]
    new_str=",".join(unique_char)
    #ordered or sorted?
    #delete last if it is not full ie. not as many timee units in it as the others?
    #shortcut for all
    return new_str


#%%
def expand_replace(df,replace,cols, sep=None, strip=True):
    """
    Takes a dictionary of shorthand codes and replacements, and returns a dictionary with all the codes expanded
    
    Example:
        expand_replace(df=df, replace={'AB04*':'b'}, col='atc')
        
        May return
            {'AB04a':'b', 'AB04b': 'b', 'AB04c':'b'}
        
    """
    # may use regex instead, but this may also be slower to use later?
    cols=listify(cols) 
    codes=list(replace.keys())
    
    codes = expand_codes(df=df, codes=codes, cols=cols, sep=None)
    
    unexpanded = {code:text for code,text in replace.items() if '*' in code}
    
    for starcode, text in unexpanded.items():
        
        startswith, endswith = starcode.split('*')
        
        #starting_codes  = ending_codes = start_and_end_codes = {}
        starting_codes  = {}
        ending_codes  = {}
        start_and_end_codes  = {}
        #may be unnecessary to do this (and it may link the dictionaries in unwanted ways?)
         
        if startswith:
            starting_codes = {code:text for code in codes if code.startswith(startswith)}
        if endswith:
            ending_codes = {code:text for code in codes if code.endswith(endswith)}
        if startswith and endswith:
            start_and_end_codes = {starting_code:starting_code[x] for x in starting_code if x in ending_code}
        
        replace.update({**starting_codes, **ending_codes, **start_and_end_codes})
        
        del replace[starcode]
    return replace
#%%
def reverse_dict(dikt):
    """
    takes a dict and return a new dict with old values as key and old keys as values (in a list)
    
    example
    
    reverse_dict({'AB04a':'b', 'AB04b': 'b', 'AB04c':'b', 'CC04x': 'c'})
    
    will return
        {'b': ['AB04a', 'AB04b', 'AB04c'], 'c': 'CC04x'}
    """
    
    new_dikt={}
    for k, v in dikt.items():
        if v in new_dikt:
            new_dikt[v].append(k)
        else:
            new_dikt[v]=[k]
    return new_dikt

#%%
def fix_args(df, codes=None, cols=None, sep=None):
    """various standard fixes to inputs
    
    make strings lists
    expand codes and columns
    separate codes (a list) and replace (a dict)
    """
    # use all cols if not specified 
    # assumes index is pid ... maybe too much magic?
    if not cols:
        # if a series, convert to dataframe to make it work? 
        # experimental, maybe not useful
        if isinstance(df, pd.Series):
            df=pd.DataFrame(df)
            col=list(df.columns)    
        # if a dataframe, use all cols (and take index as pid)
        else:
            cols=list(df.columns)
        df['pid']=df.index.values
        
    # if codes is not specified, use the five most common codes
    if not codes:
        cols=expand_columns(df=df, cols=listify(cols))
        codes=value_counts(df=df, cols=cols, sep=sep).sort_values(ascending=False)[:5]
    
    replace=None
    names=None
    
    # if codes is not just a list of codes, split it into codes, replace and name
    if isinstance(codes, dict):
        #dict with codes and both short (symbols) and long labels (names)
        if isinstance(list(codes.values())[0], tuple):
            replace = {code: value[0] for code, value in codes.items()}
            names = dict(codes.values())
        # dict with only one label
        else:
            replace=codes.copy()
        codes=list(codes.keys())
    
    codes=listify(codes)
    cols=listify(cols)        
    
    expanded_codes=expand_codes(df=df,
                                codes=codes, 
                                cols=cols, 
                                sep=sep)
    
    expanded_cols=expand_columns(df=df, 
                                 cols=cols)
    #todo? expand_replace? or just to it when necessary?
    #todo: returnnames also
    return expanded_codes, expanded_cols, codes, replace

#%%
def extract_codes(df, codes, cols, sep=None, new_sep=',', na_rep='', 
                  prefix=None, 
                  merge=False,
                  out='bool'):
    """
    Produce one or more columns with only selected codes
    
    Can produce a set of dummy columns for codes (and code groups).
    Can also produce a merged column with only extracted codes.
    Accept star notation.
    Also accepts both single value columns and columns with compund codes and seperators
    
    out can be: 'text', 'category', 'bool' or 'int'
    
    example
    to create three dummy columns, based on codes in icdmain column:
        
    extract_codes(df=df, 
              codes={'S72*':'f', 'K50*': 'cd', 'K51*':'uc'}, 
              cols='[icdmain', 'icdbi'], 
              merge=False,
              out='text') 
    
    """
     
    codes, cols, old_codes, replace = fix_args(df=df, 
                                          codes=codes, 
                                          cols=cols, 
                                          sep=sep)
    subset=pd.DataFrame(index=df.index)
    
    if replace:
        reversed_replace=reverse_dict(replace)
    else:
        reversed_replace={code:[code] for code in codes}
    
    new_codes=list(reversed_replace.keys())
    
    for k, v in reversed_replace.items():
        rows = get_rows(df=df,codes=v,cols=cols,sep=sep)
        if out=='bool':        
            subset[k]=rows           
        elif out =='int':
            subset[k]=rows.astype(int)   
        elif out=='category':
            subset.loc[rows, k]=k
            subset[k]=subset[k].astype('category')
        else:
            subset[k]=na_rep
            subset.loc[rows, k]=k
            
            
    if merge and out=='bool':
        subset=subset.astype(int).astype(str)
        
    if merge:
        headline=', '.join(new_codes)
        merged=subset.iloc[:,0].str.cat(subset.iloc[:,1:].T.values, sep=new_sep, na_rep=na_rep)
        merged=merged.str.strip(',')
        subset=merged
        subset.name=headline
        if out=='category':
            subset=subset.astype('category')
    
    return  subset
#%%
def years_in_row(df, year_col='year', groupby=None, info_bank=None, out='pct'):
    """
    average years in row patients are observed, for different start years
    
    years = years_in_row(df, year_col='aar', out='pct')
    """
    years=df[year_col].unique()
    min_year=min(years)
    max_year=end_year=max(years)
    
    pids=df.groupby(year_col)['lopenr'].unique().apply(set).to_dict()
    remains={}
    
    for start_year in range(min_year, max_year):
        remains[(start_year, start_year)]=pids[start_year]
        for end_year in range(start_year+1, max_year):
            remains[(start_year, end_year)]=remains[(start_year, end_year-1)] & pids[end_year]
    
    if out == 'pct':
        print('pct, hello')
        for start_year in range(min_year, max_year):
            start_n=len(remains[(start_year, start_year)])
            remains[(start_year, start_year)]=1
            for end_year in range(start_year+1, max_year):
                remains[(start_year, end_year)]=len(remains[(start_year, end_year)])/start_n

    return remains
        
#%%
def years_apart(df, pid='pid', year='year'):
    """
    pct of patients with observations that are x years apart
    
    years_apart(df=df[ibd])
    
    """
    a=df.groupby(pid)[year].unique()
    b=(a[a.apply(len) >1]
        .apply(sorted)
        .apply(np.diff)
        .sub(1)
        .apply(max)
        .value_counts()
        .div(len(a))
        .sort_index()
        .mul(100)
        
        )
    return b

#%%
def value_counts(df, codes=None, cols=None, sep=None, strip=True, lower_case=False, normalize=False):
    """
    count frequency of values in multiple columns
    
    allows 
        - star notation in codes and columns
        - values in cells with multiple valules can be separated (if sep is defined)
        - replacement and aggregation to larger groups (when code is a dict)
     
    example
    To count the number of stereoid events (codes starting with H2) and use of 
    antibiotics (codes starting with xx) in all columns where the column names
    starts with "atc":
        
    value_counts(df=df, 
                 codes={'H2*': 'stereoids, 'AI*':'antibiotics'},
                 cols='atc*', 
                 sep=',')
    
    more examples
    -------------
    
    value_counts(df, codes='Z51*', cols='icdmain', sep=None)
    value_counts(df, codes='Z51*', cols=['icdmain', 'icdbi'], sep=None)
    value_counts(df, codes='Z51*', cols=['icdmain', 'icdbi'], sep=',')
    value_counts(df, codes={'Z51*':'stråling'}, cols=['icdmain', 'icdbi'], sep=',')
    """
    if codes: 
        codes=listify(codes)
    cols=listify(cols)
    
    replace=None
    if isinstance(codes, dict):
        replace=codes.copy()
        codes=list(codes.keys())
        
    cols=expand_columns(df=df, cols=cols)
    
    if codes:
        codes=expand_codes(df=df, codes=codes, cols=cols, sep=sep)
        rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
        df=df[rows]
            
    if sep:
        count_df=[df[col].str
                      .split(sep, expand=True)
                      .apply(lambda x: x.str.strip())
                      .to_sparse()
                      .apply(pd.Series.value_counts)
                      .sum(axis=1)
                      for col in cols]
        
        count_df=pd.DataFrame(count_df).T
        code_count=count_df.sum(axis=1)
    else:
        code_count=df[cols].apply(pd.Series.value_counts).sum(axis=1)

    if codes:
        code_count=code_count[codes]
    
    if replace:
        replace=expand_replace(df=df, replace=replace, cols=cols, sep=sep, strip=strip)

        code_count=code_count.rename(index=replace).groupby(level=0).sum()

    return code_count                 


#%%
def lookup_codes(dikt, codes):
    """
    returns those elements in a dict where key starts with the expressions listed in codes
    
    todo: more complicated star notations: starts with, contains, endswith
    lookup(medcodes, 'L04*')
    
    """
    
    codes=listify(codes)
    codes = [code.upper().strip('*') for code in codes]
    codes=tuple(codes)

    selected_codes={k:v for k,v in dikt.items()  if str(k).upper().startswith(codes)}
    return selected_codes

#%%
def get_codes(dikt, text):
    """
    returns those elements in a dict where value contains the expressions listed in codes
    
    todo: more complicated star notations: starts with, contains, endswith
    alterative name: find_codes? get_codes?
    
    example
    get all codes that have "steroid" in the explanatory text
    
        get_codes(medcodes, 'steroid*')
    
    """
    
    text=listify(text)
    text = [txt.upper().strip('*') for txt in text]
    #codes = " ".join(codes)
    

    selected_codes={k:v for k,v in dikt.items()  if any(txt in str(v).upper() for txt in text)}

    return selected_codes
    
            

#%%
def stringify_order(df, codes=None, cols=None, pid='pid', event_start='in_date', sep=None, keep_repeats=True, only_unique=False):
    """
    
    examples
    
    codes={
        '4AB01': 'e',
        '4AB02' : 'i',
        '4AB04' : 'a',
        '4AB05' : 'x',
        '4AB06' : 'g'}
    medcodes=read_code2text()
    df['diagnosis_date']=df[df.icdmain.fillna('').str.contains('K50|K51')].groupby('pid')['start_date'].min()
    df.columns
    df.start_date
    
    bio_codes= {
     '4AA23': ('n', 'Natalizumab'),
     '4AA33': ('v', 'Vedolizumab'),
     '4AB02': ('i', 'Infliximab'),
     '4AB04': ('a', 'Adalimumab'),
     '4AB06': ('g', 'Golimumab'),
     '4AC05': ('u', 'Ustekinumab')}
    
    bio_codes= {'L04AA23': 'n',
     'L04AA33': 'v',
     'L04AB02': 'i',
     'L04AB04': 'a',
     'L04AB06': 'g',
     'L04AC05': 'u'}

    
    a=stringify_order(  
            df=df,
            codes=bio_codes,
            cols='ncmpalt',
            pid='pid',
            event_start='start_date',
            sep=',',
            keep_repeats=True,
            only_unique=True
            )
    
    codes={
        'L04AB01': 'e',
        'L04AB02' : 'i',
        'L04AB04' : 'a',
        'L04AB05' : 'x',
        'L04AB06' : 'g'}
    
    bio_rows=get_rows(df=pr, codes=list(codes.keys()), cols='atc')
    pr['first_bio']=pr[bio_rows].groupby('pid')['date'].min()
    
    a=stringify_order(  
            df=pr,
            codes=codes,
            cols='atc',
            pid='pid',
            event_date='date',
            sep=','
            )
    """
    
    
    # fix formatting of input    
    codes, cols, old_codes, replace = fix_args(df=df,codes=codes, cols=cols, sep=sep)
    
    # get the rows with the relevant columns
    rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subset=df[rows].sort_values(by=[pid, event_start]).set_index('pid')
    
    # extract relevant codes and aggregate for each person 
    code_series=extract_codes(df=subset, codes=replace, cols=cols, sep=sep, new_sep='', merge=True, out='text')
    string_df=code_series.groupby(level=0).apply(lambda codes: codes.str.cat())
    
    # eliminate repeats in string
    if not keep_repeats:
        string_df=string_df.str.replace(r'([a-z])\1+', r'\1')
    
    if only_unique:
        def uniqify(text):
            while re.search(r'([a-z])(.*)\1', text):
                text= re.sub(r'([a-z])(.*)\1', r'\1\2', text)
            return text
        string_df = string_df.apply(uniqify)
    return string_df
#%%
    
def del_repeats(str_series):
    """
    deletes consecutively repeated characters from the strings in a series
    
    del_repeats(a)
    """
    no_repeats = str_series.str.replace(r'([a-z])\1+', r'\1')
    return no_repeats


def del_singles(text):
    """
    Deletes single characters from string
    todo: how to deal with first and last position ... delete it too?
    
    b=del_singles(a)
    (.)\1{2,}
    
    lookahead \b(?:([a-z])(?!\1))+\b
    lookback ....
 
    
    no_singles = str_series.str.replace(r'(.)((?<!\1)&(?!\1))', r'')
    """
    # text with only one character are by definition singles
    if len(text)<2:
        no_singles=''
    else:
        no_singles="".join([letter for n, letter in enumerate(text[1:-1], start=1) if ((text[n-1]==letter) or (text[n+1]==letter))])
        # long textx may not have any singles, so check before continue
        if len(no_singles)<1:
            no_singles=''
        else:
            if text[0]==no_singles[0]:
                no_singles=text[0]+no_singles
            if text[-1]==no_singles[-1]:
                no_singles=no_singles+text[-1]
    
    return no_singles

#
def sankey_format(df, labels, normalize=False, dropna=False, threshold=0.01):
    """
    
    labels=dict(bio_codes.values())
    import holoviews as hv
    hv.Sankey(t1).options(label_position='left')
    hv.extension('bokeh')    
    t4=t1.copy()
    
    %store t4
    """
    a=a.apply(lambda row: ' '.join(row))
    a=a.str.split(expand=True)
    
    for col in a.columns:
        a[col]=a[col] + ' (' + str(col+1) +')'
        
    if not dropna:
        a=a.fillna(f'No new')
    
    all_counts={}
    for col in range(len(a.columns))[1:]:
        counts=a.groupby(a[col-1])[col].value_counts(normalize=normalize)
        if normalize:
            counts=counts.mul(100).astype(int).fillna(0)
           
        counts.name='value'
        #counts = counts.rename(index=labels).reset_index()
        counts=counts.reset_index()        
        counts.columns=['source', 'target', 'value']
        
        all_counts[col]=counts
    t1=pd.concat(all_counts, ignore_index=True)    
    t1=t1[t1.source != 'No new']



        
         
    a.groupby(1)[2].value_counts()
    




#%%
def stringify_order_date(df, 
                         codes=None, 
                         cols=None, 
                         pid='pid',
                         days_per_step=None,
                         
                         event_start='in_date', 
                         
                         first_date='first_event_for_person',
                         last_date=None,
                         censored_date=None,
                         
                         sep=None,
                         merge=True,
                         meta=None):
    """
    Creates a string for each individual describing events at position in time
    
    :param df: dataframe
    :param codes: codes to be used to mark an event
    :param cols: columns with the event codes
    :param pid: column with the personal identification number
    :param event_date: column containing the date for the event
    :param sep: the seperator used between events if a column has multiple events in a cell
    :param keep_repeats: identical events after each other are reduced to one (if true)
    :param only_unique: deletes all events that have occurred previously for the individual (if true)
    :return: a series with a string that describes the events for each individual
    codes={
        '4AB01': 'e',
        '4AB02' : 'i',
        '4AB04' : 'a',
        '4AB05' : 'x',
        '4AB06' : 'g'}
    
    df['diagnosis_date']=df[df.icdmain.fillna('').str.contains('K50|K51')].groupby('pid')['start_date'].min()
    
    a=stringify_order_date(  
            df=df,
            codes=codes,
            cols='ncmpalt',
            pid='pid',
            event_date='start_date',
            string_start='diagnosis_date',
            days_per_step=90,
            sep=',',
            )
    
    codes={
        'L04AB01': 'e',
        'L04AB02' : 'i',
        'L04AB04' : 'a',
        'L04AB05' : 'x',
        'L04AB06' : 'g'}
    
    bio_rows=get_rows(df=pr, codes=list(codes.keys()), cols='atc')
    pr['first_bio']=pr[bio_rows].groupby('pid')['date'].min()
    
    a=stringify_order_date(  
            df=pr,
            codes=codes,
            cols='atc',
            pid='pid',
            event_date='date',
            string_start='first_bio',
            days_per_step=90,
            sep=','
            )
    
    """

    # drop rows with missing observations in required variables
    df=df.dropna(subset= [pid, event_date])
    
    # find default min and max dates to be used if not user specified 
    min_date=df[event_start].min()
    max_date=df[event_start].max()

    # drop rows outside time period of interest
    if first_date:
        if first_date in df.columns:
            df=df[df[event_start]>=df[first_date]]
        else:
            min_date=pd.to_datetime(first_date)
            df=df[df[event_start]>=min_date]
        
    if last_date:
        if last_date in df.columns:
            df=df[df[event_start]>=df[last_date]]
        else:
            max_date=pd.to_datetime(last_date)
            df=df[df[event_date]<=max_date]
    
    # note an individual min date cannot be before overall specified min date
    # should raise error if user tries this
    # same with max: individual cannot be larger than overall
    
    max_length_days = (max_date-min_date).days
    max_length_steps = int(max_length_days/days_per_step)       
    
    # if codes are not specified, use the five most common codes
    if not codes:
        cols=expand_columns(listify(cols))
        codes=value_counts(df=df, cols=cols, sep=sep).sort_values(ascending=False)[:4]
        
    # fix formatting of input (make list out of a string input and so on)
    codes, cols, old_codes, replace = fix_args(df=df,codes=codes, cols=cols, sep=sep)
    
    # get the rows that contain the relevant codes
    rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subset=df[rows] # maybe use .copy to avoid warnings?
    
    # find position of each event (number of steps from overall min_date)
    subset['position']=(subset[event_start]-min_date).dt.days.div(days_per_step).astype(int)

    # create series with only the relevant codes for each person and position 
    code_series=extract_codes(df=subset.set_index([pid, 'position']), 
                              codes=replace, 
                              cols=cols, 
                              sep=sep, 
                              new_sep=',', 
                              merge=True, 
                              out='text')
    
    # base further aggregation on the new extracted series with its col and codes
    col=code_series.name
    codes=code_series.name.split(', ')

    # drop duplicates (same type of even in same period for same individual)
    code_series=code_series.reset_index().drop_duplicates().set_index(pid, drop=False)
    
    ## make dict with string start end end positions for each individual
    # explanation: 
    # the string is first made marking events in positions using calendar time
    # but often we want the end result to be strings that start at specified 
    # individual dates, and not the same calendar date for all
    # for instance it is often useful to start the string at the date the 
    # person receives a diagnosis
    # same with end of string: strings may end when a patient dies
    # user can specify start and end dates by pointing to columns with dates
    # or they may specify an overall start and end date
    # if individual dates are specified, the long string based on calendar 
    # time is sliced to include only the relevant events

    if string_start:
        # if a column is specified
        if string_start in subset.columns:
            start_date=subset.groupby(pid)[string_start].first().dropna().to_dict()
        # if a single overall date is specified
        else:
            date=pd.to_datetime(string_start)
            start_date={pid:date for pid in subset[pid].unique()}
        # convert start date to start position in string
        start_position={pid:int((date-min_date).days/days_per_step) 
                        for pid, date in start_date.items()}  
                
    if string_end:
        if string_end in subset:
            end_date=subset.groupby(pid)[string_end].first().dropna().to_dict()
        else:
            date=pd.to_datetime(string_end)
            end_date={pid:date for pid in subset[pid].unique()}
        # convert date to position in string
        end_position={pid:(date-min_date).dt.days.div(days_per_step).astype(int) 
                        for pid, date in end_date.items()}
    
    # takes dataframe for an individual and makes a string with the events    
    def make_string(events):
        # get pid of individual (required to find correct start and end point)
        person=events[pid].iloc[0]
        
        # make a list of maximal length with no events
        event_list = ['-'] * (max_length_steps+1)  
        
        # loop over all events the individual has and put code in correct pos.
        for pos in events['position'].values:
            event_list[pos]=code
        
        event_string = "".join(event_list)
        
        # slice to correct start and end of string (if specified)
        if string_start:
            event_string=event_string[start_position[person]:]
        if string_end:
            event_string=event_string[:-(max_position-end_position[person])]
        return event_string
    
    # new dataframe to store each string for each individual for each code
    string_df=pd.DataFrame(index=code_series[pid].unique()) 
    
    # loop over each code, aggregate strong for each individual, store in df
    for code in codes:
        code_df=code_series[code_series[col].isin([code])]
        stringified=code_df.groupby(pid, sort=False).apply(make_string)
        string_df[code]=stringified
    
    if merge:
        string_df=interleave_strings(string_df)
    return string_df
          
#%%
def read_code2text(path='C:/Users/hmelberg/Google Drive/sandre/resources', 
                   codes='all',
                   capitalized='both',
                   max_length=None): 
    """
    Reads labels for medical codes from files, returns a dictionary code:text
    
    Useful to translate from icd codes (and other codes) to text description
    Reads from semicolonseperated csv files. 
    Files should have one column with 'code' and one with 'text'
    
    
    parameters
    ----------
        codes
            'all' - returns one dictionaty with all codes
            'separate'   - returns a dict of dict (one for each code framework)
    example
        medcodes = read_code2text(codes='all')
                        
    """
    
    paths=['C:/Users/hmelberg/Google Drive/sandre/resources/health_pandas_codes',
           'C:/Users/hmelberg_adm/Google Drive/sandre/resources/health_pandas_codes',
           'C:/Users/sandresl/Google Drive/sandre/resources/health_pandas_codes',
           'C:/Users/sandresl_adm/Google Drive/sandre/resources/health_pandas_codes']
    
    for trypath in paths:
        if os.path.isdir(trypath):
            path=trypath
            break
        
    codeframes='icd atc nc drg'.split()
    
    code2text={}
    
    for frame in codeframes:
        code2text_df = pd.read_csv(f'{path}/{frame}_code2text.csv', 
                         encoding='latin-1', 
                         sep=';')
        code2text_df.code = code2text_df.code.str.strip()
        
        code2text_dict = {**code2text_df[['code', 'text']].set_index('code').to_dict()['text']}
        
        if max_length:
            code2text_dict = {code:text[max_length] for code, text in code2text_dict.items()}
        
        if capitalized=='both':
            capitalized = {str(code).upper():text for code, text in code2text_dict.items()}
            uncapitalized = {str(code).lower():text for code, text in code2text_dict.items()}
            code2text_dict= {**capitalized, **uncapitalized}
        
        code2text[frame]=code2text_dict
        
    if codes=='all':
        code2textnew={}
        for frame in codeframes:
            code2textnew.update(code2text[frame])
        code2text=code2textnew
            
    return code2text    

    

#%%
def get_cleaner(name):
    if name == 'ibd_npr_2015':
        
        cleaner={
            'rename' : {'lopenr':'pid', 'inn_mnd':'start_date'},
            'dates' : ['start_date'],
            'delete' : ['Unnamed: 0'],
            'index' : ['pid'],
            'sort' : ['pid','start_date']
            }

    elif name == 'ibd_npr_2015':
        cleaner={
            'rename' : {'id':'pid', 'innDato':'start_date'},
            'dates' : ['start_date'],
            'delete' : ['Unnamed: 0'],
            'index' : ['pid'],
            'sort' : ['pid','start_date']
            }
    return cleaner

#%%

def clean(df, rename=None, 
              dates=None, 
              delete=None, 
              categorize=None,
              sort=None,
              index=None,
              cleaner:None):
    """
    fix dataframe before use (rename, sort, convert to dates)
    
    """
    if cleaner:
        clean_instructions = get_cleaner(cleaner)
        clean(df, cleaner=None, **clean_instructions)
    
    df=df.rename(columns=rename)
    
    for col in dates:
        df[col]=pd.to_datetime(df[col])
        
    for col in delete:
        del df[col]
        
    df=df.sort_values(sort)
    
    df=df.set_index(index, drop=False)
    return df
