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
    return a list if the input is a string, if not: returns the input as it was

    Args:
        string_or_list (str or any):

    Returns:  
        A list if the input is a string, if not: returns the input as it was

    Note:
        - allows user to use a string as an argument instead of single lists
        - cols='icd10' is allowed instead of cols=['icd10']
        - cols='icd10' is transformed to cols=['icd10'] by this function

    """
    if isinstance(string_or_list, str):
        string_or_list = [string_or_list]
    return string_or_list

#%%
def sample_persons(df, pid='pid', n=None, frac=0.1):
    """
    Pick some (randomly selected) individuals and all their observations
    
    Args:
        n (int): number of individuals to sample
        frac (float): fraction of individuals to sample
    
    Returns:
        dataframe
        
    Note:
        Default: take a 10% sample
    
    Examples:
        sample_df=sample_persons(df, n=100)
    
    """
        
    ids=df.pid.unique()
    
    if not n:
        n=int(frac*len(ids))
    
    new_ids=np.random.choice(ids, size=n)
    new_sample = df[df.pid.isin(new_ids)]
    return new_sample


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
           cols=None,
           sep=None,
           strip=True,
           name=None):        
        
    """
    Get set of unique values from column(s) with multiple valules in cells
    
    Args:
        df (dataframe)
        cols (str or list of str): columns with  content used to create unique values
        sep (str): if there are multiple codes in cells, the separator has to 
             be specified. Example: sep =',' or sep=';'
             
    
    Returns:
        
    Note:
        - Each column may have multiple values (if sep is specified)
        - Star notation is allowed to describe columns: col='year*'
        - In large dataframes this function may take some time 

    
    Examples
        
        to get all unique values in all columns that start with 'drg':
            drg_codeds=unique_codes(df=df, cols=['drg*'])
        
        all unique atc codes from a column with many comma separated codes
             atc_codes==unique_codes(df=ibd, cols=['atc'], sep=',')
    
    worry
        numeric columns/content
    """
    if not cols:
        if isinstance(df, pd.Series):
            df=pd.DataFrame(df)
            cols=list(df.columns)
        elif isinstance(df, pd.DataFrame):
            cols=list(df.columns)
    else:
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
    
    single_cols = infer_single_value_columns(df=df, 
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
          sep=None,
          codebook=None):
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
    
    if codebook:
        unique_words = set(codebook)  
    else:
        unique_words=unique_codes(df=df, cols=cols, sep=sep)
    
    #expand only if there is something to expand
    if '*' not in ''.join(codes):
        matches = set(codes) & unique_words
    
    else:
        matches=set()
        
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
    return pids 

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
    
    if not replace:
        replace=codes
        
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
    """
    Expand columns with star notation to their full column names
    """
        
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
def single_columns(df, cols=None, sep=',', n=100, check_all=False):
    """
    Identify columns that do not have seperators i.e. single values in cells
    
    Args:
        cols (list of strings): columns to be examined for seperators, 
        default: all columns
        
        sep (string): seperator that may be used to distinguish values in cells
        n (int): check only a subsample (head and tail) of n observations
        check_all (bool): check all observations
    
    Returns:
        list
    
    """
    
    if not cols:
        cols = list(df.columns)
        
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
def first_event(df, codes, cols=None, pid = 'pid', date='in_date', sep=None):
    """
        Returns time of the first observation for the person based on certain 
        
        Args:
            codes (str or list of str or dict): 
                codes marking the event of interest. Star notation is ok.
            pid (string): Patient identifier
            date (string): name of column indicatinf the date of the event 
        
        Returns:
            Pandas series
            
        
        Examples:
            first_event(id_col = 'pid', date_col='diagnose_date', groupby = ['disease'], return_as='dict')
            
            df['cohort'] = df.first_event(id_col = 'pid', date_col='diagnose_date', return_as='dict')
            
            Date of first registered event with an ibd code for each individual:
                first_event(df=df, codes=['k50*','k51*'], cols='icd', date='date')   
    """
    codes=listify(codes)
    cols=listify(cols)
    
    cols = expand_columns(df=df, cols=cols)
    codes=expand_codes(df=df,codes=codes,cols=cols, sep=sep)
    
    rows_with_codes=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subdf=df[rows_with_codes]
    
    #groupby.extent(pid)
    first_date=subdf[[pid, date]].groupby(pid, sort=False)[date].min()
    
    return first_date
 
#%%    
def stringify_durations(df,
              codes=None, 
              cols=None, 
              pid='pid',
              step=120,
              sep=None,
              
              event_start='in_date',
              event_end=None,
              event_duration='ddd',
              
              first_date=None,
              last_date=None,
              censored_date=None,
              
              na_rep='-',
              time_rep=',',
              
              merge=True,
              info=None,
              report=True):
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
    
    eventstr_duration = stringify_durations(df=pr, codes=codes, cols='atc',  
                                           event_duration='ddd', 
                                        event_start='date', step=120)
    """
    # drop rows with missing observations in required variables
    
    if report:
        obs = len(df)
        npid = df[pid].nunique()
        rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
        code_obs=len(df[rows])
        code_npid=df[rows][pid].nunique()
    
    df=df.dropna(subset= [pid, event_start])    
    
    if event_end:
        df=df.dropna(subset = [event_end])
    elif event_duration:
        df=df.dropna(subset = [event_duration])
        if df[event_duration].min()<0:
            print('Error: The specified duration column contains negative values')      
    else:
        print('Error: Either event_end or event_duration has to be specified.')
    
    # find default min and max dates 
    # will be used as starting points for the string 
    # if first_date and last_date are not specified 
    min_date=df[event_start].min()
    max_date=df[event_start].max()

    # drop rows outside specified time period of interest
    if first_date:
        if first_date in df.columns:
            df=df[df[event_start]>=df[first_date]]
        elif isinstance(first_date, dict):
            pass
        else:
            #if first_date is not a column name, it is assumed to be a date
            try:
                min_date=pd.to_datetime(first_date)
                df=df[df[event_start]>=min_date]
            except:
                print('Error: The first_date argument has to be on of: None, a dict, a column name or a string that represents a date')
        
    if last_date:
        if last_date in df.columns:
            df=df[df[event_start]>=df[last_date]]
        elif isinstance(last_date, dict):
            pass
        else:
            try:
                max_date=pd.to_datetime(last_date)
                df=df[df[event_start]<=max_date]
            except:
                print('Error: The last_date argument has to be on of: None, a dict, a column name or a string the represents a date')

    
    # note an individual min date cannot be before overall specified min date
    # should raise error if user tries this
    # same with max: individual cannot be larger than overall
    
    max_length_days = (max_date-min_date).days
    max_length_steps = int(max_length_days/step)       
    
    # if codes are not specified, use the five most common codes
    if not codes:
        cols=expand_columns(listify(cols))
        codes=count_codes(df=df, cols=cols, sep=sep).sort_values(ascending=False)[:4]
        
    # fix formatting of input (make list out of a string input and so on)
    codes, cols, old_codes, replace = fix_args(df=df,codes=codes, cols=cols, sep=sep)
    
    # get the rows that contain the relevant codes
    rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subset=df[rows].copy() # maybe use .copy to avoid warnings? but takes time and memory
    if report:
        sub_obs = len(subset)
        sub_npid = subset[pid].nunique()
        
    subset=subset.sort_values([pid, event_start])
    subset=subset.set_index(pid, drop=False)
    subset.index.name='pid_index'
    
    # find start and end position of each event (number of steps from overall min_date)
    # to do: do not use those column names (may overwrite original names), use uuid names?
    subset['start_position']=(subset[event_start]-min_date).dt.days.div(step).astype(int)
   
    if event_end:
        subset['end_position']=(subset[event_end]-min_date).dt.days.div(step).astype(int)
    elif event_duration:
        subset['end_date'] = subset[event_start] + pd.to_timedelta(subset[event_duration].astype(int), unit='D')
        subset['end_position']=(subset['end_date']-min_date).dt.days.div(step).astype(int)
        
   # to do: may allow duration dict?
   # for instance: some drugs last 15 days, some drugs last 25 days . all specified in a dict
   
    
    # create series with only the relevant codes for each person and position 
    code_series=extract_codes(df=subset.set_index([pid, 'start_position', 'end_position']), 
                              codes=replace, 
                              cols=cols, 
                              sep=sep, 
                              new_sep=',', 
                              merge=True, 
                              out='text')
    
    # May need to unstack if two events in same row
    # for now: Just foce it to be 1
    if code_series.apply(len).max()>1:
        code_series=code_series.str[0]

    # base further aggregation on the new extracted series with its col and codes
    col=code_series.name
    codes=code_series.name.split(', ')

    # drop duplicates (same type of even in same period for same individual)
    code_series=code_series.reset_index().drop_duplicates().set_index(pid, drop=False)
    
    ## make dict with string start and end positions for each individual
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

    if first_date:
        # if a column is specified
        if first_date in subset.columns:
            start_date=subset.groupby(pid)[first_date].first().dropna().to_dict()
        # do nothing if a dict mapping pids to last_dates is already specified
        elif isinstance(first_date, dict):
            pass
        # if a single overall date is specified
        else:
            date=pd.to_datetime(first_date)
            start_date={pid:date for pid in subset[pid].unique()}
        # convert start date to start position in string
        string_start_position={pid:int((date-min_date).days/step) 
                        for pid, date in start_date.items()}  
                
    if last_date:
        if last_date in subset:
            end_date=subset.groupby(pid)[last_date].first().dropna().to_dict()
        # do nothing if a dict mapping pids to last_dates is already specified
        elif isinstance(last_date, dict):
            pass
        else:
            date=pd.to_datetime(last_date)
            end_date={pid:date for pid in subset[pid].unique()}
        # convert date to position in string
        string_end_position={pid:(date-min_date).dt.days.div(step).astype(int) 
                        for pid, date in end_date.items()} 
    
    # takes dataframe for an individual and makes a string with the events    
    def make_string(events):
        # get pid of individual (required to find correct start and end point)
        person=events[pid].iloc[0]
        
        # make a list of maximal length with no events
        event_list = ['-'] * (max_length_steps+1)  
        
        from_to_positions = tuple(zip(events['start_position'].tolist(), events['end_position'].tolist()))
        
        # loop over all events the individual has and put code in correct pos.
        for pos in from_to_positions:
            event_list[pos[0]:pos[1]]=code
        event_string = "".join(event_list)
        
        # slice to correct start and end of string (if specified)
        if first_date:
            event_string=event_string[string_start_position[person]:]
        if last_date:
            max_position=int((max_date-min_date).days/step)
            event_string=event_string[:-(max_position-string_end_position[person])]
        return event_string
    
    # new dataframe to store each string for each individual for each code
    string_df=pd.DataFrame(index=code_series[pid].unique()) 
    
    # loop over each code, aggregate strong for each individual, store in df
    for code in codes:
        code_df=code_series[code_series[col].isin([code])]
        code_df.index.name='pid_index' # avoid future error from pandas pid in both col and index
        stringified=code_df.groupby(pid, sort=False).apply(make_string)
        string_df[code]=stringified
    
    if merge:
        string_df=interleave_strings(string_df)
        
    if report:
        final_obs = len(subset)
        final_npid = len(string_df)
        print(f"""
                                     events,  unique ids
              Original dataframe     {obs}, {npid} 
              Filter codes           {code_obs}, {code_npid}
              Filter missing         {sub_obs}, {sub_npid}
              Final result:          {final_obs}, {final_npid}""") 
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
    units = (events[i:i+agg] for i in range(0, len(events), agg))
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
    (change a series to a df and set cols?)
    (if not codes: pick top 5 and use those as codes)
    
    """
    # use all cols if not specified 
    # assumes index is pid ... maybe too much magic?
    if not cols:
        # if a series, convert to dataframe to make it work? 
        # experimental, maybe not useful
        if isinstance(df, pd.Series):
            df=pd.DataFrame(df)
            cols=list(df.columns)    
        # if a dataframe, use all cols (and take index as pid)
        else:
            cols=list(df.columns)
        df['pid']=df.index.values
        
    # if codes is not specified, use the five most common codes
    if not codes:
        cols=expand_columns(df=df, cols=listify(cols))
        codes=count_codes(df=df, cols=cols, sep=sep).sort_values(ascending=False)[:5]
    
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
    
    Can produce a set of dummy columns for codes and code groups.
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
def label(df, labels=None, read=True, path=None):
    """
    Translate codes in index to text labels based on content of the dict labels
    
    Args:
        labels (dict): dictionary from codes to text
        read (bool): read and use internal dictionary if no dictionary is provided
    """
    if not labels:
        # making life easier for myself
        try:
            labels=read_code2text()
        except:
            labels=read_code2text(path)
    df = df.rename(index=labels)
    return df

#%%  

def count_codes(df, codes=None, cols=None, sep=None, strip=True, ignore_case=False, normalize=False, ascending=False):
    """
    Count frequency of values in multiple columns or columns with seperators
    
    Args:
        codes (str, list of str, dict): codes to be counted
        cols (str or list of str): columns where codes are
        sep (str): separator if multiple codes in cells
        strip (bool): strip spacec bore and after code before counting
        ignore_case (bool): determine if codes with same characters, 
            but different cases should be the same
        normalize (bool): If True, outputs percentages and not absolute numbers        
        
    allows 
        - star notation in codes and columns
        - values in cells with multiple valules can be separated (if sep is defined)
        - replacement and aggregation to larger groups (when code is a dict)
     
    example
    To count the number of stereoid events (codes starting with H2) and use of 
    antibiotics (codes starting with xx) in all columns where the column names
    starts with "atc":
        
    count_codes(df=df, 
                 codes={'H2*': 'stereoids, 'AI*':'antibiotics'},
                 cols='atc*', 
                 sep=',')
    
    more examples
    -------------
    
    count_codes(df, codes='Z51*', cols='icdmain', sep=None)
    count_codes(df, codes='Z51*', cols=['icdmain', 'icdbi'], sep=None)
    count_codes(df, codes='Z51*', cols=['icdmain', 'icdbi'], sep=',')
    count_codes(df, codes={'Z51*':'stråling'}, cols=['icdmain', 'icdbi'], sep=',')
    """
    # count all if codes is codes is not specified
    # use all columns if col is not specified 
    if codes: 
        codes=listify(codes)
    
    if not cols:
        if isinstance(df,pd.Series):
            df=pd.DataFrame(df)
            cols=list(df.columns)
        elif isinstance(df, pd.DataFrame):
            cols=list(df.columns)
    else:
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
    
    if ascending:
        code_count = code_count.sort_values(ascending=True)
    else:
        code_count = code_count.sort_values(ascending=False)
        
    
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

#%%
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

#%%
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
            
    #a.groupby(1)[2].value_counts()
    return t1


#%%
def stringify_time(df, 
                         codes=None, 
                         cols=None, 
                         pid='pid',
                         step=90,
                         
                         event_start='in_date', 
                         event_end=None,
                         
                         
                         first_date=None,
                         last_date=None,
                         
                         censored_date=None,
                         
                         sep=None,
                         merge=True,
                         meta=None):
    """
    Creates a string for each individual describing events at position in time
    
    Arge:
        df: dataframe
        codes: codes to be used to mark an event
        cols: columns with the event codes
        pid: column with the personal identification number
        event_date: column containing the date for the event
        sep: the seperator used between events if a column has multiple events in a cell
        keep_repeats: identical events after each other are reduced to one (if true)
        only_unique: deletes all events that have occurred previously for the individual (if true)
    
    Returns:
        series with a string that describes the events for each individual
    
    Example:
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
            first_date='diagnosis_date',
            step=90,
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
            first_date='first_bio',
            step=90,
            sep=','
            )
    
    """

    # drop rows with missing observations in required variables
    df=df.dropna(subset= [pid, event_start])
    
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
            df=df[df[event_start]<=max_date]
    
    # note an individual min date cannot be before overall specified min date
    # should raise error if user tries this
    # same with max: individual cannot be larger than overall
    
    max_length_days = (max_date-min_date).days
    max_length_steps = int(max_length_days/step)       
    
    # if codes are not specified, use the five most common codes
    if not codes:
        cols=expand_columns(listify(cols))
        codes=count_codes(df=df, cols=cols, sep=sep).sort_values(ascending=False)[:4]
        
    # fix formatting of input (make list out of a string input and so on)
    codes, cols, old_codes, replace = fix_args(df=df,codes=codes, cols=cols, sep=sep)
    
    # get the rows that contain the relevant codes
    rows=get_rows(df=df, codes=codes, cols=cols, sep=sep)
    subset=df[rows] # maybe use .copy to avoid warnings?
    
    # find position of each event (number of steps from overall min_date)
    subset['position']=(subset[event_start]-min_date).dt.days.div(step).astype(int)

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

    if first_date:
        # if a column is specified
        if first_date in subset.columns:
            start_date=subset.groupby(pid)[first_date].first().dropna().to_dict()
        # if a single overall date is specified
        else:
            date=pd.to_datetime(first_date)
            start_date={pid:date for pid in subset[pid].unique()}
        # convert start date to start position in string
        start_position={pid:int((date-min_date).days/step) 
                        for pid, date in start_date.items()}  
                
    if last_date:
        if last_date in subset:
            end_date=subset.groupby(pid)[last_date].first().dropna().to_dict()
        else:
            date=pd.to_datetime(last_date)
            end_date={pid:date for pid in subset[pid].unique()}
        # convert date to position in string
        end_position={pid:(date-min_date).dt.days.div(step).astype(int) 
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
        if first_date:
            event_string=event_string[start_position[person]:]
        if last_date:
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
def charlson(df, cols=None, sep=None, dot_notation=False):
    """
    
    Reference:
        http://isocentre.wikidot.com/data:charlson-s-comorbidity-index
    
    a=charlson(df=df, cols=['icdmain', 'icdbi'], sep=',', dot_notation=False)
    
    """
    infarct = 'I21* I22* I25.2'.split()
    heart = 'I09.9 I11.0 I13.0, I13.2 I25.5 I42.0 I42.5-I42.9 I43* I50* P29.0'.split()
    vascular = '70* I71* I73.1 I73.8 I73.9 I77.1 I79.0 I79.2 K55.1 K55.8 K55.9 Z95.8 Z95.9'.split()
    cerebro = 'G45* G46* H34.0 I60* I69*'.split()
    dementia	= 'F00* F03* F05.1 G30* G31.1'.split()
    pulmonary = 'I27.8 I27.9 J40* J47* J60* J67* J68.4, J70.1, J70.3'.split()
    tissue = 'M05* M06* M31.5 M32* M34* M35.1, M35.3 M36.0'.split()
    ulcer = 	['K25*-K28*']
    liver = 	'B18* K70.0-K70.3 K70.9 K71.3-K71.5 K71.7 K73* K74* K76.0 K76.2-K76.4 K76.8 K76.9 Z94.4'.split()
    diabetes =	['E10.0', 'E10.l', 'E10.6', 'E10.8', 'E10.9', 'E11.0', 'E11.1', 'E11.6', 'E11.8', 'E11.9', 'E12.0', 'E12.1', 'El2.6', 'E12.8', 'El2.9', 'E13.0', 'E13.1', 'E13.6', 'E13.8', 'E13.9', 'E14.0', 'E14.1', 'E14.6', 'E14.8', 'E14.9']
    hemiplegia = 	 ['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81*', 'G82*', 'G83.0-G83.4', 'G83.9']
    renal = 	['I12.0', 'I13.1', 'N03.2-N03.7', 'N05.2-N05.7', 'N18*', 'N19*', 'N25.0', 'Z49.0-Z49.2', 'Z94.0', 'Z99.2']
    dorgan = ['E10.2','E10.3','E10.4', 'E10.5', 'E10.7', 'E11.2', 'E11.5', 'E11.7', 'E12.2', 'E12.3', 'E12.4', 'E12.5', 'E12.7','E13.2','E13.3', 'E13.4', 'E13.5', 'E13.7', 'E14.2','E14.3', 'E14.4', 'E14.5', 'E14.7']
    tumor	=['C00*-C26*', 'C30*-C34*', 'C37*-41*', 'C43*-C45*', 'C58*-C60*', 'C76*-C81*', 'C85*-C88*', 'C90*-C97*']
    sliver =  ['I85.0', 'I85.9', 'I86.4', 'I98.2', 'K70.4', 'K71.1', 'K72.1', 'K72.9', 'K76.5', 'K76.6', 'K76.7']
    mtumor = 	['C77*','C78*','C79*','C80*']
    hiv = 	['B20*', 'B21*', 'B22*', 'B24']

    points = {
        'infarct' : 1,
        'heart' : 1,
        'vascular' : 1,
        'cerebro' : 1,
        'dementia' : 1,
        'pulmonary' : 1,
        'tissue' : 1,
        'ulcer' : 	1,
        'liver' : 	1,
        'diabetes' :	1,
        'hemiplegia' : 	2,
        'renal' : 	2,
        'dorgan' : 2,
        'tumor'	:2,
        'sliver' :  3,
        'mtumor' : 	6,
        'hiv' : 	6}
    
    disease_labels = list(points.keys())
    
    diseases= [
        infarct  ,
        heart  ,
        vascular  ,
        cerebro  ,
        dementia  ,
        pulmonary  ,
        tissue  ,
        ulcer  	,
        liver  	,
        diabetes 	,
        hemiplegia  	,
        renal  	,
        dorgan  ,
        tumor	,
        sliver   ,
        mtumor  	,
        hiv  	]
    
    disease_codes={}
    for i, disease in enumerate(diseases):
        all_codes=[]
        disease_str=disease_labels[i]
        for code in disease:
            expanded_codes = expand_hyphen(code)
            all_codes.extend(expanded_codes)
        disease_codes[disease_str] = all_codes
    
    expanded_disease_codes = {}  
    no_dot_disease_codes={}
    
    if not dot_notation:
        for disease, codes in disease_codes.items():
            new_codes = [code.replace('.','') for code in codes]
            no_dot_disease_codes[disease] = new_codes
        disease_codes = no_dot_disease_codes
    
    all_codes = unique_codes(df=df, cols=cols, sep=sep)          
    
    for disease, codes in disease_codes.items():
        expanded_disease_codes[disease] = expand_codes(df=df, codes=codes, cols=cols, sep=sep, codebook=all_codes)
    
    return expanded_disease_codes


#%%

def expand_hyphen(expr):
    """
    Example: Expands ('b01A-b04A') to ['b01A' ,'b02A', 'b03A', 'b04A']
    
    Args:
        code
        
    Returns:
        
    Examples:
        expand_hyphen('b01.1*-b09.9*')
        expand_hyphen('n02.2-n02.7')  
        expand_hyphen('c00*-c260') 
        expand_hyphen('b01-b09')
        expand_hyphen('b001.1*-b009.9*')
    
    Note:
        decimal expression also works: expr = 'n02.2-n02.7'
        expr = 'b01*-b09*'
        expr = 'C00*-C26*'
    Todo:
        expr = 'n00002667600.2-n05.7'
        expr = 'b001.1*-b009.9*'
    """
    if '-' in expr:
        lower, upper = expr.split('-')
        lower_str = re.search("[-+]?\d*\.\d+|\d+", lower).group()
        upper_str = re.search("[-+]?\d*\.\d+|\d+", upper).group()
        
        lower_num = float(lower_str)
        upper_num = float(upper_str)
       
        #leading_nulls = len(lower_str) - len(lower_str.lstrip('0'))
        length = len(lower_str)
        
        # must use integers in a loop, not floats
        if '.' in lower_str:
            decimals = len(lower_str.split('.')[1])
            multiplier = 10*decimals
        else:
            multiplier=1
               
        no_dec_lower = int(lower_num*multiplier)
        no_dec_upper = int((upper_num)*multiplier)+1
        
        if '.' in lower_str:
            codes = [lower.replace(lower_str, str(num/multiplier).zfill(length)) for num in range(no_dec_lower, no_dec_upper)]
        else:
            codes = [lower.replace(lower_str, str(num).zfill(length)) for num in range(no_dec_lower, no_dec_upper)]

            
    else:
        codes = [expr]
    return codes

#%%


def clean(df, rename=None, 
              dates=None, 
              delete=None, 
              categorize=None,
              sort=None,
              index=None,
              cleaner=None):
    """
    fix dataframe before use (rename, sort, convert to dates)
    
    """
    if cleaner:
        clean_instructions = get_cleaner(cleaner)
        clean(df, cleaner=None, **clean_instructions)
    
    if rename:
        df=df.rename(columns=rename)
    
    if dates:
        for col in dates:
            df[col]=pd.to_datetime(df[col])
        
    if delete:
        for col in delete:
            del df[col]
        
    if sort:
        df=df.sort_values(sort)
    
    if index:
        df=df.set_index(index, drop=False)
        
    
    return df

#%% monkeypatch to the functions become methods on the dataframe
# could use decorators/pandas_flavor
# pandas_flavor: good, but want to eliminate dependencies
# approach below may be bloated and harder to maintain 
# (must change when method names change)
series_methods =[count_persons, unique_codes, extract_codes, count_codes, label] 

frame_methods = [sample_persons, first_event, get_pids, unique_codes, 
                 expand_codes, get_rows, count_persons, stringify, 
                 extract_codes, count_codes, label]

# probably a horrible way of doing something horible!
for method in frame_methods:
    setattr(pd.DataFrame, getattr(method, "__name__"), method)

for method in series_methods:
    setattr(pd.Series, getattr(method, "__name__"), method)
