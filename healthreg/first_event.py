def first_event(df, search_col, find,id_col = 'pid', date_col='in_date'):
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
        
        Returns
            Pandas dataframe or a dictionary
            
        
    """
      
    b = np.full(len(df),False)

    for col in search_col:
        for code in find:
            a = df[col].str.contains(code,na=False).values
            b = b|a
    
    return df[b].groupby(id_col)[date_col].min()


first_event(df,search_col = ['bio','procedure_codes'],find = ['JHJ','infli','WMG'])

   
    
    
df[['k50','k51']].isin([1,2])

df[['k50','k51']].isin([1,2]) == df[id_col].isin(find)

