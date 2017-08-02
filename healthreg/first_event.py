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
