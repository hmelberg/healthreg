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


ibd = ['K50, K51']

contains = {'icdmain': ibdlist, 'icdbi':ibdlist}

contains = {('icdmain', 'icdbi'): ibdlist}

config f, arg

default.id_col = 'pid'

#%%

def get_ids(df, id_col='pid', contains=None, query=None, out='set', **kwargs):
        """
        Gets the ids that satisfy some conditions
                
        Parameters
        ----------
            df: (dataframe) Dataframe
            id_col: (string) Column name containnig the ids
            contains: (dictionary) Key is column name to search, value is list of strings to search for
            query: (string) String specifying the query. Example: "year > 2003"
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
        
    if contains:
        combined=np.array([False] * len(df))
        
        for var, searchlist in contains.items():
            searchstr = "|".join(searchlist)
            contains = df[var].str.contains(searchstr, na=False)
            combined = np.logical_or(combined, np.array(contains))
        df=df[combined]
        
    ids = set(df[idcol])
         
    return ids