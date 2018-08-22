# Health registry research
Health registry research is a tool that extends and builds on the Pandas library to make it easier to analyze data on hospital events, prescriptions and similar types of health data. 

### Examples
- **Count the number of unique persons with a diagnosis (using star notation)** 

    ```python
    df.count_persons(codes=['K50*', 'K51*]', cols='icd*')
    
    ```    
- **Select all events for some persons using logical expressions** 

    ```python
    df.select_persons(codes='K50 or K51 and not K52')
    
    df.select_persons(codes='K50 in: icd and 4AB02 in:atc1, atc2')
    
    ``` 

- **Count the number of unique codes in multiple columns with multiple values in each cell**
    ```python
    df['icd_primary', 'icd_secondary'].count_codes(sep=',')
                              
    ```
- **Calculate Charlson Comorbidity Index***
    ```python
    cci = rr.charlson(df=df, cols=['icd1', 'icd2'], sep=',')
                            
    ```
 ### Features
 - **Easy and efficient notation and methods to deal with medical codes:** Medical data often use special code systems to indicate diagnoses, pharmaceuticals and medical procedures. We integrated these tools and allow the use of different types of notation (star, hyphen, colon) to make it easy to select or count the relevant patients. 
 
- **Answer person level question using event level data:** Often health data contains information about events (a prescription, a hospital stay), while the questions we want answered are both at the event-level and person-level: 
    - Event-level: How many doses of a certain pharmaceutical is used in a year?
    - Person-level: How many people have received a given pharmaceutical?
 We have methods, such as `count_persons` that make it easy to get person-level answers from event-level data.
    
- **Deal with messy data:** Sometimes the files supplied to the analysis are multiple large files of messy administrative data. For instance procedure codes can be merged in one column (comma separated) or spread across many columns. To deal with this we have methods that accept both types of data. For instance: the method `count_codes()` can count codes from many columns, some of which may contain comma seperated codes, some of which may be single valued. 
