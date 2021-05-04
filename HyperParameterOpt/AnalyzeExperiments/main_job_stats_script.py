import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import ceil

def get_dates(s):
    l = s.split(' ')
    date = l[0].split('-')
    year = int(date[0])
    month = int(date[1])
    month_day = int(date[2])
    return np.array([year,month,month_day])

def get_hours(a):
    """ get the max number of hours
        could have number of days in it
    """
    if len(a) == 3:    
        if '-' in a[0]:
            return int(a[0][0])*24 + int(a[0][-1])
        else:
            return int(a[0])
    else:
        return 0
    
def get_minutes(a):
    """ get total minutes """
    if len(a) == 3:    
        if '-' in a[0]:
            return (int(a[0][0])*24 + int(a[0][-1])) * 60
        else:
            return (int(a[0])) * 60
    elif len(a) == 2:
        return ceil(int(a[0]))
    else:
        return int(a[0]) / 60

def job_stats(file,drop_null=False):
    """ """
    path ='Data/'
    if file[-4:] == '.csv':
        file = file[:-4]
    df = pd.read_csv(path + file + '.csv')
    df.columns = [x.lower().replace(' ','_') for x in df.columns]
    
    
    l = [l.split('_') for l in df.job_id]
    df['batch_id'] = [a[0] for a in l ]
    df['task_id'] = [a[1] if len(a)==2 else None for a in l ]
    print('note that batch_id and task_id are STRINGS')
    recent = df.job_name.unique()[:50]
    results = {}
    for i in range(len(recent)):
        results[recent[i]] = df.loc[df['job_name'] == recent[i]]['job_state'].value_counts().copy()
    print(results.keys())
    df2 = pd.DataFrame()
    for i in results.keys():
        df2 = pd.concat([df2,results[i]],axis=1,sort=False)
    df2.columns = recent
    df2.fillna(value=0,inplace=True)
    new = df2.T
    new['num_jobs'] = new.sum(axis=1)
    new['complete%'] = round(new['COMPLETE'] / new.num_jobs,3) * 100
    new['timeout%'] = round(new['TIMEOUT'] / new.num_jobs,3) * 100
    new['failed%'] = round(new['FAILED'] / new.num_jobs,3) * 100
    try:
        new['out_memory%'] = round(new['OUT_OF_MEMORY'] / new.num_jobs,3) * 100
    except:
        pass
    
    df['list_dates'] = df.time_job_finished.apply(get_dates)
    df['year'] = [x[0] for x in df.list_dates]
    df['month'] = [x[1] for x in df.list_dates]
    df['day'] = [x[2] for x in df.list_dates]
    
    if drop_null:
        nulls = df.loc[df.actual_walltime.isna()].index
        print('\n\nthere are',len(nulls),'nulls')
        df.drop(index=nulls,inplace=True)
        # there cannot be any null values for this to work
        times = [x.split(':') for x in df['actual_walltime']]
        df['walltime_hours'] = [get_hours(a) for a in times]
        df['walltime_total_minutes'] = [get_minutes(a) for a in times]
    print(df.shape,df.columns,sep='\n')
    
    return (df,new)
  
f='12-3-jobstats'
df,new = job_stats(f,drop_null=True)
hrs = df.walltime_total_minutes.sum() / (60 * 24*365)
print(f'\n\nive used {round(hrs,2)} years of compute time on super-computer')
new.reset_index(inplace=True) 
new.rename(columns={'index':'job_name'},inplace=True)

show = ['job_name','complete%', 'timeout%',
       'failed%','out_memory%','num_jobs']
show = ['job_name','complete%', 'timeout%',
       'failed%','num_jobs']
new = new[show]
# new.sort_values(by='job_name',inplace=True)
new

