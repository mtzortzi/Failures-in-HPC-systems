import datetime
import pandas as pd
import csv

DJC_BIG = pd.read_csv('/various/mtzortzi/LogAider-master/RAS-JOB/JOB/DJC_2013/ANL-ALCF-DJC-MIRA_20130409_20131231.csv')

RAS_BIG = pd.read_csv('/various/mtzortzi/LogAider-master/RAS-JOB/RAS/bigcsv/ANL-ALCF-RE-MIRA_20130409_20131231.csv')

print('columns of DIM JOB COMPOSITE: ',DJC_BIG.columns)
print('colums of RAS: ',RAS_BIG.columns)

print('I rename the column of DJC, from COBALT_JOBID to JOBID')
DJC_BIG.rename(columns = {'COBALT_JOBID':'JOBID'}, inplace = True)

print('I rename the column of DJC, from LOCATION to BLOCK, so as to have a common key for matching the two dataframes')
DJC_BIG.rename(columns = {'LOCATION':'BLOCK'}, inplace = True)
print('new columns of DJC', DJC_BIG.columns)


# I create a dataframe with the following columns
merged = pd.DataFrame(columns=['SEVERITY', 'EVENT_TIME', 'START_TIMESTAMP', 'END_TIMESTAMP','LOCATION', 'BLOCK', 'MESSAGE'])
for i, rowDJC in enumerate(DJC_BIG.itertuples(), 1):
    print('new iteration for DJC: ',i)
    for j, rowRAS in enumerate(RAS_BIG.itertuples(), 1):
        if (rowDJC.BLOCK == rowRAS.BLOCK):
            print('find same block')
            if ((rowDJC.START_TIMESTAMP < rowRAS.EVENT_TIME) and (rowRAS.EVENT_TIME <= rowDJC.END_TIMESTAMP)):
                print('find the condition')
                merged = merged.append({'SEVERITY': rowRAS.SEVERITY, 'EVENT_TIME': rowRAS.EVENT_TIME, 'START_TIMESTAMP': rowDJC.START_TIMESTAMP, 'END_TIMESTAMP': rowDJC.END_TIMESTAMP, 'LOCATION': rowRAS.LOCATION, 'BLOCK': rowRAS.BLOCK, 'MESSAGE': rowRAS.MESSAGE}, ignore_index=True)
            

merged.to_csv('/various/mtzortzi/LogAider-master/RAS-JOB/merged.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
