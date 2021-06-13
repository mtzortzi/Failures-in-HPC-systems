import pandas as pd
import csv

#phase1_nodewise = pd.read_csv('/various/mtzortzi/from_nikela/phase1_nodewise_wids.csv')
phase2 = pd.read_csv('/various/mtzortzi/from_nikela/phase2.csv')
phase3 = pd.read_csv('/various/mtzortzi/from_nikela/phase3.csv')
phase3_wids = pd.read_csv('/various/mtzortzi/from_nikela/phase3_wids.csv')

#print(set(phase1_nodewise['MULTIWORDS']))
#print(len(set(phase1_nodewise['MULTIWORDS'])))


from nltk import FreqDist

#freq_dist_pos = FreqDist(phase1_nodewise['MULTIWORDS'])
#print(freq_dist_pos.most_common(5))
#print(len(phase1_nodewise['MULTIWORDS']))



print(set(phase2['WIDS']))
#print(set(phase2['MULTIWORDS']))
#print(len(set(phase2['MULTIWORDS'])))

print('--------------------------------------------------------------------------')

print(set(phase3_wids['WIDS']))
print(set(phase3['MULTIWORDS']))
print(len(set(phase3['MULTIWORDS'])))


from nltk import FreqDist

freq_dist_pos = FreqDist(phase3['MULTIWORDS'])
print(freq_dist_pos.most_common(5))
print(len(phase3['MULTIWORDS']))

