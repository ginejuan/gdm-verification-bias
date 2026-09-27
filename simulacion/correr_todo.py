import warnings; warnings.filterwarnings('ignore')
from sim_verificacion import *
import sys
confs=[('S1','universal',True),('S1','PAI',True),('S1','PAI+edad',True),('S1','NICE',True),('S1','PAI',False),
       ('S2','universal',True),('S2','PAI',True),('S2','PAI+edad',True),('S2','NICE',True),('S2','PAI',False)]
i=int(sys.argv[1]); esc,pol,g=confs[i]
R=correr(esc,pol,reps=500,N=4000,usar_glu=g,seed=100+i,fuera=0.12)
R.to_csv(f"res_{i}.csv",index=False); print("ok",i,len(R))
