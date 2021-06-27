import numpy as np
import pandas as pd 
import random
import operator

# DECLARE transition probabilities, states and emission probabilities here.

pi1 = [0.0,1.0,0.0,0.0]
pi2=[0.0,0.0,0.0,1.0]
a1 =[[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.4,0.3,0.3],[0.3,0.2,0.2,0.3]]
a2 = [[1.0,0.0,0.0,0.0],[0.1,0.3,0.5,0.1],[0.1,0.4,0.3,0.2],[0.1,0.4,0.2,0.3]]
b1 =[[1.0,0.0,0.0,0.0,0.0],[0.0,0.5,0.5,0.0,0.0],[0.0,0.2,0.2,0.3,0.3],[0.0,0.0,0.0,0.5,0.5]]
b2 = [[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.5,0.0,0.5],[0.0,0.0,0.5,0.5,0.0],[0.0,0.5,0.0,0.0,0.5]]
h_state = {1,2,3,4}
o_state = {'S','A','B','C','D'}

trans_mat1= pd.DataFrame(a1, columns = [1 , 2, 3, 4], index=[1, 2, 3,4])
trans_mat2= pd.DataFrame(a2, columns = [1 , 2, 3, 4], index=[1, 2, 3,4])
emis_1=  pd.DataFrame(b1, columns = ['S','A','B','C','D'], index=[1, 2, 3,4])
emis_2=  pd.DataFrame(b2, columns = ['S','A','B','C','D'], index=[1, 2, 3,4])


print(trans_mat1,'\n')
print(trans_mat2,'\n')
print(emis_1,'\n')
print(emis_2,'\n')

# Generate Observations from defined HMM

def gen_one_obs():
  start = {}
  obs_for_iteration = []
  for i in range(0,4):
    if pi1[i]!=0:
      start[i+1] = pi1[i]
  ch_start = random.choice(list(start.keys()))
  curr_hstate = ch_start
  curr_obs = ''
  
  while(curr_obs!='S'):
    #print(curr_hstate)
    obs_feas = emis_1.loc[curr_hstate]
    obs_ch = obs_feas[obs_feas!=0.0].index
    curr_obs = random.choice(obs_ch)
    obs_for_iteration.append(curr_obs)
    next_list = trans_mat1.loc[curr_hstate]
    next_states = next_list[next_list!=0.0].index
    curr_hstate = random.choice(next_states)
    #print( curr_obs)
    
  return obs_for_iteration

for i in range(1,11):
  obs = gen_one_obs()
  print(i,'. ', obs)

#   FORWARD ALGORITHM

def forward_algorithm(obs):
  T = len(obs)
  nodes_n = 4
  forward_1 = pd.DataFrame(index =[1,2,3,4], columns = [i for i in range(1,T+1)])
  forward_2 = pd.DataFrame(index =[1,2,3,4], columns = [i for i in range(1,T+1)])
  #forward_1 = forward_1.fillna(0)
  #forward_2 = forward_2.fillna(0)
  for s in range(1,5):
    forward_1.at[s,1]= pi1[s-1]*emis_1.loc[s][obs[0]]
    forward_2.loc[s][1] = pi2[s-1]*emis_2.loc[s][obs[0]]
    
  for t in range(2,T):
    for s in range(1,5):
      sum_1 = 0
      sum_2 = 0
      
      for s_prev in range(1,5):
        sum_1 = sum_1 + forward_1.loc[s_prev][t-1]*trans_mat1.loc[s_prev][s]*emis_1.loc[s][obs[t-1]]
        sum_2= sum_2 + forward_2.loc[s_prev][t-1]*trans_mat2.loc[s_prev][s]*emis_2.loc[s][obs[t-1]]
      forward_1.at[s,t] = sum_1
      forward_2.at[s,t] = sum_2
      
  tot_p1=0
  tot_p2=0
  
  for s in range(1,5):
    tot_p1 = tot_p1  +  forward_1.loc[s][T-1]*trans_mat1.loc[s][1]
    tot_p2 = tot_p2  +  forward_2.loc[s][T-1]*trans_mat2.loc[s][1]
  forward_1.at[1,T] = tot_p1
  forward_2.at[1,T] = tot_p2
  
#   Check which HMM produced the observed outpu
obs_list= [['A','D','C','B','D','C','C','S'],['B','D','S'],['B','C','C','B','D','D','C','A','C','S'],['A','C','D','S'],['A','D','A','C','S'],['D','B','B','S'],['A','B','S'],['D','D','B','D','D','B','A','C','C','D','A','B','B','C','D','B','B','B','S'],['D','B','D','S'],['A','A','A','A','D','C','B','S']]
class_label = []
for i in range(0,len(obs_list)):
  P_O_1, P_O_2 = forward_algorithm(obs_list[i])
  if P_O_1>P_O_2:
    class_label.append(1)
  elif P_O_1<P_O_2:
    class_label.append(2)
  else:
    class_label.append(random.randint(1,3))
    
print(class_label)

# VITERBI ALGORITHM
def viterbi_algorithm(obs):
  T = len(obs)
  nodes_n = 4
  viterbi = pd.DataFrame(index =[1,2,3,4], columns = [i for i in range(1,T+1)])
  bp_viterbi = pd.DataFrame(index =[1,2,3,4], columns = [i for i in range(1,T+1)])
  for s in range(1,5):
    viterbi.at[s,1]= pi2[s-1]*emis_2.loc[s][obs[0]]
    bp_viterbi.at[s,1] = 0
  for t in range(2,T):
    for s in range(1,5):
      max_vit = 0.0
      bp_max = {}
      for s_prev in range(1,5):
        max_vit = max((viterbi.loc[s_prev][t-1]*trans_mat2.loc[s_prev][s]*emis_2.loc[s][obs[t-1]],max_vit))
        bp_max[s_prev] = viterbi.loc[s_prev][t-1]*trans_mat2.loc[s_prev][s]
      viterbi.at[s,t] = max_vit
      argmax_vit = max( bp_max.items(), key=operator.itemgetter(1))[0]
      bp_viterbi.at[s,t]= argmax_vit
  Tmax_vit = 0.0
  Tbp_max = {}
  for s in range(1,5):
    Tmax_vit = max((viterbi.loc[s][T-1]*trans_mat2.loc[s][1]), Tmax_vit)
    Tbp_max[s] = viterbi.loc[s][T-1]*trans_mat2.loc[s][1]
  viterbi.at[1,T] = Tmax_vit  
  Targmax_vit = max(Tbp_max.items(), key=operator.itemgetter(1))[0]
  bp_viterbi.at[1,T]= Targmax_vit
  seq= []
  print(viterbi)
  for i in range(2,T+1):
    seq.append(bp_viterbi.loc[1][i])
  seq.append(1)
  return seq

#   Find Hidden states for observed sequence

ans = []
for i in obs_list:
  ans.append(viterbi_algorithm(i))
print(ans)

print(viterbi_algorithm(obs_list[4]))
