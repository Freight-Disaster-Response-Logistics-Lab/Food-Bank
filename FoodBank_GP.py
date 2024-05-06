import gurobipy as gp
import numpy as np
import pandas as pd

# Specify file path
#data_path = 'Data_Test.xlsx'

# ##PARAMETERS##

# **Sets**:
nF = 1
F = np.arange(1, nF+1) #(Food Banks) i
nP = 261
P = np.arange(1, nP+1) #(PODs - Points of Distribution) j
nZ = 98
Z = np.arange(1, nZ+1) #(Zone centroides) k

# **Data**:
data_path = 'Final_Data.xlsx'
# **Parameters**:
pZ = pd.read_excel(pd.ExcelFile(data_path), sheet_name='ZIP_CODE_POPULATION-DEMAND', header=None)
pZ_k = pZ.iloc[1:,1:2].astype(float) # Population in Zone $Z_k$ (People) pZ_k.loc[1].values[0]
#pZ_k.columns = pZ_k.columns  

dPZ_file = pd.read_excel(pd.ExcelFile(data_path), sheet_name='DISTANCE_POD_ZIP', index_col=0)
dPZ_jk = dPZ_file.astype(float) # Distance from $Z_k$ centroid to $P_j$ (Miles) dZP_kj.loc[1,1]

dFP = pd.read_excel(pd.ExcelFile(data_path), sheet_name='DISTANCE_FOOD_BANK-PODS', index_col=0)
dFP_ij = dFP.astype(float) #Distance from $F_i$ to $P_j$ (Miles)
#dFP_ij.index = dFP_ij.index-1

cP = pd.read_excel(pd.ExcelFile(data_path), sheet_name='CAPACITY_POD', index_col=0)
cP_j = cP.iloc[0:,0:1].astype(float) # POD $P_j$ capacity (Pounds)
#cP_j.columns = cP_j.columns - 3

# Number of Trucks availables
NT = nP*100
NT_i = pd.DataFrame([NT]) 
NT_i.columns = NT_i.columns + 1 
NT_i.index = NT_i.index + 1

LCM = 0.58 # Logistic Cost per Mile ($/Mile)
LCH = 71.78 # Logistic Cost per Hour ($/Hour)
AFP = 13.39/24 # Amount of Food required by Person (Pounds/Hour)

TC = 10000 # Trucks' Capacity (Pounds)
TS = 12.5 # Trucks' Speed (Miles/Hour)
WS = 3.1 # Walking Speed (Miles/Hour)
UR = 0.0008 # Unloading Rate (Hours/Pound)

PF = 0.1 # % of People requiring Food
T = 10 # Period planning time (Days)

# Food Bank $F_i$ capacity (Pounds) Big-M
CF = np.sum(AFP*PF*pZ_k, axis=0)*T*24*100
cF_i = pd.DataFrame([CF])
cF_i.columns = cF_i.columns + 1 
cF_i.index = cF_i.index + 1

# ##MODEL##
model = gp.Model('Food_Bank') 

# ##DECISION VARIABLES##:
# x_ij if $F_i$ sends food to $P_j$
x_ij = {(i, j): model.addVar(vtype = gp.GRB.BINARY, name = "x_%d_%d" %(i, j)) for i in F for j in P}
# y_kj if $Z_k$ is served by $P_j$  
y_jk = {(j, k): model.addVar(vtype = gp.GRB.BINARY, name = "y_%d_%d" %(j, k)) for j in P for k in Z}

# Amount of food (pounds) sent from $F_i$ to $P_j$
aFP_ij = {(i, j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "aFP_%d_%d" %(i, j)) for i in F for j in P} 
# Amount of food (pounds) that $P_j$ can serve to $Z_k$
aPZ_jk = {(j, k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "aPZ_%d_%d" %(j, k)) for k in Z for j in P}
# Number of trucks sent from $F_i$ to $P_j$
nT_ij = {(i, j): model.addVar(vtype = gp.GRB.INTEGER, name = "nT_%d_%d" %(i, j)) for i in F for j in P} 

# Order received time (hours) in $P_j$
ort_ij = {(i,j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "ort_%d_%d" %(i,j)) for i in F for j in P}
# Order placed time (hours) in $P_j$
opt_ij = {(i,j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "opt_%d_%d" %(i,j)) for i in F for j in P}
# Time (hours) when the amount of food in $P_j$ is zero
#tP_kj = {(k,j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "tP_%d_%d" %(k,j)) for j in P for k in Z}
# Transportation time (hours) from $F_i$ to $P_j$
tt_ij = {(i, j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "tt_%d_%d" %(i, j)) for i in F for j in P}
# Walking time (hours) from $Z_k$ to $P_j$
wt_jk = {(j, k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "wt_%d_%d" %(j, k)) for j in P for k in Z}
# Deprivation time (hours) that people in $Z_k$ wait while food is available in $P_j$. Por proporcion de la poblacion servida en ese POD.
dt_jk = {(j, k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dt_%d_%d" %(j, k)) for j in P for k in Z}
# Unloading time (hours) in $P_j$ related to the amount of food received from $F_i$
ut_ij = {(i, j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "ut_%d_%d" %(i, j)) for i in F for j in P}
# Delivery frequency of trucks sent from $F_i$ to $P_j$
fFP_ij = {(i, j): model.addVar(vtype = gp.GRB.INTEGER, name = "fFP_%d_%d" %(i, j)) for i in F for j in P}

dt_ijk = {(i,j,k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dt_%d_%d_%d" %(i,j,k)) for i in F for j in P for k in Z}

#Update model (it actully puts everything togeter)
model.update()

# ##CONSTRAINTS##:
#M = 1000000#0000000000
M_Z = AFP*PF*np.max(pZ_k)*T*24*100
M_T = T*24*100
AFP_PF_pZ_k = AFP*PF*pZ_k
dPZ_jk_WS = dPZ_jk/WS
dFP_ij_TS = dFP_ij/TS
T_hours = T*24

# **Relation between $F_i$, $P_j$, $Z_k$**:

# A Food Bank $F_i$ sends food to at least one POD $P_j$:
Fi_sendsFood_Pj = {model.addConstr(gp.quicksum(x_ij[i, j] for j in P) >= 1, name=f'F{i}_sendsFood_Pj') for i in F}
# A POD $P_j$ receives food only for one Food Bank $F_i$:
Pj_receiveFood_Fi = {model.addConstr(gp.quicksum(x_ij[i, j] for i in F) <= 1, name=f'P{j}_receiveFood_Fi') for j in P}
# A POD $P_j$ serves food only to one Zone $Z_k$:
Pj_sendsFood_Zk = {model.addConstr(gp.quicksum(y_jk[j, k] for k in Z) <= 1, name=f'P{j}_sendsFood_Zk') for j in P}
# A Zone $Z_k$ is served by at least one POD $P_j$:
Zk_receiveFood_Pj = {model.addConstr(gp.quicksum(y_jk[j, k] for j in P) >= 1, name=f'Z{k}_receiveFood_Pj') for k in Z}

#x_ij_y_kj = {model.addConstr(gp.quicksum(x_ij[i, j] for i in F) >= gp.quicksum(y_kj[k, j] for k in Z)) for j in P}

DeprivationTime = {model.addConstr(AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F) >= AFP_PF_pZ_k.loc[k].values[0]*dt_jk[j,k] + aPZ_jk[j,k]*y_jk[j,k]) for j in P for k in Z }
DeprivationTime2 = {model.addConstr(dt_jk[j,k] <= M_T*y_jk[j,k]) for j in P for k in Z }
DeprivationTime3 = {model.addConstr(dt_jk[j,k] >= gp.quicksum(ut_ij[i,j]*y_jk[j,k] for i in F) + wt_jk[j,k]) for j in P for k in Z }

# **Food capacity**:

# The amount of food sent from $F_i$ to $P_j$ is related with the capacity of the trucks:
Fi_sendsFood_Pj_TruckCapacity = {model.addConstr(aFP_ij[i, j] <= x_ij[i, j]*cP_j.loc[j].values[0]) for i in F for j in P}         
#Fi_sendsFood_Pj_TruckCapacity = {model.addConstr(aFP_ij[i, j] <= x_ij[i, j]*M) for i in F for j in P}  
# The amount of food sent from $F_i$ to $P_j$ cannot exceed the capacity of the Food Bank:
FoodBankCapacity = {model.addConstr(gp.quicksum(aFP_ij[i, j] for j in P) <= cF_i.loc[i].values[0]) for i in F}
# The amount of food sent from $F_i$ to $P_j$ cannot exceed the capacity of the POD:
#PODCapacity = {model.addConstr(gp.quicksum(aFP_ij[i, j] for i in F) <= cP_j.loc[j].values[0]) for j in P} 
# The amount of food that $Z_k$ gets from $P_j$ must be equal to the food that $F_i$ sends to $P_j$:
#FoodGet_FoodSent = {model.addConstr(gp.quicksum(aFP_ij[i, j]*nT_ij[i,j] for i in F) == gp.quicksum(aPZ_jk[j, k] for k in Z)) for j in P} #MODIFICADA
FoodGet_FoodSent = {model.addConstr(gp.quicksum(aFP_ij[i, j]*fFP_ij[i,j] for i in F) == gp.quicksum(aPZ_jk[j, k] for k in Z)) for j in P} #MODIFICADA

new1 = {model.addConstr(aPZ_jk[j, k] <= M_Z*y_jk[j,k]) for j in P for k in Z}

#FoodDemand = {model.addConstr(gp.quicksum(aPZ_jk[j, k]*nT_ij[i,j] for i in F for j in P) >= y_jk[j,k]*AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F)) for j in P for k in Z}
FoodDemand = {model.addConstr(gp.quicksum(aPZ_jk[j, k]for j in P) >= y_jk[j,k]*AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F)) for j in P for k in Z}
#FoodDemand = {model.addConstr(gp.quicksum(aPZ_jk[j, k]for j in P) >= AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F) - (1-y_jk[j,k])*M) for j in P for k in Z}

new6 = {model.addConstr(ort_ij[i,j] <= M_T*x_ij[i,j]) for i in F for j in P}

new3 = {model.addConstr(nT_ij[i,j]*TC >= aFP_ij[i,j]) for i in F for j in P}

new4 = {model.addConstr(gp.quicksum(nT_ij[i,j] for j in P) <= NT_i.loc[i].values[0]) for i in F}

#new5 = {model.addConstr(fFP_ij[i,j] <= M*x_ij[i,j]) for i in F for j in P}
#new5 = {model.addConstr(fFP_ij[i,j] <= M_T*nT_ij[i,j]) for i in F for j in P}


WalkingTime = {model.addConstr(wt_jk[j,k] == dPZ_jk_WS.loc[j,k]*y_jk[j,k]) for j in P for k in Z } 

UnloadingTime = {model.addConstr(ut_ij[i, j] == UR*aFP_ij[i, j]) for i in F for j in P}

TransportationTime = {model.addConstr(tt_ij[i, j] == dFP_ij_TS.loc[i,j]*x_ij[i, j]) for i in F for j in P}

OrderPlacedTime = {model.addConstr(opt_ij[i,j] + tt_ij[i,j] <= ort_ij[i,j]) for i in F for j in P}

PlaningTime = {model.addConstr(ort_ij[i,j]*fFP_ij[i,j] == T_hours*x_ij[i,j]) for i in F for j in P} 

OrderPlacedTime2 = {model.addConstr(gp.quicksum(opt_ij[i,j] for i in F) >= gp.quicksum(wt_jk[j,k] for k in Z) + gp.quicksum(ut_ij[i,j] for i in F)) for j in P}

new7 = {model.addConstr(opt_ij[i,j] <= M_T*x_ij[i,j]) for i in F for j in P}

dt_ijk_total = {model.addConstr(dt_ijk[i,j,k] == dt_jk[j,k] * fFP_ij[i,j]) for i in F for j in P for k in Z}

# ##OBJECTIVE FUNCTION##:
ER = 1/2000 # Exchange Rate USD to COP Base on the paper Discrete choice Cantillo et al. (2018)
dFP_ij_LCM_LCH_TS = 2*dFP_ij*(LCM+(LCH/TS))
UR_LCH = UR*LCH

#model.setObjective(gp.quicksum((3066*dt_kj[k,j] + 349.66*(dt_kj[k,j])**2)/ER for j in P for k in Z) + gp.quicksum((aFP_ij[i,j]/TC)*2*dFP_ij.loc[j,i]*LCM_LCH_TS + UR_LCH*aFP_ij[i,j] for i in F for j in P), 
#model.setObjective(gp.quicksum(ER*(3066*dt_jk[j,k] + 349.66*(dt_jk[j,k])**2) for j in P for k in Z) + gp.quicksum(nT_ij[i,j]*fFP_ij[i,j]*dFP_ij_LCM_LCH_TS.loc[i,j] + UR_LCH*aFP_ij[i,j] for i in F for j in P), 
#model.setObjective(gp.quicksum(ER*(3066*dt_jk[j,k] + 349.66*(dt_jk[j,k])**2) for j in P for k in Z) + gp.quicksum(nT_ij[i,j]*dFP_ij_LCM_LCH_TS.loc[i,j] + UR_LCH*aFP_ij[i,j] for i in F for j in P), 
model.setObjective(gp.quicksum(dt_ijk[i,j,k]*ER*(3066 + 349.66*dt_jk[j,k]) for i in F for j in P for k in Z) + gp.quicksum(fFP_ij[i,j]*(nT_ij[i,j]*dFP_ij_LCM_LCH_TS.loc[i,j] + UR_LCH*aFP_ij[i,j]) for i in F for j in P), 
                   gp.GRB.MINIMIZE)
model.setParam('OutputFlag', 0)
model.update()
#Optimize (solve the model)
model.optimize()
#model.computeIIS()
#model.write("model.ilp")
model.write('FoodBank_2.lp')
model.write("FoodBank_2.sol")

#print('Obj: %g' % model.objVal)
