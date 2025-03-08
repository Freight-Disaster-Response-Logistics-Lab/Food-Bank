import gurobipy as gp
import numpy as np
import pandas as pd

# ##PARAMETERS##
import numpy as np
P_small_near = np.array([44,198,202,47,197])#199
Z_small_near = np.array([28,19])
P_small_far = np.array([25,140,110,196,207])
Z_small_far = np.array([93,80])
P_medium_near = np.array([
    44,
    198,
    202,
    47,
    197,
    199,
    29,
    97,
    15,
    43,
    123,
    42,
    205,
    122,
    133,
    220,
    101,
    86,
    17,
    130,
    208,
    214,
    146])
Z_medium_near = np.array([28, 19, 10, 27, 11, 2, 12 ,25, 9, 76])
P_medium_far = np.array([
    172,
    62,
    206,
    210,
    33,
    153,
    102,
    151,
    4,
    162,
    53,
    5,
    6,
    222,
    215,
    165,
    30,
    155,
    207,
    196,
    110,
    140,
    25])
Z_medium_far = np.array([63, 67, 68, 80, 93])

# **Sets**:
nF = 1
F = np.arange(1, nF+1) #(Food Banks) i
nP = 228
P = np.arange(1, nP+1) #(PODs - Points of Distribution) j
nZ = 96
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

LCM = 0.969 #0.58 # Logistic Cost per Mile ($/Mile)
LCH = 71.78 # Logistic Cost per Hour ($/Hour)
AFP = 13.39/24 # Amount of Food required by Person (Pounds/Hour)

TC = 10000 # Trucks' Capacity (Pounds)
TS = 30 #12.5 # Trucks' Speed (Miles/Hour)
WS = 12.5 #3.1 # Walking Speed (Miles/Hour)
UR = 0.00005 # Unloading Rate (Hours/Pound)
SR = 0.0003 # Setting-up Rate (Hours/Pound)
LR = 0.00005 # Loading Rate (Hours/Pound)

PF = 0.005#0.1 # % of People requiring Food
T = 10 # Period planning time (Days)

# Food Bank $F_i$ capacity (Pounds) Big-M
CF = np.sum(AFP*PF*pZ_k, axis=0)*T*24*100
cF_i = pd.DataFrame([CF])
cF_i.columns = cF_i.columns + 1 
cF_i.index = cF_i.index + 1

# Walking time (hours) from $Z_k$ to $P_j$
wt_jk = dPZ_jk/WS
# Transportation time (hours) from $F_i$ to $P_j$
tt_ij = dFP_ij/TS

M_Z = AFP*PF*np.max(pZ_k)*T*24*100
M_T = T*24*100
AFP_PF_pZ_k = AFP*PF*pZ_k
T_hours = T*24

options = {
    "WLSACCESSID": "058e9d62-b9f9-418d-86e4-a7145e6660a3",
    "WLSSECRET": "58a30013-39fb-4857-ad04-c11ba39a6f98",
    "LICENSEID": 2516598,
}
with gp.Env(params=options) as env, gp.Model(env=env) as model:

    # ##MODEL##
    #model = gp.Model('Food_Bank') 

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
    # Deprivation time (hours) that people in $Z_k$ wait while food is available in $P_j$. Por proporcion de la poblacion servida en ese POD.
    dt_jk = {(j, k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dt_%d_%d" %(j, k)) for j in P for k in Z}
    # Unloading time (hours) in $P_j$ related to the amount of food received from $F_i$
    ut_ij = {(i, j): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "ut_%d_%d" %(i, j)) for i in F for j in P}
    # Delivery frequency of trucks sent from $F_i$ to $P_j$
    fFP_ij = {(i, j): model.addVar(vtype = gp.GRB.INTEGER, name = "fFP_%d_%d" %(i, j)) for i in F for j in P}

    #dt_ijk = {(i,j,k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dt_%d_%d_%d" %(i,j,k)) for i in F for j in P for k in Z}
    dt_k = {(k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dt_%d" %(k)) for k in Z}
    dtt_k = {(k): model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dtt_%d" %(k)) for k in Z}

    dep_cost = model.addVar(vtype = gp.GRB.CONTINUOUS, name = "dep_cost")
    log_cost = model.addVar(vtype = gp.GRB.CONTINUOUS, name = "log_cost")

    #Update model (it actully puts everything togeter)
    model.update()

    # ##CONSTRAINTS##:

    Fi_sendsFood_Pj = {model.addConstr(gp.quicksum(x_ij[i, j] for j in P) >= 1, name=f'F{i}_sendsFood_Pj') for i in F}
    Pj_receiveFood_Fi = {model.addConstr(gp.quicksum(x_ij[i, j] for i in F) <= 1, name=f'P{j}_receiveFood_Fi') for j in P}
    Pj_sendsFood_Zk = {model.addConstr(gp.quicksum(y_jk[j, k] for k in Z) <= 1, name=f'P{j}_sendsFood_Zk') for j in P}
    Zk_receiveFood_Pj = {model.addConstr(gp.quicksum(y_jk[j, k] for j in P) >= 1, name=f'Z{k}_receiveFood_Pj') for k in Z}

    Fi_sendsFood_Pj_TruckCapacity = {model.addConstr(aFP_ij[i, j] <= x_ij[i, j]*cP_j.loc[j].values[0]) for i in F for j in P}
    FoodBankCapacity = {model.addConstr(gp.quicksum(aFP_ij[i, j] for j in P) <= cF_i.loc[i].values[0]) for i in F}
    Truck_available = {model.addConstr(gp.quicksum(nT_ij[i,j] for j in P) <= NT_i.loc[i].values[0]) for i in F}

    FoodGet_FoodSent = {model.addConstr(gp.quicksum(aFP_ij[i, j] for i in F) == gp.quicksum(aPZ_jk[j, k] for k in Z)) for j in P} 
    Food_sent_capacity = {model.addConstr(nT_ij[i,j]*TC >= aFP_ij[i,j]) for i in F for j in P}

    FoodDemand = {model.addConstr(gp.quicksum(aPZ_jk[j, k] for j in P) >= y_jk[j,k]*AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F)) for j in P for k in Z}
    #FoodDemand = {model.addConstr(gp.quicksum(aPZ_jk[j, k]*fFP_ij[i,j] for i in F for j in P) >= y_jk[j,k]*AFP_PF_pZ_k.loc[k].values[0]*T_hours) for j in P for k in Z}

    UnloadingTime = {model.addConstr(ut_ij[i, j] == (UR+SR)*aFP_ij[i, j]) for i in F for j in P}

    DeprivationTime = {model.addConstr(AFP_PF_pZ_k.loc[k].values[0]*gp.quicksum(ort_ij[i,j] for i in F)*y_jk[j,k] == AFP_PF_pZ_k.loc[k].values[0]*dt_jk[j,k]+ aPZ_jk[j,k]) for j in P for k in Z }
    DeprivationTime3 = {model.addConstr(dt_jk[j,k] >= gp.quicksum(ut_ij[i,j]*y_jk[j,k] for i in F) + wt_jk.loc[j,k]*y_jk[j,k]) for j in P for k in Z }

    OrderPlacedTime = {model.addConstr(opt_ij[i,j] + 2*tt_ij.loc[i,j]*x_ij[i,j] + (1/7)*ut_ij[i,j] == ort_ij[i,j]) for i in F for j in P}
    PlaningTime = {model.addConstr(ort_ij[i,j]*fFP_ij[i,j] >= T_hours*x_ij[i,j]) for i in F for j in P} 
    
    DeprivationTime2 = {model.addConstr(dt_jk[j,k] <= M_T*y_jk[j,k]) for j in P for k in Z }
    new1 = {model.addConstr(aPZ_jk[j, k] <= M_Z*y_jk[j,k]) for j in P for k in Z}
    new6 = {model.addConstr(ort_ij[i,j] <= M_T*x_ij[i,j]) for i in F for j in P}
    new5 = {model.addConstr(fFP_ij[i,j] <= M_T*nT_ij[i,j]) for i in F for j in P}
    new7 = {model.addConstr(opt_ij[i,j] <= M_T*x_ij[i,j]) for i in F for j in P}
    new8 = {model.addConstr(nT_ij[i,j] <= x_ij[i,j]*NT_i.loc[i].values[0]) for i in F for j in P}

    #dt_k_total = {model.addConstr(dt_k[k] >= dt_jk[j,k]) for j in P for k in Z}
    #dt_ijk_total = {model.addConstr(dt_ijk[i,j,k] >= dt_k[k] * fFP_ij[i,j]) for i in F for j in P for k in Z}
    
    dtt_k_total = {model.addConstr(dtt_k[k] >= dt_jk[j,k]*fFP_ij[i,j]) for i in F for j in P for k in Z}
    #dt_k_total = {model.addConstr(dtt_k[k] >= dt_k[k]*fFP_ij[i,j]) for i in F for j in P for k in Z}
    dt_k_total = {model.addConstr(dt_k[k] >= dt_jk[j,k]) for j in P for k in Z}

    # ##OBJECTIVE FUNCTION##:
    ER = 1/2000 # Exchange Rate USD to COP Base on the paper Discrete choice Cantillo et al. (2018)
    dFP_ij_LCM_LCH_TS = 2*dFP_ij*(LCM+(LCH/TS))
    UR_LCH = (UR+LR+SR)*LCH #(7/6*UR)*LCH

    #dep_cost_const = model.addConstr(dep_cost == gp.quicksum(pZ_k.loc[k].values[0]*PF*dtt_k[k]*ER*(3066 + 349.66*dt_k[k]) for k in Z))
    dep_cost_const = model.addConstr(dep_cost == gp.quicksum(pZ_k.loc[k].values[0]*PF*dtt_k[k]*(0.05) for k in Z))
    log_cost_const = model.addConstr(log_cost == gp.quicksum(fFP_ij[i,j]*(nT_ij[i,j]*dFP_ij_LCM_LCH_TS.loc[i,j] + UR_LCH*aFP_ij[i,j]) for i in F for j in P))
    #obj = dep_cost + log_cost
    
    model.setObjective(dep_cost + log_cost, 
                       gp.GRB.MINIMIZE)
    model.setParam('Presolve', 2) #to reduce the model size and tighten it further, if possible
    model.setParam('NodefileStart', 0.5) #to avoid exhausts memory
    model.setParam('Threads', 30) #to avoid exhausts memory
    model.setParam('PreSparsify', 1) #This reduction can sometimes significantly reduce the number of non-zero values in the presolved model
    model.setParam('MIPFocus', 1)#to focus on finding a feasible solution
    model.setParam('Cuts', 2)#Use value 0 to shut off cuts, 1 for moderate cut generation, 2 for aggressive cut generation, and 3 for very aggressive cut generation. The default -1 value chooses automatically.
    #model.update()
    #model.update()
    #Optimize (solve the model)
    model.optimize()
    #model.computeIIS()
    #model.write("model.ilp")
    #model.write('FoodBank_Final.lp')
    model.write('FoodBank_Final_Lineal.sol')


FBLocation_file = pd.read_excel(pd.ExcelFile(data_path), sheet_name='FB-LOCATION', index_col=0)
FBLocation_i = FBLocation_file.iloc[:,0:4]
#FBLocation_i.index = FBLocation_i.index + 1

PODLocation_file = pd.read_excel(pd.ExcelFile(data_path), sheet_name='POD-LOCATION', index_col=0)
PODLocation_j = PODLocation_file.iloc[:,0:4]
#PODLocation_j.index = PODLocation_j.index + 1

ZoneLocation_file = pd.read_excel(pd.ExcelFile(data_path), sheet_name='ZIP_CODE-LOCATION', index_col=0)
ZoneLocation_k = ZoneLocation_file.iloc[:,0:4]
#ZoneLocation_k.index = ZoneLocation_k.index + 1

FP_solution = []
FP_variables = []
PZ_solution = []
PZ_variables = []
FP_noSol = []
FP_varNoSol = []
for i in F:
    for j in P:
        if x_ij[i,j].X > 0:
            FB_lat = FBLocation_i.loc[i]['Latitude']
            FB_long = FBLocation_i.loc[i]['Longitude']
            POD_lat = PODLocation_j.loc[j]['Latitude']
            POD_long = PODLocation_j.loc[j]['Longitude']
            FP_variables.append([i,j])
            FP_solution.append(((FB_long, FB_lat),(POD_long, POD_lat)))
        else:
            FB_lat = FBLocation_i.loc[i]['Latitude']
            FB_long = FBLocation_i.loc[i]['Longitude']
            POD_lat = PODLocation_j.loc[j]['Latitude']
            POD_long = PODLocation_j.loc[j]['Longitude']
            FP_varNoSol.append([i,j])
            FP_noSol.append(((FB_long, FB_lat),(POD_long, POD_lat)))

for j in P:
    for k in Z:
        if y_jk[j,k].X > 0:
            POD_lat = PODLocation_j.loc[j]['Latitude']
            POD_long = PODLocation_j.loc[j]['Longitude']
            Zone_lat = ZoneLocation_k.loc[k]['Latitude']
            Zone_long = ZoneLocation_k.loc[k]['Longitude']
            PZ_variables.append([j,k])
            PZ_solution.append(((POD_long, POD_lat),(Zone_long, Zone_lat)))

import openrouteservice
from openrouteservice import convert
import folium
from folium.plugins import BeautifyIcon

client = openrouteservice.Client(key='5b3ce3597851110001cf62487131ee07189f41618740241c328082ee')

m = folium.Map(location=[FBLocation_i.loc[1]['Latitude'],FBLocation_i.loc[1]['Longitude']], zoom_start=10, control_scale=True, tiles="cartodbpositron")

nFP_sol = 0
for coords in FP_solution:
    res = client.directions(coords)
    geometry = client.directions(coords)['routes'][0]['geometry']
    decoded = convert.decode_polyline(geometry)
    #coords = ((80.21787585263182,6.025423265401452),(80.23990263756545,6.018498276842677))
    
    distance_miles = round(res['routes'][0]['summary']['distance'] / 1609.34, 1)  # Originally this is in meters
    distance_km = round(res['routes'][0]['summary']['distance'] / 1000, 1)  # Originally this is in meters
    
    duration_hours = round(res['routes'][0]['summary']['duration'] / 3600, 1)  # Originally this is in seconds
    duration_mins = round(res['routes'][0]['summary']['duration'] / 60, 1)  # Originally this is in seconds

    # Update HTML text with miles and hours
    distance_txt = "<h4><b>Distance:&nbsp;<strong>" + str(distance_miles) + " miles</strong></h4></b>"
    duration_txt = "<h4><b>Duration:&nbsp;<strong>" + str(duration_hours) + " hours</strong></h4></b>"

    #distance_txt = "<h4> <b>Distance :&nbsp" + "<strong>"+str(distance_km)+" Km </strong>" +"</h4></b>"
    #duration_txt = "<h4> <b>Duration :&nbsp" + "<strong>"+str(duration_mins)+" minutes </strong>" +"</h4></b>"

    
    folium.GeoJson(decoded).add_child(folium.Popup(distance_txt+duration_txt, max_width = 300)).add_to(m)

    folium.Marker(
        location = list(coords[0][::-1]),
        popup = f"Food Bank {FP_variables[nFP_sol][0]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(FP_variables[nFP_sol][0]),
            spin = True,
            text_color = 'green',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)

    folium.Marker(
        location = list(coords[1][::-1]),
        popup = f"POD {FP_variables[nFP_sol][1]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(FP_variables[nFP_sol][1]),
            spin = True,
            text_color = 'red',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)

    nFP_sol += 1

nPZ_sol = 0
for coords in PZ_solution:
    res = client.directions(coords)
    geometry = client.directions(coords)['routes'][0]['geometry']
    decoded = convert.decode_polyline(geometry)
    #coords = ((80.21787585263182,6.025423265401452),(80.23990263756545,6.018498276842677))
    
    distance_miles = round(res['routes'][0]['summary']['distance'] / 1609.34, 1)  # Originally this is in meters
    distance_km = round(res['routes'][0]['summary']['distance'] / 1000, 1)  # Originally this is in meters
    
    duration_hours = round(res['routes'][0]['summary']['duration'] / 3600, 1)  # Originally this is in seconds
    duration_mins = round(res['routes'][0]['summary']['duration'] / 60, 1)  # Originally this is in seconds

    # Update HTML text with miles and hours
    distance_txt = "<h4><b>Distance:&nbsp;<strong>" + str(distance_miles) + " miles</strong></h4></b>"
    duration_txt = "<h4><b>Duration:&nbsp;<strong>" + str(duration_hours) + " hours</strong></h4></b>"

    #distance_txt = "<h4> <b>Distance :&nbsp" + "<strong>"+str(distance_km)+" Km </strong>" +"</h4></b>"
    #duration_txt = "<h4> <b>Duration :&nbsp" + "<strong>"+str(duration_mins)+" minutes </strong>" +"</h4></b>"

    
    folium.GeoJson(decoded).add_child(folium.Popup(distance_txt+duration_txt, max_width = 300)).add_to(m)

    '''folium.Marker(
        location = list(coords[0][::-1]),
        popup = f"POD {PZ_variables[nPZ_sol][0]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(PZ_variables[nPZ_sol][0]),
            spin = True,
            text_color = 'red',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)'''

    folium.Marker(
        location = list(coords[1][::-1]),
        popup = f"Zone {PZ_variables[nPZ_sol][1]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(PZ_variables[nPZ_sol][1]),
            spin = True,
            text_color = 'blue',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)

    nPZ_sol += 1

nFP_noSol = 0
for coords in FP_noSol:
    folium.Marker(
        location = list(coords[0][::-1]),
        popup = f"Food Bank {FP_varNoSol[nFP_noSol][0]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(FP_varNoSol[nFP_noSol][0]),
            spin = True,
            text_color = 'green',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)

    folium.Marker(
        location = list(coords[1][::-1]),
        popup = f"POD {FP_varNoSol[nFP_noSol][1]}",
        icon = BeautifyIcon(
            icon_shape = 'marker',
            number = int(FP_varNoSol[nFP_noSol][1]),
            spin = True,
            text_color = 'red',
            background_color = "#FFF",
            inner_icon_style = "font-size:12px;padding-top:-5px;"
        ),
    ).add_to(m)

    nFP_noSol += 1

m.save('map_small5.html')