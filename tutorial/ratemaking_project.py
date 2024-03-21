
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns",None)

# lets import streamlit
import streamlit as st
import plotly.graph_objs as go

# create different pages

# home page configs 
st.set_page_config(layout="wide")
st.title("Worker's Compensation")
st.subheader("Pricing(Ratemaking) Worker's Compensation Premiums using Actuarial Techniques")

# our csv file
filepath = "./wkcomp_pos.csv"

# load the dataset
@st.cache_data # for faster execution
def load_data():
    df = pd.DataFrame(pd.read_csv(filepath))
    return df
'''##### - This is our dataset'''
dataset = load_data()
st.dataframe(dataset) # the worker's compensation dataset

# dataset columns
columns = dataset.columns
st.write("Features in dataset:",columns)

# correlation heatmap
st.subheader("Correlation heatmap")
df_corr = dataset.drop(columns=['GRCODE','GRNAME'])
fig, ax = plt.subplots()
sns.heatmap(df_corr.corr(), ax=ax, annot=True, linewidths=0.36, linecolor="black", fmt=".2f")
st.write(fig)

"""We see that there's a strong positive correlation between the following features:
- PostedReserve97_D with IncurLoss_D, CumPaidLoss_D, EarnedPremDIR_D and EarnedPremNet_D  
- EarnedPremNet_D with CumPaidLoss_D and IncurLoss_D
"""

# display companies in the dataset
col1, col2, col3= st.columns(3)
dataset["GRCODE-GRNAME"] = dataset['GRCODE'].astype('str')+"-"+dataset['GRNAME']
companies = dataset["GRCODE-GRNAME"]
companies = pd.DataFrame({'Companies':pd.unique(companies)})
with col1:
    st.subheader("Companies in the dataset:")
    st.dataframe(companies, hide_index=True, width = 300)

# sample of 5 companies chosen for this project
sample_companies = ["Allstate Ins Co Grp", "California Cas Grp", "Celina Mut Grp", "Federal Ins Co Grp", "Farm Bureau of MI Grp"]
grcodes = [86, 337, 353, 388, 671]

df_comp = pd.DataFrame(
    {
        'GRCODE':grcodes,
        'NAME':sample_companies,
    }
)
with col2:
    st.subheader("For this project, we use a sample of 5 companies:")
    st.dataframe(df_comp, hide_index=True,)

# select a grcode
slt_comp = st.selectbox("Select a company by GRCODE:", grcodes, index=0, placeholder="Choose an option")


"""# Let's see some Triangles"""

# python class that consists of 4 different averaging methods for averaging loss-development factors
class AveragingMethods:
    def __init__(self, data):
        '''Here, Data is of type list'''
        self.data = data
    def SimpleAvg(self): # simple average
        sum = 0
        for i in self.data:
            sum += i
        return round( sum/len(self.data), 4)
    def VolumeAvg(self, dt1, dt2): # volume-weighted average
        sum1 = 0
        for i in dt1:
            sum1 += i
        sum2 = 0
        for j in dt2:
            sum2 += j
        return round( sum1/sum2,4)
    def MedialAvg(self): # medial average
        minimum = min(self.data)
        maximum = max(self.data)
        sum = 0
        if len(self.data) > 2:
            for i in self.data:
                sum += i
            sum -= (maximum+minimum)
            return round( sum/(len(self.data)-2),4)
        else:
            return  round( (maximum+minimum)/2,4)
    def GeometricAvg(self): # geometric average
        sum = 1
        for i in self.data:
            sum *= i
        return round( sum**(1/len(self.data)),4)


def LossData(grcode):
    '''This function extracts the loss data of a specific company corresponding to its GRCODE
        Here data is of type: dataframe'''
    company = dataset[dataset["GRCODE"]==grcode]
    return(company)


def createLossTriangle(data):
    '''This function extracts and creates Loss triangles
        Here data is of type: dataframe'''
    trframe = {}      # dict containing loss triangle values for various accident years
    for i in range(1988,1998):
        L = []
        for j in range(i,1998):
            condition = ( (data['AccidentYear']==i) & (data['DevelopmentYear']==j) )
            L.append(int(data.loc[condition]['CumPaidLoss_D']))
        i = int(i)
        trframe[i] = L
    return trframe


def displayTriangleData(data):
    '''This function displays Loss Triangle data
       Here data is of type: dictionary'''
    for i in data.keys():
        print(i, end = "\t\t")
        for j in data[i]:
            print(j, end = "\t")
        print("\n")


def computeLDF(data):
    '''This function computes Loss Development Factors
       Here data is of type: dictionary'''
    trframe = {}
    for i in data.keys():
        L = []
        for j in range(len(data[i])-1):
            ldf = data[i][j+1]/data[i][j]
            L.append( round(ldf,4) )
        i = int(i)
        trframe[i] = L
    return trframe


def computeAverageLDF(ldf_info, loss_info):
    '''This function computes various Averages of Loss Development Factors
       Here data is of type: dictionary'''
    # print("Available averaging methods:\n\
    #         1. Simple Average :- Latest 5\n\
    #         2. Volume-Weighted Average :- Latest 5\n\
    #         3. Medial Average :- Latest 5\n\
    #         4. Geometric Average :- Latest 5\n")
    DK = list(ldf_info.keys())
    DK = sorted(DK, reverse=True)
    trframe = {
        'SimpleAvg':[],
        'VolumeAvg':[],
        'MedialAvg':[],
        'GeometricAvg':[]
    }
    # for Medial, Simple and Geometric Averages
    for i in range(0,10):
        L = []
        c = 1
        for j in DK:
            try:    # to avoid Index Out of Bounds
                if ldf_info[j][i] and c<=5:
                    L.append(ldf_info[j][i])
                    c+=1
            except:
                pass
        if(L!=[]):
            obj = AveragingMethods(L)               # object of class Averaging methods
            simp_avg = obj.SimpleAvg()
            med_avg  = obj.MedialAvg()
            geo_avg  = obj.GeometricAvg()
            trframe['SimpleAvg'].append(simp_avg)
            trframe['MedialAvg'].append(med_avg)
            trframe['GeometricAvg'].append(geo_avg)
    # only for Volume-Weighted Average
    for i in range(1,10):
        L1 = []
        L2 = []
        c = 1
        for j in DK:
            try:
                if loss_info[j][i] and loss_info[j][i-1] and c<=5:
                    L1.append(loss_info[j][i])
                    L2.append(loss_info[j][i-1])
                    c+=1
            except:
                pass
        if(L1!=[] and L2!=[]):
            obj = AveragingMethods(L1)
            vol_avg = obj.VolumeAvg(L1, L2)
            trframe['VolumeAvg'].append(vol_avg)
    return trframe

# the loss data of selected company
loss_data = LossData(slt_comp)
st.write("Loss data of selected company:",loss_data)

# the loss development triangle
st.write("Loss Development Triangle")
loss_triangle = createLossTriangle(loss_data)
# Determine the maximum length of the arrays
max_length = max(len(arr) for arr in loss_triangle.values())
# Pad the arrays with NaN to make them all the same length
data_padded = {key: arr + [np.nan] * (max_length - len(arr)) for key, arr in loss_triangle.items()}
# Create DataFrame from the padded dictionary
df = pd.DataFrame.from_dict(data_padded)
df = df.T
for i in range(0,max_length):
    df.rename(columns={i:(i+1)*12,}, inplace=True)
st.dataframe(df,width=1000)

# triangle of LDFs
st.write("Loss Development Factors")
ldf_triangle = computeLDF(loss_triangle)
# Determine the maximum length of the arrays
max_length = max(len(arr) for arr in ldf_triangle.values())
# Pad the arrays with NaN to make them all the same length
data_padded = {key: arr + [np.nan] * (max_length - len(arr)) for key, arr in ldf_triangle.items()}
# Create DataFrame from the padded dictionary
df = pd.DataFrame.from_dict(data_padded)
df = df.T
for i in range(0,max_length):
    df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
st.dataframe(df,width=1000)

# averages of ldf
avg_ldf = computeAverageLDF(ldf_triangle, loss_triangle)

col3,col4 = st.columns(2, gap="large")

# Plot line chart
fig = go.Figure()

# Add lines for each list
for key, values in avg_ldf.items():
    fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode='lines', name=key))

# Customize layout
fig.update_layout(title='Trend in Averages',
                  xaxis_title='Data Points', yaxis_title='Values')

# Display the chart
col4.plotly_chart(fig)
avg_ldf_df = pd.DataFrame(
    {
    'Simple Average':avg_ldf['SimpleAvg'],
    'Medial Average':avg_ldf['MedialAvg'],
    'Volume-Weighted':avg_ldf['VolumeAvg'],
    'Geometric Average':avg_ldf['GeometricAvg'],
    })
avg_ldf_df = avg_ldf_df.T
for i in range(0,max_length):
    avg_ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
col3.subheader("Averages of LDFs")
col3.dataframe(avg_ldf_df)

# Select LDF. Here we take the Volume-Weighted Averages.
ldf_choices = list(avg_ldf.keys())
chosen_Ldf = st.selectbox("Select an averaging method for the LDFs:", ldf_choices, index=0, placeholder="Choose an option")
selected_Ldf = avg_ldf[chosen_Ldf]

# We select an arbitrary tail factor
tail = 1.0000
selected_Ldf.append(tail)
selected_Ldf = selected_Ldf[::-1] # revert the list for finding CDFs

selected_Ldf_df = pd.DataFrame({chosen_Ldf:selected_Ldf[::-1],})
selected_Ldf_df = selected_Ldf_df.T
for i in range(0,max_length+1):
    if i==max_length:
        selected_Ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,'ult'),}, inplace=True)
    else:
        selected_Ldf_df.rename(columns={i:"{}-{}".format((i+1)*12,(i+2)*12),}, inplace=True)
st.subheader("Selected LDFs")
st.dataframe(selected_Ldf_df)

# Cumulative Loss Development factors
cdf = []
for i in range(1, len(selected_Ldf)+1):
    f = 1
    for j in range(0, i):
        f*=selected_Ldf[j]
    cdf.append( round( f,4) )
cdf_df = pd.DataFrame({'CDF':cdf,})
cdf_df = cdf_df.T
for i in range(0,max_length+1):
        cdf_df.rename(columns={i:"{}-{}".format((i+1)*12,'ult'),}, inplace=True)
st.subheader("Cumulative Development Factors")
st.dataframe(cdf_df)

# Projected Ultimate Losses
proj_ultLosses = {}
for i in range(0, len(cdf)):
    for j in range(0, len(loss_triangle)):
        if(i==j):
            proj_ultLosses[ list(loss_triangle.keys())[j] ] = round( list(loss_triangle.values())[i][-1]*cdf[i],4)
st.subheader("Projected Ultimate Losses")
st.dataframe(proj_ultLosses, width=300)
            




"""## Lets evaluate the closeness of our projected ultimate losses to the actual ultimate losses."""

# metrics used: mean absolute error, and r^2 coefficient
from sklearn.metrics import mean_absolute_error as mae, r2_score as r2
# Actual Ultimate Losses
act_ultLosses = {}
for i in range(1988,1998):
        condition = ( (loss_data['AccidentYear']==i) & (loss_data['DevelopmentLag']==10) )
        act_ultLosses[i] = int( loss_data.loc[condition]['CumPaidLoss_D'])
st.subheader("Actual Ultimate Losses")

st.dataframe(act_ultLosses, width=300)

# print( "\nMean Absolute Error =",mae(list(act_ultLosses.values()), list(proj_ultLosses.values())) )
# print("\nR^2 coefficient =", r2(list(act_ultLosses.values()), list(proj_ultLosses.values()) ) )

"""The R^2 coefficient is close to 1, which is very good. This means that Chain-Ladder Method is performing sufficiently well.

# Semi-Stochastic Method for Loss Development
# """

# # loss_triangle
# # ldf_triangle

# # Expectations
# def Avg(ldf):
#     try:
#         return sum(ldf)/(len(ldf)-1)
#     except:
#         return 0

# def ldf_colwise(ldf_tri):
#     stoc_ldf = {}
#     i = 1
#     for k in range(0,len(ldf_tri.keys())):
#         L = []
#         for j in ldf_tri.keys():
#             try:
#                 L.append(ldf_tri[j][k])
#             except:
#                 pass
#         L.append(0)
#         stoc_ldf[i] = L
#         i+=1
#     return stoc_ldf

# def change_keys(data):
#     k = {}
#     for i in data.keys():
#         k[i-1987] = data[i]
#     return k

# def calcUlt(avg, tridata):
#     ult = {}
#     n = len(tridata.keys())
#     i = 2
#     j = len(avg.keys())-1
#     mult = 1
#     ult[1988] = tridata[1][9]
#     while(i!=n+1 and j!=-1):
#             print(tridata[i][j-1])

#             mult*=avg[j]
#             print(mult)
#             mm = tridata[i][j-1]*mult
#             ult[1988+i-1] = round(mm,3)
#             i+=1
#             j-=1
#     return ult

# #displayTriangleData(ldf_triangle)
# stoc = ldf_colwise(ldf_triangle)
# print(stoc)

# avg_ldf = {}
# for i in stoc.keys():
#     avg_ldf[i] = Avg(stoc[i])
# print(avg_ldf)

# triloss = change_keys(loss_triangle)
# #displayTriangleData(triloss)
# ultclaims = calcUlt(avg_ldf,triloss)
# print(ultclaims)





"""# Calculating Rate and Benefit Adjustment Factors
We use a simple general formula derived by Richard A. Bill for automating the calculation of rate and benefit adjustment factors. This is based on the parallelogram method. More details on the formula can be found in the paper by Richard A. Bill on:
https://www.casact.org/abstract/generalized-earned-premium-rate-adjustment-factors


We exclude any fluctuations arising due to legal changes.





## Testing the formula with a sample case
"""

import datetime
T = 1; E = 1;
start_date = datetime.date(2015,1,1)
rate_date = datetime.date(2015,4,1)

def months_between(date1,date2):

    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2

    return months/12

D = months_between(rate_date, start_date)

A = D+T
B = max( A-E, 0 )
C = max( D, 0 )

P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
P

rate_changes = {
            datetime.date(2015,4,1):0.05, datetime.date(2016,1,1):0.1,
                datetime.date(2017,7,1):-0.02,
                }
earned_prem = {2015: 20400, 2016: 21000, 2017: 22800, 2018: 23200}

# first calculate the rate change indeces
rates = list(rate_changes.values())
rate_index =[1.00]+[ (1+i) for i in rates ] # including initial index without changes = 1.00 (rate change = 0%)
#print("Rate change indeces:\n",rate_index)
cum_index = []
f = 1
for i in rate_index:
    f *= i
    cum_index.append( round(f, 4))
#print("Cumulative rate change indeces:\n",cum_index)
current_cum_rate_index = cum_index[-1]

import datetime
T = 1; E = 1;

def months_between(date1,date2):

    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2

    return months/12



def find_remains(L):
    if L!=[]:
        L.append(0)
        to_return = []
        max = 1
        for i in range(0, len(L)):
            if L[i]!=0:
                diff = max - L[i]
                to_return.append(round( diff,5))
                max = L[i]
                if L[i+1]==0:
                    to_return.append(round( max,5))
            else:
                to_return.append(0)

        to_return.pop()
        return to_return




def earnedPortion(rate_dates, earned_prem_years):

    portion = {}
    for i in earned_prem_years:
        portion[i] = []
    for i in earned_prem_years:
        start_date = datetime.date(i,1,1)
        for j in rate_dates:

            if months_between(j, start_date)<1 and months_between(j, start_date)>-1:

                D = months_between(j, start_date)

                A = D+T
                B = max( A-E, 0 )
                C = max( D, 0 )

                P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
                portion[i].append( round(P, 5))
            else:
                portion[i].append(0)


    for i in portion.keys():
        portion[i] = find_remains(portion[i])
    return(portion)


rate_effec_dates = list( rate_changes.keys())
years_toAdjust = list( earned_prem.keys() )
earned_PremPortion = earnedPortion(rate_effec_dates, years_toAdjust)
#print("The portion earned by the premium in the years w.r.t. the rate changes are:\n",earned_PremPortion)
# Average Cumulative Rate Level Indices

def AvgCumulIndices(L, cumul_indices):
    prod = L*cumul_indices
    sum = 0
    for i in prod:
        sum+=i
    return round(sum, 5)

for i in earned_PremPortion.keys():
    earned_PremPortion[i] = np.array(earned_PremPortion[i])
cum_index = np.array(cum_index)

avg_CumulIndices = {}
for i in earned_PremPortion.keys():
    avg_CumulIndices[i] = AvgCumulIndices(earned_PremPortion[i], cum_index)

#print("The average cumulative rate level indices are:\n",avg_CumulIndices)

# Finally, On-Level Factors
onlevel = {}
for i in avg_CumulIndices.keys():
    onlevel[i] = round( current_cum_rate_index/avg_CumulIndices[i], 5 )
#print("On-Level Factors for premium:\n",onlevel)

"""## Adjusting Premiums for Rate Changes

"""

# Net Premium Earned (Earned Premium - Ceded Earned Premium(or Reinsurance costs))
net_prem_earned = {}
for i in range(1988,1998):
    net_prem_earned[i] = list( loss_data[loss_data['AccidentYear']==i]['EarnedPremNet_D'] )[0]

# print("Net Premium Earned\n")
# for i in net_prem_earned.keys():
#     print(i,"\t==>",net_prem_earned[i])
    





"""We will assume some rate changes."""

# Assume rate changes
rate_changes = {
            datetime.date(1988,4,1):0.05, #datetime.date(1989,1,1):0.1,
                datetime.date(1990,7,1):-0.02, #datetime.date(1991,4,1):-0.04,
            datetime.date(1991,5,1):0.11,  #datetime.date(1992,3,1):0.07,
            datetime.date(1993,8,1):-0.05, #datetime.date(1994,2,1):0.08,
            datetime.date(1996,8,1):0.15
                }
# first calculate the rate change indeces
rates = list(rate_changes.values())
rate_index =[1.00]+[ (1+i) for i in rates ] # including initial index of segment without changes = 1.00 (rate change = 0%)
#print("Rate change indeces:\n",rate_index)
cum_index = []
f = 1
for i in rate_index:
    f *= i
    cum_index.append( round(f, 4))
#print("Cumulative rate change indeces:\n",cum_index)
current_cum_rate_index = cum_index[-1]
#print("Current Cumulative Rate Level Index =",current_cum_rate_index)

# To calculate the portions earned by premiums under each rate change
import datetime
T = 1; E = 1;

def months_between(date1,date2):
    '''This function calculates the difference between 2 given dates in months
    date1, date2 are in datetime.date() format'''
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2    # difference between the dates

    return months/12



def find_remains(rate_dates, earned_prem_year, L):
    '''This function calculates the remaining portions of earned premium under the rate changes
       rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
    if L!=[]:
        L.append(0) # appending 0 as a means for calculating the last portion
        to_return = []  # the list that contains the portions
        max = 1 # maximum value (total area of an year of earned premium)
        for i in range(0, len(L)):
            if L[i]!=0:
                diff = max - L[i]   # calculate remaining portion
                to_return.append(round( diff,5))
                max = L[i]
                if L[i+1]==0:   # for the last portion to be appended
                    to_return.append(round( max,5))
            else:
                to_return.append(0)

        if to_return.count(0) == len(to_return):
            to_return = earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, to_return)

        to_return.pop()
        return to_return



def earnedPortion_ForUnaffectedYear(rate_dates, earned_prem_year, L):
    '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
    rate_dates is a dictionary containing the dates of rate changes, earned_prem_year is the year whose premiums are being adjusted, L is a list'''
    c = 0
    start_date = datetime.date(earned_prem_year,1,1)
    for i in rate_dates:
        if( months_between(i, start_date)>0 ):      # checking where to insert 1
            break
        else:
            c+=1
    L.insert(c,1)   # insert 1 as portion earned by premium
    return L



def earnedPortion(rate_dates, earned_prem_years):
    '''This function calculates the portion of earned premium under given rate changes
    rate_dates is a dictionary containing the dates of rate changes and earned_prem_years is also a dictionary containing the years in which premium is earned'''
    portion = {}
    for i in earned_prem_years:
        portion[i] = []
    for i in earned_prem_years:
        start_date = datetime.date(i,1,1)
        for j in rate_dates:

            if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
                # algorithm for calculating portions of earned premium
                D = months_between(j, start_date)

                A = D+T
                B = max( A-E, 0 )
                C = max( D, 0 )

                P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
                portion[i].append( round(P, 5))
            else:
                portion[i].append(0)

    # print(portion)
    for i in portion.keys():
        portion[i] = find_remains(rate_dates, i, portion[i])
    return(portion)


rate_effec_dates = list( rate_changes.keys())
years_toAdjust = list( net_prem_earned.keys() )
earned_NetPremPortion = earnedPortion(rate_effec_dates, years_toAdjust)
#print("The portion earned by the premium in the years w.r.t. the rate changes are:\n",earned_NetPremPortion)

# Average Cumulative Rate Level Indices

def AvgCumulIndices(L, cumul_indices):
    '''This function calculates the average cumulative rate level indices for the earned premium
    L, cumul_indices are numpy arrays where L contains the portions of earned premiums and cumul_indices contains the cumulative rate level indices'''
    prod = L*cumul_indices
    sum = 0
    for i in prod:
        sum+=i
    return round(sum, 5)

# changing to numpy arrays
for i in earned_NetPremPortion.keys():
    earned_NetPremPortion[i] = np.array(earned_NetPremPortion[i])
cum_index = np.array(cum_index)

avg_CumulIndices = {}
for i in earned_NetPremPortion.keys():
    avg_CumulIndices[i] = AvgCumulIndices(earned_NetPremPortion[i], cum_index)

# print("\nThe Average Cumulative Rate Level Indices are:")
# for i in avg_CumulIndices.keys():
#     print(i,"\t==>",avg_CumulIndices[i])

# On-Level Factors for the premiums
onlevel = {}
#print("Current Cumulative Rate Level Index =",current_cum_rate_index)
for i in avg_CumulIndices.keys():
    onlevel[i] = round( current_cum_rate_index/avg_CumulIndices[i], 5 )
# print("\nThe On-Level Factors are:")
# for i in onlevel.keys():
#     print(i,"\t==>",onlevel[i])

# On-Levelling the Premiums
AdjustedPrem = {}
for i in onlevel.keys():
    AdjustedPrem[i] = round( net_prem_earned[i] * onlevel[i], 5)
# print("The On-Level Net Premiums are:\n")
# for i in AdjustedPrem.keys():
#     print(i,"\t==>",AdjustedPrem[i])

"""## Adjusting Losses for Benefit Changes"""

# Assume benefit changes
benefit_changes = {
            datetime.date(1988,4,1):0.05,
            #     datetime.date(1989,1,1):0.1,
            datetime.date(1990,7,1):-0.02,
            #     datetime.date(1991,4,1):-0.04,
            datetime.date(1991,5,1):0.11,
            #     datetime.date(1992,3,1):0.07,
            datetime.date(1993,8,1):-0.05,
            #     datetime.date(1994,2,1):0.08,
            datetime.date(1996,8,1):0.15
                }
# first calculate the benefit change indeces
benefits = list(benefit_changes.values())
benefit_index =[1.00]+[ (1+i) for i in benefits ] # including initial index without changes = 1.00 (rate change = 0%)
# print("Benefit change indeces:\n",benefit_index)
loss_lvl = []
f = 1
for i in benefit_index:
    f *= i
    loss_lvl.append( round(f, 4))
# print("Loss Level indeces (Cumulative indeces):\n",loss_lvl)
current_loss_lvl = loss_lvl[-1]
# print("Current Loss Level =",current_loss_lvl)

import datetime
T = 1; E = 1;

def months_between(date1,date2):
    '''This function calculates the difference between 2 given dates in months
    date1, date2 are in datetime.date() format'''
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m1-m2    # difference between the dates

    return months/12



def find_remains(ben_dates, loss_year, L):
    '''This function calculates the remaining portions of earned premium under the rate changes
       ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
    if L!=[]:
        L.append(0) # appending 0 as a means for calculating the last portion
        to_return = []  # the list that contains the portions
        max = 1 # maximum value (total area of an year of losses)
        for i in range(0, len(L)):
            if L[i]!=0:
                diff = max - L[i]   # calculate remaining portion
                to_return.append(round( diff,5))
                max = L[i]
                if L[i+1]==0:   # for the last portion to be appended
                    to_return.append(round( max,5))
            else:
                to_return.append(0)

        if to_return.count(0) == len(to_return):
            to_return = Portion_ForUnaffectedYear(ben_dates, loss_year, to_return)

        to_return.pop()
        return to_return



def Portion_ForUnaffectedYear(ben_dates, loss_year, L):
    '''This function sets the portion earned by premium for that year as 1 if there are no rate changes affecting that year
    ben_dates is a list containing the dates of benefit changes, loss_year is the year whose losses are being adjusted, L is a list'''
    c = 0
    start_date = datetime.date(loss_year,1,1)
    for i in ben_dates:
        if( months_between(i, start_date)>0 ):      # checking where to insert 1
            break
        else:
            c+=1
    L.insert(c,1)   # insert 1 as portion earned by premium
    return L



def LossPortion(ben_dates, loss_years):
    '''This function calculates the portion of earned premium under given rate changes
    ben_dates is a list containing the dates of benefit changes and loss_years is also a list containing the years in which losses occur'''
    portion = {}
    for i in loss_years:
        portion[i] = []
    for i in loss_years:
        start_date = datetime.date(i,1,1)
        for j in ben_dates:

            if months_between(j, start_date)<1 and months_between(j, start_date)>-1:
                # algorithm for calculating portions of earned premium
                D = months_between(j, start_date)

                A = D+T
                B = max( A-E, 0 )
                C = max( D, 0 )

                P = 1 - ( (pow(A,2)-pow(B,2)-pow(C,2)) / (2*E*T) )
                portion[i].append( round(P, 5))
            else:
                portion[i].append(0)

    # print(portion)
    for i in portion.keys():
        portion[i] = find_remains(ben_dates, i, portion[i])
    return(portion)


ben_effec_dates = list( benefit_changes.keys())
years_toAdjust = list( proj_ultLosses.keys() )
LossesPortion = LossPortion(ben_effec_dates, years_toAdjust)
# print("The portion of the losses in the years w.r.t. the benefit changes are:\n",LossesPortion)

# Average Loss Levels

def AvgLossLevel(L, loss_levels):
    '''This function calculates the average Loss levels for the historical periods
    L, loss_levels are numpy arrays where L contains the portions of losses and loss_levels contains the loss levels'''
    prod = L*loss_levels
    sum = 0
    for i in prod:
        sum+=i
    return round(sum, 5)

for i in LossesPortion.keys():
    LossesPortion[i] = np.array(LossesPortion[i])
loss_lvl = np.array(loss_lvl)

avg_LossLvl = {}
for i in LossesPortion.keys():
    avg_LossLvl[i] = AvgLossLevel(LossesPortion[i], loss_lvl)

# print("\nThe Average Loss Levels are:")
# for i in avg_LossLvl.keys():
#     print(i,"\t==>",avg_LossLvl[i])

# Adjustment Factors
adjusts = {}
# print("Current Loss Level =",current_loss_lvl)
for i in avg_LossLvl.keys():
    adjusts[i] = round( current_loss_lvl/avg_LossLvl[i], 5 )
# print("\nThe Adjustment Factors are:")
# for i in adjusts.keys():
#     print(i,"\t==>",adjusts[i])

# Adjusting the Losses
AdjustedLosses = {}
for i in adjusts.keys():
    AdjustedLosses[i] = round( proj_ultLosses[i] * adjusts[i], 5)
# print("The Adjusted Losses are:\n")
# for i in AdjustedLosses.keys():
#     print(i,"\t==>",AdjustedLosses[i])

"""# Trending Loss Ratios

### We are getting data related to Annual Inflation Rates by country from World Bank's website: [data.worldbank.org](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=US&view=chart)
"""

# Lets work on Inflation Rates first
filepath = "./605_InflationRates.xlsx"

inflation_rates = pd.read_excel(filepath)
inflation_rates

# inflation rates in USA
inf_us = inflation_rates[inflation_rates['Country Name'] == "United States"]
inf_us

inf_us.iloc[0,1998-1960+4]

inf_index = {}
start = 1988; end = 1997
for i in range(start, end+1):
    inf_index[i] = inf_us.iloc[0, i-1960+4]

# print("Inflation indeces are:\n",inf_index)

inf_avg = {}
keys = list(inf_index.keys())
for i in range(0,len(keys)):
    avg=0
    for j in range(i,len(keys)):
        avg+= inf_index[keys[j]]
    inf_avg[keys[i]] = avg/(j-i+1)

# print("Average Inflation rates =",inf_avg)

"""## Our Assumptions are:
### --> Policies are written uniformly over time.
### --> Premiums are earned uniformly over the policy period.
### --> Losses occur uniformly over the policy period.
### --> Policies have annual terms.

## Trend losses for inflation.
##### Our experience periods are the historical accident years from 1988 to 1997.
##### Future policy period begins in Jan 1, 1998. Inflation rate will be in effect for 12 months. Thus our forecast period average accident date is:
##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
"""

loss_inf_period = {}
loss_forecast_Date = datetime.date(1999,1,1)
for i in inf_index.keys():
    expDate = datetime.date(i,7,1)
    diff = months_between(loss_forecast_Date,expDate)
    loss_inf_period[i] = diff

# print("The trend periods for losses are :\n", loss_inf_period)

loss_inf_factor = {}
for i in loss_inf_period.keys():
    loss_inf_factor[i] = (1 + (0.01*inf_avg[i]))**loss_inf_period[i]

# print("The trend factors for losses are :\n", loss_inf_factor)

# Now we trend the losses
inf_trendedLosses = {}
for i in loss_inf_factor.keys():
    inf_trendedLosses[i] = AdjustedLosses[i]*loss_inf_factor[i]

# print("The Projected_Ultimate_Losses\tBenefit_Adjusted_Losses\tInflation_Trended_Losses are:\n")
# for i in inf_trendedLosses.keys():
#     print(i,"\t  ===>",proj_ultLosses[i],"\t ===>",AdjustedLosses[i],"\t==>",inf_trendedLosses[i])

"""## Trend Premiums for inflation.
##### Trend will be estimated from earned premium data. The trend period will be from the average earned date in each historical period to the average earned date at the new rate level. Because of the uniform assumption, the average earned date of a period is the midpoint of the first and last dates that premiums could be earned in that period. So, these dates will depend on the policy term length.
##### Future policy period begins in Jan 1, 1998. Inflation rate will be in effect for 12 months. Thus our forecast period average earned date is:
##### Midpoint of the period 1/1/1998 to 12/31/1999 = 1/1/1999
"""

prem_inf_period = {}
prem_forecast_Date = datetime.date(1999,1,1)
for i in inf_index.keys():
    expDate = datetime.date(i,1,1)
    diff = months_between(prem_forecast_Date,expDate)
    prem_inf_period[i] = diff

# print("The trend periods for premium are :\n", prem_inf_period)

prem_inf_factor = {}
for i in prem_inf_period.keys():
    prem_inf_factor[i] = (1 + (0.01*inf_avg[i]))**prem_inf_period[i]

# print("The trend factors for premium are :\n", prem_inf_factor)

# Now we trend the premiums
inf_trendedPrems = {}
for i in prem_inf_factor.keys():
    inf_trendedPrems[i] = AdjustedPrem[i]*prem_inf_factor[i]

# print("The Net_Premiums_Earned\tRate_Adjusted_Premiums\tInflation_Trended_Premiums are:\n")
# for i in inf_trendedPrems.keys():
#     print(i,"\t  ===>",net_prem_earned[i],"\t ===>",AdjustedPrem[i],"\t==>",inf_trendedPrems[i])

"""# Expenses and Profits

## Assume fixed expense provision and variable expense provision. Also assume underwiting profit provision.
"""

fixed_exp_provision = 0.08      # 8%
variable_exp_provision = 0.1    # 10%
profit_provision = 0.07         # 7%
ulae_ratio = 0.05               # 5%

# permissible loss ratio
permissibleLR = 1 - (variable_exp_provision+profit_provision)
# print("Permissible Loss Ratio = ", round(permissibleLR*100,3),"%")

"""# Overall Indicated Rate Change"""

# find the loss and alae ratios
loss_ratio = {}
for i in inf_trendedLosses.keys():
    loss_ratio[i] = inf_trendedLosses[i]/inf_trendedPrems[i]
avg_loss_ratio = 0
for i in loss_ratio.keys():
    avg_loss_ratio+=loss_ratio[i]
avg_loss_ratio/=len(loss_ratio.keys())
avg_loss_ratio*=(1+ulae_ratio)
# print("Average loss ratio = ",round(avg_loss_ratio*100,2),"%")

# if(avg_loss_ratio <= permissibleLR):
#     print("Since, average loss ratio %.3f is less than permissible loss ratio %.3f,\nThe Company met underwriting profit expectations.\n"%(avg_loss_ratio,permissibleLR))
# else :
#     print("Since, average loss ratio %.3f is greater than permissible loss ratio %.3f,\nThe Company did not meet underwriting profit expectations.\n"%(avg_loss_ratio,permissibleLR))

# find overall rate level indicated change
indicated_avg_rate_change = ((avg_loss_ratio+fixed_exp_provision)/(1-variable_exp_provision-profit_provision)) - 1
# print("Indicated average rate change for is=",round(indicated_avg_rate_change*100,4),"%")

st.write("Overall change",round(indicated_avg_rate_change*100,4))