# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:07:38 2019

@author: zclna
"""

import math
from mpmath import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.gridspec import GridSpec

def plot_tangent(starting_point, ending_point, tangent_line_range):
    x1=math.log10(starting_point[0])
    y1=starting_point[1]
    x2=math.log10(ending_point[0])
    y2=ending_point[1]
    
    slope=(y2-y1)/(x2-x1)
    intercept=y1-slope*x1
    
    x3=np.linspace(tangent_line_range[0],tangent_line_range[1],10)
    y3=[]
    for i in x3:
        value=slope*math.log10(i)+intercept
        y3.append(value)
    
    return (x3,y3,slope,intercept)

def plot_creep_line(data_set_x, data_set_y, polyfit_range):   
    x_creep=[]
    for i in data_set_x[polyfit_range[0]:polyfit_range[1]]:
        value = math.log10(i)
        x_creep.append(value)
    
    y_creep=data_set_y[polyfit_range[0]:polyfit_range[1]]
    z1 = np.polyfit(x_creep, y_creep, 1)
    slope=z1[0]
    intercept=z1[1]

    x1=np.linspace(0.1,2000,10)
    y1=[]
    for i in x1:
        value = slope*math.log10(i)+intercept
        y1.append(value)
    
    return (x1,y1,slope,intercept)


Area = 4.40E-03
MOULD_HEIGHT = 20.0733 #mm
Hs = 13.891 #mm height of solids
SG = 2.66685 #gr/cm3 specific gravity
Load_1 = 1.25 #kg
Load_2 = 2.5 #kg
Load_3 = 5 #kg
Load_4 = 10 #kg


SELECTED_TIME_1=[0.0001, 0.1251, 0.2501, 0.5001, 1.0001, 2.0001, 4.0001, 8.0001,
                 16.0001, 32.0001, 60.0001, 120.0001, 240.0001, 300.0001, 360.0001,
                 420.0001, 480.0001, 540.0001, 600.0001, 720.0001, 840.0001,
                 960.0001, 1080.0001, 1200.0001, 1440.0001, 1500.0001]
SELECTED_TIME_2=[0.0001, 0.1251, 0.2501, 0.5001, 1.0001, 2.0001, 4.0001, 8.0001,
                 16.0001, 32.0001, 60.0001, 120.0001, 240.0001, 300.0001, 360.0001,
                 420.0001, 480.0001, 540.0001, 600.0001, 720.0001, 840.0001,
                 960.0001, 1080.0001, 1200.0001, 1440.0001, 1680.0001, 1920.0001,
                 2160.0001, 2400.0001, 3000.0001, 3600.0001]
SELECTED_TIME_3=[0.0001, 0.1251, 0.2501, 0.5001, 1.0001, 2.0001, 4.0001, 8.0001,
                 16.0001, 32.0001, 60.0001, 120.0001, 240.0001, 300.0001, 360.0001,
                 420.0001, 480.0001, 540.0001, 600.0001, 720.0001, 840.0001,
                 960.0001, 1080.0001, 1200.0001, 1440.0001]
SELECTED_TIME_4=[0.0001, 0.1251, 0.2501, 0.5001, 1.0001, 2.0001, 4.0001, 8.0001,
                 16.0001, 32.0001, 60.0001, 120.0001, 240.0001, 300.0001, 360.0001,
                 420.0001, 480.0001, 540.0001, 600.0001, 720.0001, 840.0001,
                 960.0001, 1080.0001, 1200.0001, 1440.0001]
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 15,
        } 

df=pd.read_excel('data.xlsx')

df.Time -= df.Time[0] #Tare the first reading to 0

df['Time_min'] = df.Time/60 #create a new column to convert time_s to time_min
df['Force'] = df.Load*9.81/1000 #create a new column called force
df['Vertical stress'] = df.Force/Area

df1 = df[df.Load==1.25]
df2 = df[df.Load==2.50]
df3 = df[df.Load==5.00]
df4 = df[df.Load==10.00]

"""-----------------------------------------------------Overall-------------"""
#overall plot
fig=plt.figure(num='Overview for all load increments')
ax0=fig.add_subplot(1,1,1)      
ax0.plot(df1['Time_min'], df1['Vertical extension'],label='25kPa')
ax0.plot(df2['Time_min'], df2['Vertical extension'],label='50kPa')
ax0.plot(df3['Time_min'], df3['Vertical extension'],label='100kPa')
ax0.plot(df4['Time_min'], df4['Vertical extension'],label='200kPa')
ax0.set_title('Settlement curve for all load increments', fontdict=font, pad=10)
ax0.set_xlabel('Time (min)', fontdict=font)
ax0.set_ylabel('Settlement (mm)', fontdict=font)
legend = ax0.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

#The reason to add 0.0001 is that log plot has a non-zero staring point 
df4.loc[:,'Time_min']=df4.loc[:,'Time_min']-df3.loc[df3.index[-1],'Time_min']+0.0001
df3.loc[:,'Time_min']=df3.loc[:,'Time_min']-df2.loc[df2.index[-1],'Time_min']+0.0001
df2.loc[:,'Time_min']=df2.loc[:,'Time_min']-df1.loc[df1.index[-1],'Time_min']+0.0001
df1.loc[:,'Time_min']=df1.loc[:,'Time_min']+0.0001

"""--------------------------------------------------Load cycle 1"""
x=[]
y=[]
for i in SELECTED_TIME_1:
    x.append(i)
    y.append(abs(df1['Vertical extension'][df1.Time_min==i].values[0]))

fig=plt.figure(num='Load Increment 1 - 1.25kN',constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.2) 

ax1 = fig.add_subplot(gs[0, 0])   
ax1.plot(x, y,ls='-',marker='o',label='200kPa')
ax1.invert_yaxis()
ax1.set_title('Settlement Curve (Time in normal scale)', fontdict=font, pad=10)
ax1.set_xlabel('Time (min)', fontdict=font)
ax1.set_ylabel('Settlement (mm)', fontdict=font)
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
 
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x,y,ls='-',marker='o',label='25kPa')
ax2.semilogx()
ax2.invert_yaxis()
ax2.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax2.set_xlabel('Time (min)', fontdict=font)
ax2.set_ylabel('Settlement (mm)', fontdict=font)
ax2.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax2.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax3=fig.add_subplot(2,1,2)
ax3.plot(x, y,ls='-',marker='o',label='25kPa')  
ax3.semilogx()
ax3.invert_yaxis()
ax3.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax3.set_xlabel('Time (min)', fontdict=font)
ax3.set_ylabel('Settlement (mm)', fontdict=font)
ax3.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax3.grid(True, which="both",axis="y", ls="-", color='0.3')

#plot the steepest tangent line
tangent=plot_tangent((x[3],y[3]),(x[4],y[4]),(0.01,1000))
ax3.plot(tangent[0],tangent[1],label='steepest tangent')
#plot the straight line in creep stage
creep=plot_creep_line(x, y, (-3,-1))
ax3.plot(creep[0],creep[1],label='creep straightline')
legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

"""find the intersection of the straight line and the tangent line"""
#tangent line function under log scale: y=tangent[2]*x+tangent[3]
#tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
#straight line function under log scale: y=creep[2]*x+creep[3]
#straight line function under normal scale: y=(10**creep[3])*(x**(creep[2]))

intersection_x = 10**((creep[3]-tangent[3])/(tangent[2]-creep[2]))
intersection_y = tangent[2]*math.log10(intersection_x)+tangent[3]

plt.plot(intersection_x,intersection_y,marker='1')
plt.axvline(x=intersection_x,ls='--',linewidth=1, color='red')
plt.axhline(y=intersection_y,ls='--',linewidth=1, color='red')


"""labels and annotations"""
#print(ax3.get_ylim())
plt.annotate(xy=[intersection_x,ax3.get_ylim()[1]], s=f'T100={round(intersection_x,3)}')
plt.annotate(xy=[ax3.get_xlim()[0],intersection_y], s='100% Primary Consolidation') 

settlement_1 = y[-1]-y[0] # delta_H for the fourth load cycle
total_settlement_1 = 0 + settlement_1 #total_settlement_0 = 0
h_1 = MOULD_HEIGHT - total_settlement_1
average_height_1 = (MOULD_HEIGHT+h_1)/(2*10) #convert mm to cm

#find and plot 50% settlement
plt.axhline(y=y[0],ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],y[0]], s='0% Primary Consolidation') 
settlement_for_50_consolidation = (intersection_y+y[0])/2
plt.axhline(y=settlement_for_50_consolidation,ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],settlement_for_50_consolidation], s='50% Primary Consolidation') 
#find and plot the time for 50% settlement
#remember tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
t50_1 = (10**(settlement_for_50_consolidation)/(10**tangent[3]))**(1/tangent[2])
plt.axvline(x=t50_1,ls='--',linewidth=1, color='red')
plt.annotate(xy=[t50_1,ax3.get_ylim()[1]], s=f'T50={round(t50_1,3)}')

Cv_1 = 0.196*(average_height_1**2)/(t50_1*60) #convert the unit of t50 from s to min

"""---------------------------------------------Load cycle 2"""
x=[]
y=[]
for i in SELECTED_TIME_2:
    x.append(i)
    y.append(abs(df2['Vertical extension'][df2.Time_min==i].values[0]))

fig=plt.figure(num='Load Increment 2 - 2.5kN',constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.2) 

ax1 = fig.add_subplot(gs[0, 0])   
ax1.plot(x, y,ls='-',marker='o',label='200kPa')
ax1.invert_yaxis()
ax1.set_title('Settlement Curve (Time in normal scale)', fontdict=font, pad=10)
ax1.set_xlabel('Time (min)', fontdict=font)
ax1.set_ylabel('Settlement (mm)', fontdict=font)
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
 
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x,y,ls='-',marker='o',label='50kPa')
ax2.semilogx()
ax2.invert_yaxis()
ax2.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax2.set_xlabel('Time (min)', fontdict=font)
ax2.set_ylabel('Settlement (mm)', fontdict=font)
ax2.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax2.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax3=fig.add_subplot(2,1,2)
ax3.plot(x, y,ls='-',marker='o',label='50kPa')  
ax3.semilogx()
ax3.invert_yaxis()
ax3.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax3.set_xlabel('Time (min)', fontdict=font)
ax3.set_ylabel('Settlement (mm)', fontdict=font)
ax3.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax3.grid(True, which="both",axis="y", ls="-", color='0.3')

#plot the steepest tangent line
tangent=plot_tangent((x[1],y[1]),(x[2],y[2]),(0.01,1000))
ax3.plot(tangent[0],tangent[1],label='steepest tangent')
#plot the straight line in creep stage
creep=plot_creep_line(x, y, (-5,-1))
ax3.plot(creep[0],creep[1],label='creep straightline')
legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

"""find the intersection of the straight line and the tangent line"""
#tangent line function under log scale: y=tangent[2]*x+tangent[3]
#tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
#straight line function under log scale: y=creep[2]*x+creep[3]
#straight line function under normal scale: y=(10**creep[3])*(x**(creep[2]))

intersection_x = 10**((creep[3]-tangent[3])/(tangent[2]-creep[2]))
intersection_y = tangent[2]*math.log10(intersection_x)+tangent[3]

plt.plot(intersection_x,intersection_y,marker='1')
plt.axvline(x=intersection_x,ls='--',linewidth=1, color='red')
plt.axhline(y=intersection_y,ls='--',linewidth=1, color='red')

"""labels and annotations"""
#print(ax3.get_ylim())
plt.annotate(xy=[intersection_x,ax3.get_ylim()[1]], s=f'T100={round(intersection_x,3)}')
plt.annotate(xy=[ax3.get_xlim()[0],intersection_y], s='100% Primary Consolidation') 

settlement_2 = y[-1]-y[0] # delta_H for the fourth load cycle
total_settlement_2 = total_settlement_1 + settlement_2
h_2 = MOULD_HEIGHT - total_settlement_2
average_height_2 = (h_1+h_2)/(2*10) #convert mm to cm

#find and plot 50% settlement
plt.axhline(y=y[0],ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],y[0]], s='0% Primary Consolidation') 
settlement_for_50_consolidation = (intersection_y+y[0])/2
plt.axhline(y=settlement_for_50_consolidation,ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],settlement_for_50_consolidation], s='50% Primary Consolidation') 
#find and plot the time for 50% settlement
#remember tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
t50_2 = (10**(settlement_for_50_consolidation)/(10**tangent[3]))**(1/tangent[2])
plt.axvline(x=t50_2,ls='--',linewidth=1, color='red')
plt.annotate(xy=[t50_2,ax3.get_ylim()[1]], s=f'T50={round(t50_2,3)}')

Cv_2 = 0.196*(average_height_2**2)/(t50_2*60) #convert the unit of t50 from s to min

"""---------------------------------------------Load cycle 3"""
x=[]
y=[]
for i in SELECTED_TIME_3:
    x.append(i)
    y.append(abs(df3['Vertical extension'][df3.Time_min==i].values[0]))

fig=plt.figure(num='Load Increment 3 - 5kN',constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.2) 

ax1 = fig.add_subplot(gs[0, 0])   
ax1.plot(x, y,ls='-',marker='o',label='200kPa')
ax1.invert_yaxis()
ax1.set_title('Settlement Curve (Time in normal scale)', fontdict=font, pad=10)
ax1.set_xlabel('Time (min)', fontdict=font)
ax1.set_ylabel('Settlement (mm)', fontdict=font)
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
 
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x,y,ls='-',marker='o',label='100kPa')
ax2.semilogx()
ax2.invert_yaxis()
ax2.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax2.set_xlabel('Time (min)', fontdict=font)
ax2.set_ylabel('Settlement (mm)', fontdict=font)
ax2.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax2.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax3=fig.add_subplot(2,1,2)
ax3.plot(x[1:], y[1:],ls='-',marker='o',c="green",label='100kPa')  
ax3.semilogx()
ax3.invert_yaxis()
ax3.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax3.set_xlabel('Time (min)', fontdict=font)
ax3.set_ylabel('Settlement (mm)', fontdict=font)
ax3.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax3.grid(True, which="both",axis="y", ls="-", color='0.3')

#make line smooth
xnew = np.linspace(x[0],x[2],30) 
x = np.array(x)
y = np.array(y)
spl = make_interp_spline(x, y, k=3) #BSpline object
y_smooth = spl(xnew)
plt.plot(xnew,y_smooth,c="green",label='steepest tangent')
#plot the steepest tangent line
tangent=plot_tangent((x[1],y[1]),(x[2],y[2]),(0.01,100))
ax3.plot(tangent[0],tangent[1],label='creep straightline')
#plot the straight line in creep stage
creep=plot_creep_line(x, y, (-10,-1))
ax3.plot(creep[0],creep[1])
legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
"""find the intersection of the straight line and the tangent line"""
#tangent line function under log scale: y=tangent[2]*x+tangent[3]
#tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
#straight line function under log scale: y=creep[2]*x+creep[3]
#straight line function under normal scale: y=(10**creep[3])*(x**(creep[2]))

intersection_x = 10**((creep[3]-tangent[3])/(tangent[2]-creep[2]))
intersection_y = tangent[2]*math.log10(intersection_x)+tangent[3]

plt.plot(intersection_x,intersection_y,marker='1')
plt.axvline(x=intersection_x,ls='--',linewidth=1, color='red')
plt.axhline(y=intersection_y,ls='--',linewidth=1, color='red')

"""labels and annotations"""
#print(ax3.get_ylim())
plt.annotate(xy=[intersection_x,ax3.get_ylim()[1]], s=f'T100={round(intersection_x,3)}')
plt.annotate(xy=[ax3.get_xlim()[0],intersection_y], s='100% Primary Consolidation') 

settlement_3 = y[-1]-y[0] # delta_H for the fourth load cycle
total_settlement_3 = total_settlement_2 + settlement_3
h_3 = MOULD_HEIGHT - total_settlement_3
average_height_3 = (h_2+h_3)/(2*10) #convert mm to cm

#find and plot 50% settlement
plt.axhline(y=y[0],ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],y[0]], s='0% Primary Consolidation') 
settlement_for_50_consolidation = (intersection_y+y[0])/2
plt.axhline(y=settlement_for_50_consolidation,ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],settlement_for_50_consolidation], s='50% Primary Consolidation') 
#find and plot the time for 50% settlement
#remember tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
t50_3 = (10**(settlement_for_50_consolidation)/(10**tangent[3]))**(1/tangent[2])
plt.axvline(x=t50_3,ls='--',linewidth=1, color='red')
plt.annotate(xy=[t50_3,ax3.get_ylim()[1]], s=f'T50={round(t50_3,3)}')

Cv_3 = 0.196*(average_height_3**2)/(t50_3*60) #convert the unit of t50 from s to min

"""---------------------------------------------Load cycle 4"""
x=[]
y=[]
for i in SELECTED_TIME_4:
    x.append(i)
    y.append(abs(df4['Vertical extension'][df4.Time_min==i].values[0]))

fig=plt.figure(num='Load Increment 4 - 10kN',constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.2) 

ax1 = fig.add_subplot(gs[0, 0])   
ax1.plot(x, y,ls='-',marker='o',label='200kPa')
ax1.invert_yaxis()
ax1.set_title('Settlement Curve (Time in normal scale)', fontdict=font, pad=10)
ax1.set_xlabel('Time (min)', fontdict=font)
ax1.set_ylabel('Settlement (mm)', fontdict=font)
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
 
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x,y,ls='-',marker='o',label='200 kPa')
ax2.semilogx()
ax2.invert_yaxis()
ax2.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax2.set_xlabel('Time (min)', fontdict=font)
ax2.set_ylabel('Settlement (mm)', fontdict=font)
ax2.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax2.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white')                  

ax3=fig.add_subplot(gs[1, :])
ax3.plot(x[1:], y[1:],ls='-',marker='o',c="green",label='200 kPa')  
ax3.semilogx()
ax3.invert_yaxis()
ax3.set_title('Settlement Curve (Time in log scale)', fontdict=font, pad=10)
ax3.set_xlabel('Time (min)', fontdict=font)
ax3.set_ylabel('Settlement (mm)', fontdict=font)
ax3.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax3.grid(True, which="both",axis="y", ls="-", color='0.3')
#make line smooth
xnew = np.linspace(x[0],x[2],30) 
x = np.array(x)
y = np.array(y)
spl = make_interp_spline(x, y, k=3) #BSpline object
y_smooth = spl(xnew)
plt.plot(xnew,y_smooth,c="green") 
#plot the steepest tangent line
tangent=plot_tangent((x[1],y[1]),(x[2],y[2]),(0.01,10))
ax3.plot(tangent[0],tangent[1],label='steepest tangent')
#plot the straight line in creep stage
creep=plot_creep_line(x, y, (-10,-1))
ax3.plot(creep[0],creep[1],label='creep straightline')
legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 
"""find the intersection of the straight line and the tangent line"""
#tangent line function under log scale: y=tangent[2]*x+tangent[3]
#tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
#straight line function under log scale: y=creep[2]*x+creep[3]
#straight line function under normal scale: y=(10**creep[3])*(x**(creep[2]))

intersection_x = 10**((creep[3]-tangent[3])/(tangent[2]-creep[2]))
intersection_y = tangent[2]*math.log10(intersection_x)+tangent[3]

plt.plot(intersection_x,intersection_y,marker='1')
plt.axvline(x=intersection_x,ls='--',linewidth=1, color='red')
plt.axhline(y=intersection_y,ls='--',linewidth=1, color='red')

"""labels and annotations"""
#print(ax3.get_ylim())
plt.annotate(xy=[intersection_x,ax3.get_ylim()[1]], s=f'T\u2081\u2080\u2080={round(intersection_x,3)}')
plt.annotate(xy=[ax3.get_xlim()[0],intersection_y], s='100% Primary Consolidation') 

settlement_4 = y[-1]-y[0] # delta_H for the fourth load cycle
total_settlement_4 = total_settlement_3 + settlement_4
h_4 = MOULD_HEIGHT - total_settlement_4
average_height_4 = (h_3+h_4)/(2*10) #convert mm to cm

#find and plot 50% settlement
plt.axhline(y=y[0],ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],y[0]], s='0% Primary Consolidation') 
settlement_for_50_consolidation = (intersection_y+y[0])/2
plt.axhline(y=settlement_for_50_consolidation,ls='--',linewidth=1, color='red')
plt.annotate(xy=[ax3.get_xlim()[0],settlement_for_50_consolidation], s='50% Primary Consolidation') 
#find and plot the time for 50% settlement
#remember tangent line function under normal scale: y=(10**tangent[3])*(x**(tangent[2]))
t50_4 = (10**(settlement_for_50_consolidation)/(10**tangent[3]))**(1/tangent[2])
plt.axvline(x=t50_4,ls='--',linewidth=1, color='red')
plt.annotate(xy=[t50_4,ax3.get_ylim()[1]], s=f'T\u2085\u2080={round(t50_4,3)}')

Cv_4 = 0.196*(average_height_4**2)/(t50_4*60) #convert the unit of t50 from s to min


"""------------------------------------------------------- overall result"""
#t50
print('-'*35)
t50_result = {'25 kPa':t50_1,
             '50 kPa':t50_2,
             '100 kPa':t50_3,
             '200 kPa':t50_4}
for i in t50_result:
   print(f"t50 for {i} is {round(t50_result[i],3)} min")
#Cv
print('-'*35)
Cv_result = {'25 kPa':Cv_1,
             '50 kPa':Cv_2,
             '100 kPa':Cv_3,
             '200 kPa':Cv_4}  
#cv_result.keys()
#cv_result.values()  
for i in Cv_result:
    print(f"Cv for {i} is {round(Cv_result[i],4)} cm2/s")
Cv_average = sum(Cv_result.values())/4
print(f"The avergae Cv is {round(Cv_average,4)} cm2/s")   
#Hv
Hv_0=MOULD_HEIGHT-Hs     
Hv_1=MOULD_HEIGHT-Hs-total_settlement_1   
Hv_2=MOULD_HEIGHT-Hs-total_settlement_2 
Hv_3=MOULD_HEIGHT-Hs-total_settlement_3 
Hv_4=MOULD_HEIGHT-Hs-total_settlement_4 
#void ratio e 
print('-'*35)   
e_0 = Hv_0/Hs
e_1 = Hv_1/Hs
e_2 = Hv_2/Hs
e_3 = Hv_3/Hs
e_4 = Hv_4/Hs
e_result = {'0.1 kPa':e_0,
            '25 kPa':e_1,
            '50 kPa':e_2,
            '100 kPa':e_3,
            '200 kPa':e_4}
for i in e_result:
   print(f"Void ratio e for {i} is {round(e_result[i],3)}")
#pressure
p_0 = 0.1 #kPa
p_1 = (Load_1*9.81/Area)/1000
p_2 = (Load_2*9.81/Area)/1000
p_3 = (Load_3*9.81/Area)/1000
p_4 = (Load_4*9.81/Area)/1000
lst_pressure = [p_0,p_1,p_2,p_3,p_4]
#Mv
print('-'*35)
Mv_1 = (e_0-e_1)/(p_1-p_0)*1/(1+e_1)
Mv_2 = (e_1-e_2)/(p_2-p_1)*1/(1+e_2)
Mv_3 = (e_2-e_3)/(p_3-p_2)*1/(1+e_3)
Mv_4 = (e_3-e_4)/(p_4-p_3)*1/(1+e_4)
Mv_result = {'25 kPa':Mv_1,
             '50 kPa':Mv_2,
             '100 kPa':Mv_3,
             '200 kPa':Mv_4}
for i in Mv_result:
   print(f"Mv for {i} is {round(Mv_result[i],6)} 1/kPa")
Mv_average = sum(Mv_result.values())/4
print(f"The avergae Mv is {round(Mv_average,6)} 1/kPa") 
#permeability k
print('-'*35)
k_1 = Cv_1*Mv_1*9.81/10000 #m/s
k_2 = Cv_2*Mv_2*9.81/10000
k_3 = Cv_3*Mv_3*9.81/10000
k_4 = Cv_4*Mv_4*9.81/10000
k_result = {'25 kPa':k_1,
             '50 kPa':k_2,
             '100 kPa':k_3,
             '200 kPa':k_4}
for i in k_result:
   print(f"k for {i} is {round(k_result[i],10)} m/s")
k_average = sum(k_result.values())/4
print(f"The avergae k is {round(k_average,10)} m/s") 
print('-'*35)
dd_0 = SG/(e_0+1)
dd_1 = SG/(e_1+1)
dd_2 = SG/(e_2+1)
dd_3 = SG/(e_3+1)
dd_4 = SG/(e_4+1)
dd_result = {'0.1 kPa':dd_0,
            '25 kPa':dd_1,
            '50 kPa':dd_2,
            '100 kPa':dd_3,
            '200 kPa':dd_4}
for i in dd_result:
   print(f"Dry Density for {i} is {round(dd_result[i],3)} gr/cm3")
print('-'*35)

"""--------------------------------------Plot void ratio e VS Applied stress"""
lst_p = [p_0,p_1,p_2,p_3,p_4]
lst_k = list(k_result.values())
lst_e = list(e_result.values())
lst_dd = list(dd_result.values())


fig=plt.figure(num='Other Properties',constrained_layout=True)
fig.subplots_adjust(hspace=0.3) 

ax1 = fig.add_subplot(3,2,1)   
ax1.plot(lst_p, lst_e,ls='-',marker='o',label='Void Ratio e')
ax1.set_title('Void Ratio VS Applied Stress', fontdict=font, pad=10)
ax1.set_xlabel('Applied Stress (kPa)', fontdict=font)
ax1.set_ylabel('Void Ratio e', fontdict=font)
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax2 = fig.add_subplot(3,2,2)   
ax2.plot(lst_p, lst_e,ls='-',marker='o',label='Void Ratio e')
ax2.semilogx()
ax2.set_title('Void Ratio VS Applied Stress', fontdict=font, pad=10)
ax2.set_xlabel('Applied Stress (kPa)', fontdict=font)
ax2.set_ylabel('Void Ratio e', fontdict=font)
ax2.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax2.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax3 = fig.add_subplot(3,2,3)   
ax3.plot(lst_p, lst_dd,ls='-',marker='o',label='Dry Density')
ax3.set_title('Dry Density VS Applied Stress', fontdict=font, pad=10)
ax3.set_xlabel('Applied Stress (kPa)', fontdict=font)
ax3.set_ylabel('Dry Density (t/m3)', fontdict=font)
legend = ax3.legend(loc='lower right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white')

ax4 = fig.add_subplot(3,2,4)   
ax4.plot(lst_p, lst_dd,ls='-',marker='o',label='Dry Density')
ax4.semilogx()
ax4.set_title('Dry Density VS Applied Stress', fontdict=font, pad=10)
ax4.set_xlabel('Applied Stress (kPa)', fontdict=font)
ax4.set_ylabel('Dry Density (t/m3)', fontdict=font)
ax4.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax4.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax4.legend(loc='lower right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 

ax5 = fig.add_subplot(3,2,5)   
ax5.plot(lst_p[1:], lst_k,ls='-',marker='o',label='Saturated Hydraulic Conductivity')
ax5.loglog()
ax5.set_xlim((0.1, 1000))
ax5.set_ylim((1e-9, 1e-5))
ax5.set_title('Hydraulic Conductivity VS Applied Stress', fontdict=font, pad=10)
ax5.set_xlabel('Applied Stress (kPa)', fontdict=font)
ax5.set_ylabel('Saturated Hydraulic Conductivity (m/s)', fontdict=font)
ax5.grid(True, which="both",axis="x", ls="--", color='0.5') #color=opacity
ax5.grid(True, which="both",axis="y", ls="-", color='0.3')
legend = ax5.legend(loc='upper right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('white') 















    
    