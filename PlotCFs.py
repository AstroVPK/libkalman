import numpy as np
import math as m
import KalmanFast as KF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from mpl_settings import *
import triangleVPK as VPK
import sys as s
from scipy.linalg import toeplitz as tp
import pdb

def gamma(lag,y,mask,numPts):
	runSum=0.0
	numVals=0
	for i in xrange(numPts-lag):
		runSum+=(mask[i]*y[i,0])*(mask[i+lag]*y[i+lag,0])
		numVals+=mask[i]*mask[i+lag]
	if (numVals>0):
		acvf=runSum/numVals
	else:
		acvf=np.nan
	return acvf

secPerSiderealDay=86164.0905 
intTime=6.019802903
readTime=0.5189485261
numIntLC=270
deltat=(numIntLC*(intTime+readTime))/secPerSiderealDay

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=600
nbins=100
set_plot_params(fontsize=12)

basePath=s.argv[1]
resultFilePath=basePath+'result.dat'
resultFile=open(resultFilePath,'w')

yFilePath=basePath+'y.dat'
yFile=open(yFilePath)
line=yFile.readline()
line.rstrip("\n")
values=line.split()
numPts=int(values[1])

t=np.zeros((numPts,2))
y=np.zeros((numPts,2))
mask=np.zeros(numPts)
x=np.zeros((numPts,2))
v=np.zeros((numPts,2))

for i in range(numPts):
	line=yFile.readline()
	line.rstrip("\n")
	values=line.split()
	t[i,1]=deltat*i
	y[i,0]=float(values[0])
	y[i,1]=float(values[1])
	if (values[1]=='1.3407807929942596e+154'):
		mask[i]=0.0
	else:
		mask[i]=1.0

ySum=0.0
for i in range(numPts):
	ySum+=mask[i]*y[i,0]
for i in range(numPts):
	y[i,0]/=ySum

maxLag=input('Enter maximum lag: ')
ACVF=np.zeros((maxLag,2))
for i in xrange(maxLag):
	print "Computing ACVF for %d"%(i)
	ACVF[i,0]=i
	ACVF[i,1]=gamma(i,y,mask,numPts)

plt.figure(1,figsize=(fwid,fhgt))
plt.plot(ACF[:,0],ACF[:,1])
plt.show()

