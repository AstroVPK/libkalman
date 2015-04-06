import numpy as np
import math as m
import random as r
import KalmanFast as KF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from mpl_settings import *
import scipy.linalg as la
import numpy.linalg as npla 
import triangleVPK as VPK
import sys as s
import pdb

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=600
nbins=100
set_plot_params(fontsize=12)

secPerSiderealDay = 86164.0905 
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
deltat=(numIntLC*(intTime+readTime))/secPerSiderealDay

def NumVals(y,mask,numPts):
	count=0.0
	for i in xrange(numPts):
		count+=mask[i]
	return count

def ACVF(lag,y,mask,numPts,numVals):
	runSum=0.0
	for i in xrange(numPts-lag):
		runSum+=(mask[i]*y[i,0])*(mask[i+lag]*y[i+lag,0])
	acvf=runSum/numVals
	return acvf

def PACF(lag,ACVFVals):
	gammaMatrix=np.matrix(la.toeplitz(ACVFVals[0:lag,1]))
	gammaVec=np.transpose(np.matrix(ACVFVals[1:lag+1,1]))
	solution=np.matrix(np.dot(np.matrix(npla.pinv(gammaMatrix)),gammaVec))
	result=float(solution[lag-1,0])
	del gammaMatrix
	del gammaVec
	del solution
	return result

basePath=s.argv[1]
maxLag=int(s.argv[2])

yFilePath=basePath+'y.dat'
yFile=open(yFilePath)
line=yFile.readline()
line.rstrip("\n")
values=line.split()
numPts=int(values[1])
line=yFile.readline()
line.rstrip("\n")
values=line.split()
numObs=int(values[1])
line=yFile.readline()
line.rstrip("\n")
values=line.split()
meanY=float(values[1])

t=np.zeros((numPts,2))
y=np.zeros((numPts,2))
mask=np.zeros(numPts)
x=np.zeros((numPts,2))
v=np.zeros((numPts,2))

for i in range(numPts):
	line=yFile.readline()
	line.rstrip("\n")
	values=line.split()
	t[i,0]=int(values[0])
	t[i,1]=deltat*i
	mask[i]=float(values[1])
	y[i,0]=float(values[2])
	y[i,1]=float(values[3])

#ySum=0.0
#maskSum=0.0
#for i in range(numPts):
#	ySum+=mask[i]*y[i,0]
#	maskSum+=mask[i]
#ySum/=maskSum
#print ySum
#for i in range(numPts):
#	y[i,0]/=ySum

numVals=NumVals(y,mask,numPts)
ACVFVals=np.zeros((maxLag,2))
PACFVals=np.zeros((maxLag,2))
for i in xrange(maxLag):
	ACVFVals[i,0]=i
	PACFVals[i,0]=i
	ACVFVals[i,1]=ACVF(i,y,mask,numPts,numVals)
PACFVals[0,1]=1.0
for i in xrange(1,maxLag):
	PACFVals[i,1]=PACF(i,ACVFVals)

plt.figure(1,figsize=(fwid,fhgt))

plt.subplot(311)
yMax=np.max(y[np.nonzero(y[:,0]),0])
yMin=np.min(y[np.nonzero(y[:,0]),0])
plt.ylabel('$F$ (arb units)')
plt.xlabel('$t$ (d)')
for i in range(numPts):
	if (mask[i]==1.0):
		plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(yMin,yMax)
plt.subplot(312)
plt.vlines(x=0,ymin=min(0,ACVFVals[0,1]/ACVFVals[0,1]),ymax=max(0,ACVFVals[0,1]/ACVFVals[0,1]),colors='k')
plt.ylabel('$ACF$')
plt.xlabel('Lag')
plt.subplot(313)
plt.ylabel('$PACF$')
plt.xlabel('Lag')
plt.vlines(x=0,ymin=min(0,PACFVals[0,1]),ymax=max(0,PACFVals[0,1]),colors='k')
for i in xrange(1,maxLag-1):
	plt.subplot(312)
	plt.vlines(x=i,ymin=min(0,ACVFVals[i,1]/ACVFVals[0,1]),ymax=max(0,ACVFVals[i,1]/ACVFVals[0,1]),colors='k')
	plt.subplot(313)
	plt.vlines(x=i,ymin=min(0,PACFVals[i,1]),ymax=max(0,PACFVals[i,1]),colors='k')
plt.subplot(312)
plt.hlines(y=[1.96/m.sqrt(numObs),-1.96/m.sqrt(numObs)],xmin=0,xmax=maxLag-1,linewidth=1, color='r',linestyle='dashed')
plt.hlines(y=0,xmin=0,xmax=maxLag-1,linewidth=2, color='k')
plt.xlim=(0,maxLag-1)
plt.subplot(313)
plt.hlines(y=[1.96/m.sqrt(numObs),-1.96/m.sqrt(numObs)],xmin=0,xmax=maxLag-1,linewidth=1, color='r',linestyle='dashed')
plt.hlines(y=0,xmin=0,xmax=maxLag-1,linewidth=2, color='k')
plt.xlim=(0,maxLag-1)

plt.tight_layout()
plt.savefig(basePath+"cfs.jpg",dpi=dotsPerInch)
plt.clf()