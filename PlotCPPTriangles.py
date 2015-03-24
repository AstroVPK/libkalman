import numpy as np
import math as m
import KalmanFast as KF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from mpl_settings import *
import triangleVPK as VPK
import sys as s
import pdb

secPerSiderealDay = 86164.0905 
intTime = 6.019802903
readTime = 0.5189485261
numIntLC = 270
deltat=(numIntLC*(intTime+readTime))/secPerSiderealDay

def MAD(a):
	medianVal=np.median(a)
	b=np.copy(a)
	for i in range(a.shape[0]):
		b[i]=abs(b[i]-medianVal)
	return np.median(b)

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=300
nbins=100
set_plot_params(fontsize=12)

basePath=s.argv[1]
pMax=int(s.argv[2])
chop=int(s.argv[3])
dictDIC=dict()
fiftiethQ=dict()

resultFilePath=basePath+'result.dat'
resultFile=open(resultFilePath,'w')

for pNum in range(1,pMax+1,1):
	for qNum in range(0,pNum,1):
		TriFilePath=basePath+'mcmcOut_%d_%d.dat'%(pNum,qNum)
		TriFile=open(TriFilePath)
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		nsteps=int(values[1])
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		nwalkers=int(values[1])
		line=TriFile.readline()
		line.rstrip("\n")
		values=line.split()
		ndim=int(values[1])
		walkers=np.zeros((nsteps,nwalkers,ndim))
		deviances=np.zeros((nsteps,nwalkers))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=TriFile.readline()
				line.rstrip("\n")
				values=line.split()
				for k in range(ndim):
					walkers[i,j,k]=float(values[k+4])
				deviances[i,j]=-2.0*float(values[-1])
		TriFile.close()

		medianWalker=np.zeros((nsteps,ndim))
		medianDevWalker=np.zeros((nsteps,ndim))
		for i in range(nsteps):
			for k in range(ndim):
				medianWalker[i,k]=np.median(walkers[i,:,k])
				medianDevWalker[i,k]=MAD(walkers[i,:,k])
		stepArr=np.arange(nsteps)

		plt.figure(1,figsize=(fwid,fhgt))
		for k in range(ndim):
			plt.subplot(ndim,1,k+1)
			for j in range(nwalkers):
				plt.plot(walkers[:,j,k],c='#000000',alpha=0.025,zorder=-5)
			plt.fill_between(stepArr[:],medianWalker[:,k]+medianDevWalker[:,k],medianWalker[:,k]-medianDevWalker[:,k],color='#ff0000',edgecolor='#ff0000',alpha=0.5,zorder=5)
			plt.plot(stepArr[:],medianWalker[:,k],c='#dc143c',linewidth=1,zorder=10)
			plt.xlabel('$step$')
			if (0<k<pNum+1):
				plt.ylabel("$\phi_{%d}$"%(k))
			elif ((k>=pNum+1) and (k<pNum+qNum+1)):
				plt.ylabel("$\\theta_{%d}$"%(k-pNum))
			else:
				plt.ylabel("$\sigma_{w}$")
		plt.savefig(basePath+"mcmcWalkers_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		plt.clf()

		samples=walkers[chop:,:,:].reshape((-1,ndim))
		sampleDeviances=deviances[chop:,:].reshape((-1))
		DIC=0.5*m.pow(np.std(sampleDeviances),2.0) + np.mean(sampleDeviances)
		dictDIC["%d %d"%(pNum,qNum)]=DIC
		lbls=list()
		lbls.append("$\sigma_{w}$")
		for i in range(pNum):
			lbls.append("$\phi_{%d}$"%(i+1))
		for i in range(qNum):
			lbls.append("$\\theta_{%d}$"%(i+1))
		figVPK,quantiles,qvalues=VPK.corner(samples,labels=lbls,fig_title="DIC: %f"%(dictDIC["%d %d"%(pNum,qNum)]),show_titles=True,title_args={"fontsize": 12},quantiles=[0.16, 0.5, 0.84],verbose=False,plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth)
		figVPK.savefig(basePath+"mcmcVPKTriangles_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		figVPK.clf()

		line="p: %d; q: %d\n"%(pNum,qNum)
		resultFile.write(line)
		line="DIC: %f\n"%(DIC)
		resultFile.write(line)
		for k in range(ndim):
			if (0<k<pNum+1):
				line="phi_%d\n"%(k)
			elif ((k>=pNum+1) and (k<pNum+qNum+1)):
				line="theta_%d\n"%(k-pNum)
			else:
				line="sigma_w\n"
			resultFile.write(line)
			fiftiethQ["%d %d %s"%(pNum,qNum,line.rstrip("\n"))]=float(qvalues[k][1])
			for i in range(len(quantiles)):
				line="Quantile: %f; Value: %f\n"%(quantiles[i],qvalues[k][i])
				resultFile.write(line)
		line="\n"
		resultFile.write(line)

		del walkers
		del deviances
		del stepArr
		del medianWalker
		del medianDevWalker

sortedDICVals=sorted(dictDIC.items(),key=operator.itemgetter(1))
line="Best models (descending order).\n"
resultFile.write(line)
for DICVal in sortedDICVals:
	line="%s: %s\n"%(DICVal[0],DICVal[1])
	resultFile.write(line);
resultFile.close()

pBest=int(sortedDICVals[0][0].split()[0])
qBest=int(sortedDICVals[0][0].split()[1])

sigmaBest=fiftiethQ["%d %d sigma_w"%(pBest,qBest)]
phiBest=list()
thetaBest=list()
for i in range(pBest):
	phiBest.append(fiftiethQ["%d %d phi_%d"%(pBest,qBest,i+1)])
for i in range(qBest):
	thetaBest.append(fiftiethQ["%d %d theta_%d"%(pBest,qBest,i+1)])

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

(n,p,q,F,I,D,Q,H,R,K)=KF.makeSystem(pBest,qBest)
(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,phiBest,thetaBest,sigmaBest,F,I,D,Q)
LnLike=KF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
#print "LnLike: %f"%(LnLike)
KF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)

plt.figure(2,figsize=(4*fwid,4*fhgt))

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
#plt.tight_layout()

nBins=50
binVals,binEdges=np.histogram(v[~np.isnan(v[:,0]),0],bins=nBins,range=(np.nanmin(v[1:numPts,0]),np.nanmax(v[1:numPts,0])))
binMax=np.nanmax(binVals)
newBinVals=np.zeros(nBins)
newBinErrors=np.zeros(nBins)
tMax=np.nanmax(t[:,1])
for i in range(nBins):
	newBinVals[i]=(tMax/4.0)*(float(binVals[i])/binMax)
	newBinErrors[i]=(tMax/4.0)*(m.sqrt(float(binVals[i]))/binMax)
binWidth=np.asscalar(binEdges[1]-binEdges[0])

plt.subplot(312)
yMax=np.max(y[np.nonzero(y[:,0]),0])
yMin=np.min(y[np.nonzero(y[:,0]),0])
plt.ylabel('$F$ (arb units)')
plt.xlabel('$t$ (d)')
for i in range(numPts):
	if (mask[i]==1.0):
		plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=-5)
plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=10)
plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=0.1,zorder=5)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(yMin,yMax)
#plt.tight_layout()

plt.subplot(313)
vMax=np.max(v[np.nonzero(v[:,0]),0])
vMin=np.min(v[np.nonzero(v[:,0]),0])
plt.ylabel('$\Delta F$ (arb units)')
plt.xlabel('$t$ (d)')
for i in range(numPts):
	if (mask[i]==1.0):
		plt.errorbar(t[i,1],v[i,0],yerr=v[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
plt.barh(binEdges[0:nBins],newBinVals[0:nBins],xerr=newBinErrors[0:nBins],height=binWidth,alpha=0.4,zorder=10)
plt.xlim(t[0,1],t[-1,1])
plt.ylim(vMin,vMax)
#plt.tight_layout()

plt.savefig(basePath+"lc_%d_%d.jpg"%(pBest,qBest),dpi=dotsPerInch)