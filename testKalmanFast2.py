import KalmanFast as KF
import math as m
import numpy as np
import random as r
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_settings import *
import scipy.optimize as opt
import emcee
import triangle
import time
import pdb

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=300
nbins=100
set_plot_params(fontsize=24)
TalkPath="/home/exarkun/Documents/AASWinter2015/Talk/"
KalmanPath="/home/exarkun/Documents/Research/KalmanEstimation/"
KeplerPath="/home/exarkun/Documents/Research/Kepler/"
startCadence=21145
#startCadence=24600

burnSeed=1311890535
distSeed=2603023340
noiseSeed=2410288857

outPath=KalmanPath
pMaster=[1.25,-0.3]
qMaster=[0.75]#,0.95]
pNum=len(pMaster)
qNum=len(qMaster)
distMaster=2.5
noiseMaster=0.25
numBurn=10000
numPts=1000

ndim=pNum+qNum+1
nwalkers=100
nsteps=500
chop=150

def ARMALnLike(Theta,y,p,q,n,F,I,D,Q,H,R,K):
	pTrial=Theta[0:p].tolist()
	qTrial=Theta[p:p+q].tolist()
	distTrial=Theta[-1]
	if (KF.checkParams(pList=pTrial,qList=qTrial,dist=distTrial)==0.0):
		(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pTrial,qTrial,distTrial,F,I,D,Q)
		logLike=-np.inf
	else:
		(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pTrial,qTrial,distTrial,F,I,D,Q)
		logLike=KF.getLnLike(y,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
	return logLike

def ARMALnLikeMissing(Theta,y,mask,p,q,n,F,I,D,Q,H,R,K):
	pTrial=Theta[0:p].tolist()
	qTrial=Theta[p:p+q].tolist()
	distTrial=Theta[-1]
	if (KF.checkParams(pList=pTrial,qList=qTrial,dist=distTrial)==0.0):
		(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pTrial,qTrial,distTrial,F,I,D,Q)
		logLike=-np.inf
	else:
		(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pTrial,qTrial,distTrial,F,I,D,Q)
		logLike=KF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
	return logLike

def ARMANegLnLike(Theta,y,p,q,n,F,I,D,Q,H,R,K):
	return -ARMALnLike(Theta,y,p,q,n,F,I,D,Q,H,R,K)

def ARMANegLnLikeMissing(Theta,y,mask,p,q,n,F,I,D,Q,H,R,K):
	return -ARMALnLikeMissing(Theta,y,mask,p,q,n,F,I,D,Q,H,R,K)

if KF.checkParams(pList=pMaster,qList=qMaster,dist=distMaster):

	'''burnFilePath="/home/exarkun/Desktop/burn.dat"
	distFilePath="/home/exarkun/Desktop/dist.dat"
	noiseFilePath="/home/exarkun/Desktop/noise.dat"
	burnRand=np.loadtxt(burnFilePath)
	distRand=np.loadtxt(distFilePath)
	noiseRand=np.loadtxt(noiseFilePath)'''

	t=np.zeros((numPts,2))
	y=np.zeros((numPts,2))
	mask=np.zeros(numPts)
	x=np.zeros((numPts,2))
	v=np.zeros((numPts,2))

	(n,p,q,F,I,D,Q,H,R,K)=KF.makeSystem(pNum,qNum)
	(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pMaster,qMaster,distMaster,F,I,D,Q)
	X=KF.burnSystem(X,F,D,distMaster,numBurn,burnSeed)

	KeplerObj="kplr009650715"
	KeplerFilePath=KeplerPath+KeplerObj+"/"+KeplerObj+"-calibrated.dat"
	data=np.loadtxt(KeplerFilePath,skiprows=2)
	numCadences=data.shape[0]
	startIndex=np.where(data[:,0]==startCadence)[0][0]
	deltat=data[1,2]-data[0,2]
	counter=0
	for i in range(startIndex,startIndex+numPts):
		t[counter,0]=data[i,0]
		t[counter,1]=deltat*counter
		if (data[i,2]!=0.0):
			mask[counter]=1.0
		#mask[counter]=1.0
		counter+=1
	(X,y)=KF.obsSystemMissing(X,F,D,H,distMaster,noiseMaster,y,mask,numPts,distSeed,noiseSeed)
	#(X,y)=KF.obsSystemFixed(X,F,D,H,distMaster,noiseMaster,y,numPts,distRand,noiseRand)


	'''yMean=np.nanmean(y[:,0])
	for i in range(numPts):
		y[i,0]-=yMean'''

	plt.figure(1,figsize=(fwid,fhgt))
	yMax=np.max(y[np.nonzero(y[:,0]),0])
	yMin=np.min(y[np.nonzero(y[:,0]),0])
	plt.ylabel('$F$ (arb units)')
	plt.xlabel('$t$ (d)')
	for i in range(numPts):
		if (mask[i]==1.0):
			plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	#plt.plot(t[:,1],y[:,0],c='#e66101',marker='o',markeredgecolor='none',zorder=5)
	plt.xlim(t[0,1],t[-1,1])
	plt.ylim(yMin,yMax)
	plt.tight_layout()
	plt.draw()
	plt.savefig(outPath+"ARMA(%d,%d)_Master.jpg"%(pNum,qNum),dpi=dotsPerInch)
	#plt.show()

	(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pMaster,qMaster,distMaster,F,I,D,Q)
	#print "Master LnLike: %f"%(KF.getLnLikeMissing(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K))
	print "Master LnLike: %f"%(KF.getLnLike(y,mask,X,P,XMinus,PMinus,F,I,D,Q,H,R,K))

	thetaInit = list()
	for i in range(pNum+qNum):
		thetaInit.append(r.gauss(0.0,1e-3))
	thetaInit.append(1.0)
	#result=opt.fmin_powell(ARMANegLnLike,x0=thetaInit,args=(y,mask,p,q,n,F,I,D,Q,H,R,K),ftol=0.001,disp=1)
	result=opt.fmin_powell(ARMANegLnLikeMissing,x0=thetaInit,args=(y,mask,p,q,n,F,I,D,Q,H,R,K),ftol=0.001,disp=1)

	pInferred=list()
	qInferred=list()
	distInferred=0.0
	for i in range(0,p):
		pInferred.append(result[i])
		print "phi[%d]: %f"%(i,result[i])
	for i in range(p,p+q):
		qInferred.append(result[i])
		print "theta[%d]: %f"%(i-p,result[i])
	distInferred=result[-1]
	print "sigmaDist: %f"%(result[-1])

	(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pInferred,qInferred,distInferred,F,I,D,Q)
	KF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
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

	plt.figure(2,figsize=(fwid,fhgt))
	plt.subplot(211)
	yMax=np.max(y[np.nonzero(y[:,0]),0])
	yMin=np.min(y[np.nonzero(y[:,0]),0])
	plt.ylabel('$F$ (arb units)')
	plt.xlabel('$t$ (d)')
	for i in range(numPts):
		if (mask[i]==1.0):
			plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=-5)
	plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=0.75,zorder=-10)
	plt.xlim(t[0,1],t[-1,1])
	plt.ylim(yMin,yMax)
	plt.tight_layout()
	plt.draw()
	plt.subplot(212)
	vMax=np.max(v[np.nonzero(v[:,0]),0])
	vMin=np.min(v[np.nonzero(v[:,0]),0])
	plt.ylabel('$\Delta F$ (arb units)')
	plt.xlabel('$t$ (d)')
	for i in range(numPts):
		if (mask[i]==1.0):
			plt.errorbar(t[i,1],v[i,0],yerr=v[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	#plt.errorbar(t[:,1],v[:,0],yerr=v[:,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
	plt.barh(binEdges[0:nBins],newBinVals[0:nBins],xerr=newBinErrors[0:nBins],height=binWidth,alpha=0.4,zorder=10)
	plt.xlim(t[0,1],t[-1,1])
	plt.ylim(vMin,vMax)
	plt.tight_layout()
	plt.draw()
	plt.savefig(outPath+"ARMA(%d,%d)_PowellFit.jpg"%(pNum,qNum),dpi=dotsPerInch)
	#plt.show()

	YesNo=input("Happy? (1/0): ")
	#YesNo=1

	if (YesNo):
		#try:
		pos=[np.array(result)+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		#except ValueError:
		#	pdb.set_trace()
		sampler=emcee.EnsembleSampler(nwalkers,ndim,ARMALnLikeMissing,a=2.0,args=(y,mask,p,q,n,F,I,D,Q,H,R,K),threads=2)
		beginT=time.time()
		sampler.run_mcmc(pos,nsteps)
		endT=time.time()
		print "That took %f sec"%(endT-beginT)
		plt.figure(3,figsize=(fwid,fhgt))
		for j in range(ndim):
			plt.subplot(ndim,1,j)
			for i in range(nwalkers):
				plt.plot(sampler.chain[i,:,j],c='#000000',alpha=0.5)
			plt.xlabel('$step$')
			if (j<pNum):
				plt.ylabel("$\phi_{%d}$"%(j+1))
			elif ((j>=pNum) and (j<pNum+qNum)):
				plt.ylabel("$\\theta_{%d}$"%(j-pNum+1))
			else:
				plt.ylabel("$\sigma_{w}^{2}$")
			plt.tight_layout()
			plt.draw()
		#plt.show()
		plt.savefig(outPath+"ARMA(%d,%d)_Walkers.jpg"%(pNum,qNum),dpi=dotsPerInch)
		WalkersFilePath=outPath+'ARMA(%d,%d)_Walkers.dat'%(pNum,qNum)
		WalkersFile=open(WalkersFilePath,'w')
		WalkersFile.write("nsteps: %d\n"%(nsteps))
		WalkersFile.write("nwalkers: %d\n"%(nwalkers))
		WalkersFile.write("ndim: %d\n"%(ndim))
		pMasterNum=len(pMaster)
		qMasterNum=len(qMaster)
		WalkersFile.write("pMasterNum: %d\n"%(pMasterNum))
		WalkersFile.write("qMasterNum: %d\n"%(qMasterNum))
		for i in range(pMasterNum):
			WalkersFile.write("p_%d: %f\n"%(i+1,pMaster[i]))
		for i in range(qMasterNum):
			WalkersFile.write("q_%d: %f\n"%(i+1,qMaster[i]))
		WalkersFile.write("sigma_w^2: %f\n"%(distMaster))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=''
				for k in range(ndim):
					line+='%f '%(sampler.chain[j,i,k])
				line+='\n'
				WalkersFile.write(line)
		WalkersFile.close()
		samples=sampler.chain[:,chop:,:].reshape((-1,ndim))

		lbls=[]
		for i in range(pNum):
			lbls.append("$\phi_{%d}$"%(i+1))
		for i in range(qNum):
			lbls.append("$\\theta_{%d}$"%(i+1))
		lbls.append("$\sigma_{w}^{2}$")	
		fig=triangle.corner(samples,labels=lbls,truths=pMaster+qMaster+[distMaster],truth_color='#000000',show_titles=True,title_args={"fontsize":12},quantiles=[0.16, 0.5, 0.84],plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth,bins=nbins)
		fig.savefig(outPath+"ARMA(%d,%d)_MCMC.jpg"%(pNum,qNum),dpi=dotsPerInch)
		#plt.show()

		pMCMC=list()
		qMCMC=list()
		distMCMC=0.0
		for i in range(0,p):
			pMCMC.append(np.percentile(samples[:,i],50.0))
			print "phi[%d]: %f"%(i,pMCMC[i])
		for i in range(p,p+q):
			qMCMC.append(np.percentile(samples[:,i],50.0))
			print "theta[%d]: %f"%(i-p,qMCMC[i-p])
		distMCMC=np.percentile(samples[:,-1],50.0)
		print "sigmaDist: %f"%(distMCMC)

		(X,P,XMinus,PMinus,F,I,D,Q)=KF.setSystem(p,q,n,pMCMC,qMCMC,distMCMC,F,I,D,Q)
		KF.fixedIntervalSmoother(y,v,x,X,P,XMinus,PMinus,F,I,D,Q,H,R,K)
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

		plt.figure(5,figsize=(fwid,fhgt))
		plt.subplot(211)
		yMax=np.max(y[np.nonzero(y[:,0]),0])
		yMin=np.min(y[np.nonzero(y[:,0]),0])
		plt.ylabel('$F$ (arb units)')
		plt.xlabel('$t$ (d)')
		for i in range(numPts):
			if (mask[i]==1.0):
				plt.errorbar(t[i,1],y[i,0],yerr=y[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		#plt.errorbar(t[:,1],y[:,0],yerr=y[:,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		plt.plot(t[:,1],x[:,0],c='#5e3c99',zorder=-5)
		plt.fill_between(t[:,1],x[:,0]+x[:,1],x[:,0]-x[:,1],facecolor='#b2abd2',edgecolor='none',alpha=0.75,zorder=-10)
		plt.xlim(t[0,1],t[-1,1])
		plt.ylim(yMin,yMax)
		plt.tight_layout()
		plt.draw()
		plt.subplot(212)
		vMax=np.max(v[np.nonzero(v[:,0]),0])
		vMin=np.min(v[np.nonzero(v[:,0]),0])
		plt.ylabel('$\Delta F$ (arb units)')
		plt.xlabel('$t$ (d)')
		for i in range(numPts):
			if (mask[i]==1.0):
				plt.errorbar(t[i,1],v[i,0],yerr=v[i,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		#plt.errorbar(t[:,1],v[:,0],yerr=v[:,1],c='#e66101',fmt='.',marker=".",capsize=0,zorder=5)
		plt.barh(binEdges[0:nBins],newBinVals[0:nBins],xerr=newBinErrors[0:nBins],height=binWidth,alpha=0.4,zorder=10)
		plt.xlim(t[0,1],t[-1,1])
		plt.ylim(vMin,vMax)
		plt.tight_layout()
		plt.draw()
		plt.savefig(outPath+"ARMA(%d,%d)_MCMCFit.jpg"%(pNum,qNum),dpi=dotsPerInch)
		#plt.show()

else:
	print "Bad Initial Params!"
