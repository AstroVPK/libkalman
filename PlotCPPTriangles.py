import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_settings import *
import triangle
import sys as s
import pdb

s1=2
s2=9
fwid=13
fhgt=13
dotsPerInch=300
nbins=100
set_plot_params(fontsize=12)
#path="/home/exarkun/Documents/Research/KalmanEstimation/"
#desktopPath="/home/exarkun/Desktop/"

pMax=int(s.argv[1])
chop=int(s.argv[2])

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
		walkers=np.zeros((nwalkers,nsteps,ndim))
		deviances=np.zeros((nwalkers,nsteps))
		for i in range(nsteps):
			for j in range(nwalkers):
				line=TriFile.readline()
				line.rstrip("\n")
				values=line.split()
				for k in range(ndim):
					walkers[j,i,k]=float(values[k+4])
				deviances[j,i]=-2.0*float(values[-1])
		TriFile.close()

		plt.figure(1,figsize=(fwid,fhgt))
		for j in range(ndim):
			plt.subplot(ndim,1,j+1)
			for i in range(nwalkers):
				plt.plot(walkers[i,:,j],c='#000000',alpha=0.5)
			plt.xlabel('$step$')
			if (0<j<pNum+1):
				plt.ylabel("$\phi_{%d}$"%(j))
			elif ((j>=pNum+1) and (j<pNum+qNum+1)):
				plt.ylabel("$\\theta_{%d}$"%(j-pNum))
			else:
				plt.ylabel("$\sigma_{w}$")
			plt.tight_layout()
			plt.draw()
		plt.savefig(basepath+"mcmcWalkers_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)

		samples=walkers[:,chop:,:].reshape((-1,ndim))
		sampleDeviances = deviances[:,chop:].reshape((-1))
		DIC = 0.5*m.pow(np.std(sampleDeviances),2.0) + np.mean(sampleDeviances)
		lbls=[]
		lbls.append("$\sigma_{w}$")
		for i in range(pNum):
			lbls.append("$\phi_{%d}$"%(i+1))
		for i in range(qNum):
			lbls.append("$\\theta_{%d}$"%(i+1))
		print "DIC: %f"%(DIC)
		fig=triangle.corner(samples,labels=lbls,show_titles=True,title_args={"fontsize": 12},quantiles=[0.16, 0.5, 0.84],plot_contours=True,plot_datapoints=True,plot_contour_lines=False,pcolor_cmap=cm.gist_earth)
		fig.savefig(basePath+"mcmcTriangles_%d_%d.jpg"%(pNum,qNum),dpi=dotsPerInch)
		#plt.show()
	