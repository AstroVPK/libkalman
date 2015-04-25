#include <malloc.h>
#include <sys/time.h>
#include <limits>
#include <mathimf.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_types.h>
#include <iostream>
#include <vector>
#include "Constants.hpp"
#include "Correlation.hpp"
#include <stdio.h>

//#define TIMEALL
//#define TIMEPER
//#define TIMEFINE
//#define DEBUG
//#define DEBUG_LNLIKE
//#define WRITE
//#define DEBUG_FUNC
//#define DEBUG_SETDLM
//#define DEBUG_SETDLM_DEEP
//#define DEBUG_CHECKARMAPARAMS
//#define DEBUG_BURNSYSTEM
//#define DEBUG_CTORDLM
//#define DEBUG_DTORDLM
//#define DEBUG_ALLOCATEDLM
//#define DEBUG_DEALLOCATEDLM
//#define DEBUG_DEALLOCATEDLM_DEEP
//#define DEBUG_RESETSTATE
//#define DEBUG_CALCLNLIKE

#ifdef WRITE
#include <fstream>
#endif

using namespace std;

void viewMatrix(int nRows, int nCols, double* mat) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			cout << mat[j*nCols + i] << " ";
			}
		cout << endl;
		}
	}

double dtime() {
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,(struct timezone*)0);
	tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
	return( tseconds );
	}

void ACVF(int numCadences, int numLags, const double* const y, const double* const mask, double* acvf) {
	/*! First remove the mean. */
	double sum = 0.0, numObs = 0.0;
	for (int cadCounter = 0; cadCounter < numCadences; ++cadCounter) {
		sum += mask[cadCounter]*y[cadCounter];
		numObs += mask[cadCounter];
		}
	double mean = sum/numObs;

	double* yScratch = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	#pragma omp parallel for simd default(none) shared(numCadences, yScratch, y, mean, mask)
	for (int cadCounter = 0; cadCounter < numCadences; ++cadCounter) {
		yScratch[cadCounter] = y[cadCounter] - mean*mask[cadCounter];
		}

	/*! Now compute for each lag. */
	#pragma omp parallel for default(none) shared(numLags, acf, numCadences, mask, yScratch, numObs) private(cadCounter)
	for (lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acf[lagCounter] = 0.0;
		for (cadCounter = 0; cadCounter < numCadences - lagCounter; ++cadCounter) {
			acf[lagCounter] += mask[cadCounter]*mask[cadCounter + lagCounter]*yScratch[cadCounter]*yScratch[cadCounter + lagCounter]; 
			}
		acf[lagCounter] /= numObs;
		}

	_mm_free(yScratch;)
	}

void ACF(int numLags, const double* const acvf, double* acf) {
	#pragma omp parallel for simd default(none) shared(numLags, acf, acvf)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acf[lagCounter] = acvf[lagCounter]/acvf[0];
		}
	}

void SF1(int numLags, const double* const acvf, double* sf1) {
	#pragma omp parallel for simd default(none) shared(numLags, sf1, acvf)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		sf1[lagCounter] = 2.0*(acvf[0] - acvf[lagCounter]);
		}
	}

void ACF_SF1(int numLags, const double* const acvf, double* acf, double* sf1) {
	#pragma omp parallel for simd default(none) shared(numLags, acf, af1, acvf)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acf[lagCounter] = acvf[lagCounter]/acvf[0];
		sf1[lagCounter] = 2.0*(acvf[0] - acvf[lagCounter]);
		}
	}