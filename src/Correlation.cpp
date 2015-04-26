#include <malloc.h>
#include <limits>
#include <mathimf.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_types.h>
#include "Correlation.hpp"
#include <stdio.h>

using namespace std;

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

	/*! Now, following "Spectrum Estimation with Missing Observations" by Richard H. Jones (Rcvd 1969, Rvsd 1971) we compute for each lag. */
	/*#pragma omp parallel for default(none) shared(numLags, acvf, numCadences, mask, yScratch, numObs)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acvf[lagCounter] = 0.0;
		double numPts = 0.0;
		for (int cadCounter = 0; cadCounter < numCadences - lagCounter; ++cadCounter) {
			acvf[lagCounter] += mask[cadCounter]*mask[cadCounter + lagCounter]*yScratch[cadCounter]*yScratch[cadCounter + lagCounter]; 
			numPts += mask[cadCounter]*mask[cadCounter + lagCounter];
			}
		acvf[lagCounter] /= numPts;
		}*/

	/*! Following "Modern Applied Statistics with S" by W.N. Venables & B.D. Ripley (4th Ed, 1992) we compute for each lag. */
	#pragma omp parallel for default(none) shared(numLags, acvf, numCadences, mask, yScratch, numObs)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acvf[lagCounter] = 0.0;
		for (int cadCounter = 0; cadCounter < numCadences - lagCounter; ++cadCounter) {
			acvf[lagCounter] += mask[cadCounter]*mask[cadCounter + lagCounter]*yScratch[cadCounter]*yScratch[cadCounter + lagCounter]; 
			}
		acvf[lagCounter] /= numObs;
		}

	_mm_free(yScratch);
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
	#pragma omp parallel for simd default(none) shared(numLags, acf, sf1, acvf)
	for (int lagCounter = 0; lagCounter < numLags; ++lagCounter) {
		acf[lagCounter] = acvf[lagCounter]/acvf[0];
		sf1[lagCounter] = 2.0*(acvf[0] - acvf[lagCounter]);
		}
	}