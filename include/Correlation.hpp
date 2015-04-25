#ifndef CORRELATION_HPP
#define CORRELATION_HPP

#include <mkl.h>
#include <mkl_types.h>

using namespace std;

void viewMatrix(int nRows, int nCols, double* mat);

double dtime();

void ACVF(int numCadences, int numLags, const double* const y, const double* const mask, double* acvf);

void ACF(int numLags, const double* const acvf, double* acf);

void SF1(int numLags, const double* const acvf, double* sf1);

void ACFandSF1(int numLags, const double* const acvf, double* acf, double* sf1);

#endif
