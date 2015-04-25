#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <array>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <nlopt.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/system/linux_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/detail/quoted_manip.hpp>
#include "Acquire.hpp"
#include "Kalman.hpp"
#include "Universe.hpp"
#include "Kepler.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

//#define DEBUG_MASK

using namespace std;
using namespace nlopt;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: computeCFs" << endl;
	cout << "Purpose: Program to compute the ACF/ACVF and the PACF for light curves" << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	double maxDouble = numeric_limits<double>::max();
	double sqrtMaxDouble = sqrt(maxDouble);

	string basePath;
	vector<string> word(2);

	AcquireDirectory(cout,cin,"Path to working directory: ","Invalid path!\n",basePath);
	basePath += "/";

	string line, yFilePath = basePath + "y.dat";
	cout << "Input LC: " << yFilePath << endl;
	ifstream yFile;
	yFile.open(yFilePath);

	getline(yFile,line);
	istringstream record1(line);
	for (int i = 0; i < 2; ++i) {
		record1 >> word[i];
		}
	int numCadences = stoi(word[1]);
	cout << "numCadences: " << numCadences << endl;

	getline(yFile,line);
	istringstream record2(line);
	for (int i = 0; i < 2; ++i) {
		record2 >> word[i];
		}
	int numObservations = stoi(word[1]);
	cout << "numObservations: " << numObservations << endl;

	getline(yFile,line);
	istringstream record3(line);
	for (int i = 0; i < 2; ++i) {
		record3 >> word[i];
		}
	double meanFlux = stod(word[1]);
	cout << "meanFlux: " << meanFlux << endl;

	int* cadence = static_cast<int*>(_mm_malloc(numCadences*sizeof(double),64));
	double* mask = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	vector<string> wordNew(4);
	string lineNew;
	istringstream recordNew;
	cout.precision(16);
	int i = 0;
	while (!yFile.eof()) {
		getline(yFile,lineNew); 
		istringstream record(lineNew);
		for (int j = 0; j < 4; ++j) {
			record >> wordNew[j];
			}
		cadence[i] = stoi(wordNew[0]);
		mask[i] = stod(wordNew[1]);
		y[i] = stod(wordNew[2]);
		yerr[i] = stod(wordNew[3]);
		i += 1;
		} 

	cout << "Computing ACVF" << endl;
	int numLags = numCadences - 1;

	double* acvf = static_cast<double*>(_mm_malloc(numLags*sizeof(double),64));

	#ifdef TIME_ACVF
	double timeACVFBegin = 0.0, timeACVFEnd = 0.0, timeACVF = 0.0;
	timeACVFBegin = dtime();
	#endif

	ACVF(numCadences, numLags, y, mask, acvf);

	#ifdef TIME_ACVF
	timeACVFEnd = dtime();
	timeACVF = timeACVFEnd - timeACVFBegin;
	cout << "ACVF computed in " << timeACVF << " (s)!" << endl;
	#endif

	cout << "Writing ACVF to ";
	ACVFPath = basePath + "acvf.dat";
	cout << ACVFPath << endl;
	ofstream ACVFFile;
	ACVFFile.open(ACVFPath);
	ACVFFile.precision(16);
	ACVFFile << noshowpos << fixed << "numCadences: " << numCadences << endl;
	ACVFFile << noshowpos << fixed << "numObservations: " << numObservations << endl;
	ACVFFile << noshowpos << fixed << "numLags: " << numLags << endl;
	for (int lagNum = 0; lagNum < numLags - 1; ++lagNum) {
		ACVFFile << noshowpos << scientific << acvf[lagNum] << endl;
		}
	ACVFFile << noshowpos << scientific << acvf[numLags - 1];
	ACVFFile.close();
	cout << "ACVF written!" << endl;

	cout << "Computing ACF" << endl;

	double* acf = static_cast<double*>(_mm_malloc(numLags*sizeof(double),64));

	#ifdef TIME_AVF
	double timeACFBegin = 0.0, timeACFEnd = 0.0, timeACF = 0.0;
	timeACFBegin = dtime();
	#endif

	ACF(numLags, acvf, acf);

	#ifdef TIME_ACF
	timeACFEnd = dtime();
	timeACF = timeACFEnd - timeACFBegin;
	cout << "ACF computed in " << timeACF << " (s)!" << endl;
	#endif

	cout << "Writing ACF to ";
	ACVFPath = basePath + "acf.dat";
	cout << ACFPath << endl;
	ofstream ACFFile;
	ACFFile.open(ACFPath);
	ACFFile.precision(16);
	ACFFile << noshowpos << fixed << "numCadences: " << numCadences << endl;
	ACFFile << noshowpos << fixed << "numObservations: " << numObservations << endl;
	ACFFile << noshowpos << fixed << "numLags: " << numLags << endl;
	for (int lagNum = 0; lagNum < numLags - 1; ++lagNum) {
		ACFFile << noshowpos << scientific << acf[lagNum];
		}
	ACFFile << noshowpos << scientific << acf[numLags - 1];
	ACFFile.close();
	cout << "ACF written!" << endl;

	cout << "Computing SF1" << endl;

	double* sf1 = static_cast<double*>(_mm_malloc(numLags*sizeof(double),64));

	#ifdef TIME_SF1
	double timeSF1Begin = 0.0, timeSF1End = 0.0, timeSF1 = 0.0;
	timeSF1Begin = dtime();
	#endif

	SF1(numLags, acvf, sf1);

	#ifdef TIME_SF1
	timeSF1End = dtime();
	timeSF1 = timeSF1End - timeSF1Begin;
	cout << "SF1 computed in " << timeSf1 << " (s)!" << endl;
	#endif

	cout << "Writing SF1 to ";
	ACVFPath = basePath + "sf1.dat";
	cout << SF1Path << endl;
	ofstream SF1File;
	SF1File.open(SF1Path);
	SF1File.precision(16);
	SF1File << noshowpos << fixed << "numCadences: " << numCadences << endl;
	SF1File << noshowpos << fixed << "numObservations: " << numObservations << endl;
	SF1File << noshowpos << fixed << "numLags: " << numLags << endl;
	for (int lagNum = 0; lagNum < numLags - 1; ++lagNum) {
		SF1File << noshowpos << scientific << sf1[lagNum];
		}
	SF1File << noshowpos << scientific << sf1[numLags - 1];
	SF1File.close();
	cout << "SF1 written!" << endl;

	cout << "Program exiting...Have a nice day!" << endl;

	_mm_free(sf1);
	_mm_free(acf);
	_mm_free(acvf);
	_mm_free(cadence);
	_mm_free(mask);
	_mm_free(y);
	_mm_free(yerr);
	}
