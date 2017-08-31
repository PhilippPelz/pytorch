#include "THCTensorFFT.h"

inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	FILE *f;
	f = fopen("/home/philipp/cufftSafeCall.log", "a+");
	if (CUFFT_SUCCESS != err) {
		fprintf(f,"CUFFT error in file '%s', line %d\n %s\nerror: %d\nterminating!\n",
		file, line, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
		// assert(0);
	}
	fclose(f);
}

#include "generic/THCTensorFFT.cu"
#include "THCGenerateComplexTypes.h"
