#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateZFloatType.h"
#endif

#include <complex>
#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

#define real ccx
#define cureal cufftComplex
#define cufft cufftExecC2C
#define cufftname CUFFT_C2C
#define accreal ccx
#define part float
#define Real ZFloat
#define CReal CudaZFloat
#define THC_REAL_IS_ZFLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef cureal
#undef cufft
#undef cufftname
#undef accreal
#undef part
#undef Real
#undef CReal
#undef THC_REAL_IS_ZFLOAT

#if !(defined(THCGenerateAllTypes) || defined(THCGenerateComplexTypes))
#undef THC_GENERIC_FILE
#endif
