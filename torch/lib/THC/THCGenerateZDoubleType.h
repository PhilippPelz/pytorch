#ifndef THC_GENERIC_FILE
#error                                                                         \
    "You must define THC_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#include <complex>
#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

#define real zcx
#define accreal zcx
#define part double
#define Real ZDouble
#define CReal CudaZDouble
#define THC_REAL_IS_ZDOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef part
#undef Real
#undef CReal
#undef THC_REAL_IS_ZDOUBLE

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
