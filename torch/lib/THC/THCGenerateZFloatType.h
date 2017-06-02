#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#include <complex>
#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

#define real ccx
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal ccx
#define part float
#define Real ZFloat
#define CReal CudaZFloat
#define THC_REAL_IS_ZFLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef part
#undef Real
#undef CReal
#undef THC_REAL_IS_ZFLOAT

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
