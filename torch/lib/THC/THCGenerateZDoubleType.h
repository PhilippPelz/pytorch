#ifndef THC_GENERIC_FILE
#error                                                                         \
    "You must define THC_GENERIC_FILE before including THGenerateZDoubleType.h"
#endif

// #ifndef toZx_defined
// #define toZx_defined
// zx toZx(zcx val) {
//   union {
//     double x[2];
//     zx y;
//   } v;
//   v.x[0] = val.real();
//   v.x[1] = val.imag();
//   return v.y;
// }
// #endif

#define real zcx
#define cureal cufftDoubleComplex
#define cufft cufftExecZ2Z
#define cufftname CUFFT_Z2Z
#define accreal zcx
#define part double
#define Real ZDouble
#define CReal CudaZDouble
#define Part Double
#define CPart CudaDouble
#define SINCOS sincos
#define THC_REAL_IS_ZDOUBLE
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
#undef Part
#undef CPart
#undef SINCOS
#undef THC_REAL_IS_ZDOUBLE

#if !(defined(THCGenerateAllTypes) || defined(THCGenerateComplexTypes))
#undef THC_GENERIC_FILE
#endif
