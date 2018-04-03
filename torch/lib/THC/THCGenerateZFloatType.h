#ifndef THC_GENERIC_FILE
#error                                                                         \
    "You must define THC_GENERIC_FILE before including THGenerateZFloatType.h"
#endif

// #ifndef toCx_defined
// #define toCx_defined
// cx toCx(ccx val) {
//   union {
//     float x[2];
//     cx y;
//   } v;
//   v.x[0] = val.real();
//   v.x[1] = val.imag();
//   return v.y;
// }
// #endif

#define real ccx
#define cureal cufftComplex
#define cufft cufftExecC2C
#define cufftname CUFFT_C2C
#define accreal ccx
#define part float
#define Real ZFloat
#define CReal CudaZFloat
#define Part Float
#define CPart Cuda
#define SINCOS sincosf
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
#undef Part
#undef CPart
#undef SINCOS
#undef THC_REAL_IS_ZFLOAT

#if !(defined(THCGenerateAllTypes) || defined(THCGenerateComplexTypes))
#undef THC_GENERIC_FILE
#endif
