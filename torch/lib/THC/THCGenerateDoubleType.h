#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define real double
#define part double
#define accreal double
#define Real Double
#define CReal CudaDouble
#define Part Double
#define CPart CudaDouble
#define THC_REAL_IS_DOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef part
#undef accreal
#undef Real
#undef CReal
#undef Part
#undef CPart
#undef THC_REAL_IS_DOUBLE

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
