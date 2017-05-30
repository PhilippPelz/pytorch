#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#include <cuComplex.h>

#define real cuFloatComplex
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal cuFloatComplex
#define Real ZFloat
#define CReal ZCuda
#define THC_REAL_IS_FLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_FLOAT

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
