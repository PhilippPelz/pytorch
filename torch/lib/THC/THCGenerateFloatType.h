#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define real float
#define part float
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal float
#define Real Float
#define CReal Cuda
#define Part Float
#define CPart Cuda
#define THC_REAL_IS_FLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef part
#undef accreal
#undef Real
#undef CReal
#undef Part
#undef CPart
#undef THC_REAL_IS_FLOAT


#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
