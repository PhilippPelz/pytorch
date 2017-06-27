#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define real char
#define part char
#define accreal long
#define Real Char
#define CReal CudaChar
#define Part Char
#define CPart CudaChar
#define THC_REAL_IS_CHAR
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef part
#undef accreal
#undef Real
#undef CReal
#undef Part
#undef CPart
#undef THC_REAL_IS_CHAR

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
