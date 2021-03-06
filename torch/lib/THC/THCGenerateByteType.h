#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real uint8_t
#define accreal int64_t
#define part uint8_t
#define Real Byte
#define CReal CudaByte
#define Part Byte
#define CPart CudaByte
#define THC_REAL_IS_BYTE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef part
#undef accreal
#undef Real
#undef CReal
#undef Part
#undef CPart
#undef THC_REAL_IS_BYTE

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
