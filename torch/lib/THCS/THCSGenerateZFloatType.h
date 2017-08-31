#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THCSGenerateZFloatType.h"
#endif

#define real ccx
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal ccx
#define part float
#define Real ZFloat
#define CReal CudaZFloat
#define THCS_REAL_IS_ZFLOAT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef part
#undef Real
#undef CReal
#undef THCS_REAL_IS_ZFLOAT

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateComplexTypes
#undef THCS_GENERIC_FILE
#endif
#endif
