#ifndef THCS_GENERIC_FILE
#error                                                                         \
    "You must define THCS_GENERIC_FILE before including THCSGenerateZDoubleType.h"
#endif

#define real zcx
#define accreal zcx
#define part double
#define Real ZDouble
#define CReal CudaZDouble
#define THCS_REAL_IS_ZDOUBLE
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef part
#undef Real
#undef CReal
#undef THCS_REAL_IS_ZDOUBLE

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateComplexTypes
#undef THCS_GENERIC_FILE
#endif
#endif
