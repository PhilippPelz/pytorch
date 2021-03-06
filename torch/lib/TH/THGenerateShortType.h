#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define real int16_t
#define part short
#define ureal uint16_t
#define accreal int64_t
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real Short
#define Part Short
#define THInf SHRT_MAX
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef ureal
#undef accreal
#undef Real
#undef Part
#undef part
#undef THInf
#undef TH_REAL_IS_SHORT
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
