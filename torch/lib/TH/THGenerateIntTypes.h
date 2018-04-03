#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THIntLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#define CREAL
#define CABS fabs

#include "THGenerateByteType.h"
#include "THGenerateCharType.h"
#include "THGenerateIntType.h"
#include "THGenerateLongType.h"
#include "THGenerateShortType.h"

#undef CABS
#undef CREAL

#ifdef THIntLocalGenerateManyTypes
#undef THIntLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
