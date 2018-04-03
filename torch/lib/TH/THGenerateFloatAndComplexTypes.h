#ifndef TH_GENERIC_FILE
#error                                                                         \
    "You must define TH_GENERIC_FILE before including THGenerateFloatAndComplexTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THFCLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateComplexTypes.h"
#include "THGenerateFloatTypes.h"

#ifdef THFCLocalGenerateManyTypes
#undef THFCLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
