#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatAndComplexTypes.h"
#endif

#define THGenerateFloatAndComplexTypes

#include "THGenerateComplexTypes.h"
#include "THGenerateFloatTypes.h"

#undef THGenerateFloatAndComplexTypes
#undef TH_GENERIC_FILE
