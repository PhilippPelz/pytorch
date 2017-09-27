#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#if !(defined(THComplexLocalGenerateManyTypes) ||                              \
      defined(THFCLocalGenerateManyTypes))
#define THComplexLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateZDoubleType.h"
#include "THGenerateZFloatType.h"

#ifdef THComplexLocalGenerateManyTypes
#undef THComplexLocalGenerateManyTypes
#undef THGenerateManyTypes
#endif
