#ifndef THCS_GENERIC_FILE
#error                                                                         \
    "You must define THCS_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define THCSGenerateComplexTypes

#define THCTypeIdxByte 1
#define THCTypeIdxChar 2
#define THCTypeIdxShort 3
#define THCTypeIdxInt 4
#define THCTypeIdxLong 5
#define THCTypeIdxFloat 6
#define THCTypeIdxDouble 7
#define THCTypeIdxHalf 8
#define THCTypeIdxZFloat 9
#define THCTypeIdxZDouble 10
#define THCTypeIdx_(T) TH_CONCAT_2(THCTypeIdx, T)

#include "THCSGenerateZFloatType.h"
#include "THCSGenerateZDoubleType.h"

#undef THCTypeIdxByte
#undef THCTypeIdxChar
#undef THCTypeIdxShort
#undef THCTypeIdxInt
#undef THCTypeIdxLong
#undef THCTypeIdxFloat
#undef THCTypeIdxDouble
#undef THCTypeIdxHalf
#undef THCTypeIdx_
#undef THCTypeIdxZFloat
#undef THCTypeIdxZDouble

#undef THCSGenerateComplexTypes
#undef THCS_GENERIC_FILE
