#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include "THStorage.h"
#include "THTensorApply.h"

#define THTensor TH_CONCAT_3(TH, Real, Tensor)
#define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME)
// #pragma message "NOW DOING THTensor"
/* basics */
#include "generic/THTensor.h"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.h"
#include "THGenerateHalfType.h"
// #pragma message "NOW DOING THTensorCopy"
#include "generic/THTensorCopy.h"
#include "THGenerateAllTypes.h"

#include "generic/THTensorCopy.h"
#include "THGenerateHalfType.h"
#include "THTensorMacros.h"
// #pragma message "NOW DOING THTensorRandom"
/* random numbers */
#include "THRandom.h"
#include "generic/THTensorRandom.h"
#include "THGenerateAllTypes.h"
// #pragma message "NOW DOING THTensorMath"
/* maths */
#include "generic/THTensorMath.h"
#include "THGenerateAllTypes.h"
// #pragma message "NOW DOING THTensorConv"
/* convolutions */
#include "generic/THTensorConv.h"
#include "THGenerateAllTypes.h"
// #pragma message "NOW DOING THTensorLapack"
/* lapack support */
#include "generic/THTensorLapack.h"
#include "THGenerateFloatTypes.h"
// #pragma message "NOW DOING THTensorLapack complex"
#define TH_GENERIC_FILE "generic/THTensorLapack.h"
// #include "generic/THTensorLapack.h"
#include "THGenerateComplexTypes.h"

#endif
