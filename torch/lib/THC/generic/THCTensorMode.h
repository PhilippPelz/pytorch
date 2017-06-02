#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMode.h"
#else
#if !(defined(THC_REAL_IS_ZFLOAT) || defined(THC_REAL_IS_ZDOUBLE))
/* Returns the mode, and index of the mode, for the set of values
 * along a given dimension in the input tensor. */
THC_API void THCTensor_(mode)(THCState *state,
                              THCTensor *values,
                              THCudaLongTensor *indices,
                              THCTensor *input,
                              int dimension);
#endif
#endif // THC_GENERIC_FILE
