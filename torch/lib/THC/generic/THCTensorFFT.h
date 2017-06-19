#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorFFT.h"
#else

#if defined(THC_REAL_IS_ZFLOAT) || defined(THC_REAL_IS_ZDOUBLE)

THC_API void THCTensor_(fftn)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(fft)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(fft2)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(fft3)(THCState *state, THCTensor *self, THCTensor *result);

THC_API void THCTensor_(ifftn)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(ifft)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(ifft2)(THCState *state, THCTensor *self, THCTensor *result);
THC_API void THCTensor_(ifft3)(THCState *state, THCTensor *self, THCTensor *result);

THC_API void THCTensor_(fftnBatched)(THCState *state, THCTensor *self,
                                       THCTensor *result);
THC_API void THCTensor_(ifftnBatched)(THCState *state, THCTensor *self,
                                        THCTensor *result);
#endif

#endif
