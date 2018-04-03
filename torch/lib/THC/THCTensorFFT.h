#ifndef TH_CUDA_TENSOR_FFT_INC
#define TH_CUDA_TENSOR_FFT_INC

#include "TH.h"
#include "THC.h"
#include "THCTensor.h"
#include "THCGeneral.h"


#include "device_launch_parameters.h"

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)



#include "generic/THCTensorFFT.h"
#include "THCGenerateComplexTypes.h"

#endif
