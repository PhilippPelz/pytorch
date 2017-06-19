#ifndef TH_CUDA_TENSOR_FFT_INC
#define TH_CUDA_TENSOR_FFT_INC

#include "TH.h"
#include "THC.h"
#include "THCTensor.h"
#include "THCGeneral.h"

#include <cufft.h>
#include <cufftXt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
		case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

		case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

		case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

		case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

		case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

		case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

		case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

		case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

		case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

		case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";

    // case ‘CUFFT_INVALID_DEVICE’:
    // return "‘CUFFT_INVALID_DEVICE’";
		//
    // case ‘CUFFT_PARSE_ERROR’:
    // return "‘CUFFT_PARSE_ERROR’";
		//
    // case ‘CUFFT_NO_WORKSPACE’:
    // return "‘CUFFT_NO_WORKSPACE’";
		//
    // case ‘CUFFT_NOT_IMPLEMENTED’:
    // return "‘CUFFT_NOT_IMPLEMENTED’";
		//
    // case ‘CUFFT_LICENSE_ERROR’:
    // return "‘CUFFT_LICENSE_ERROR’";
		//
    // case ‘CUFFT_NOT_SUPPORTED’:
    // return "‘CUFFT_NOT_SUPPORTED’";
	}

	return "<unknown>";
}
#endif
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_INVALID_DEVICE’ not handled in switch [-Wswitch]
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_PARSE_ERROR’ not handled in switch [-Wswitch]
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_NO_WORKSPACE’ not handled in switch [-Wswitch]
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_NOT_IMPLEMENTED’ not handled in switch [-Wswitch]
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_LICENSE_ERROR’ not handled in switch [-Wswitch]
// /mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/projects/pytorch/torch/lib/tmp_install/include/THC/THCTensorFFT.h:18:9: warning: enumeration value ‘CUFFT_NOT_SUPPORTED’ not handled in switch [-Wswitch]

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	FILE *f;
	f = fopen("/home/philipp/cufftSafeCall.log", "a+");
	if (CUFFT_SUCCESS != err) {
		fprintf(f,"CUFFT error in file '%s', line %d\n %s\nerror: %d\nterminating!\n",
		file, line, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
		// assert(0);
	}
	fclose(f);
}

#include "generic/THCTensorFFT.h"
#include "THCGenerateComplexTypes.h"

#endif
