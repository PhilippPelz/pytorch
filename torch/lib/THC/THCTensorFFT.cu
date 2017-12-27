#include "THCTensorFFT.h"
#include "cuda_runtime.h"
#include <cufft.h>
#include <cufftXt.h>

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

		case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
	}

	return "<unknown>";
}
#endif

inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	if (CUFFT_SUCCESS != err) {
		fprintf(stderr,"CUFFT error in file '%s', line %d\n %d\nerror: %s\nterminating!\n",
				file, line, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
	}
}

#include "generic/THCTensorFFT.cu"
#include "THCGenerateComplexTypes.h"
