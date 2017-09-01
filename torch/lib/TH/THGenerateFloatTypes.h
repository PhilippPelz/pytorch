#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

//#include <complex.h>
//#undef I

#define TH_REAL_IS_REAL // well this is odd...

#if !(defined(THGenerateManyTypes) || defined(THFCLocalGenerateManyTypes))
#define THFloatLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#define SWAPS sswap_
#define SCAL sscal_
#define COPY scopy_
#define AXPY saxpy_
#define DOT sdot_
#define GEMV sgemv_
#define GER sger_
#define GEMM sgemm_

#define GESV sgesv_
#define TRTRS strtrs_
#define GELS sgels_
#define SYEV ssyev_
#define GEEV sgeev_
#define GESVD sgesvd_
#define GETRF sgetrf_
#define GETRS sgetrs_
#define GETRI sgetri_
#define POTRF spotrf_
#define POTRI spotri_
#define POTRS spotrs_
#define GEQRF sgeqrf_
#define ORGQR sorgqr_
#define ORMQR sormqr_
#define PSTRF spstrf_

#define CABS fabs
#define CACOS acos
#define CACOSH acosh
#define CASIN asin
#define CASINH asinh
#define CATAN atan
#define CATANH atanh
#define CCOS cos
#define CCOSH cosh
#define CEXP exp
#define CLOG log
#define CPOW pow
#define CPROJ proj
#define CSIN sin
#define CSINH sinh
#define CSQRT sqrt
#define CTAN tan
#define CTANH tanh
#define CREAL
#define CIMAG
#define CARG
#define CONJ

#define part float
#define Part Float

#include "THGenerateFloatType.h"

#undef part
#undef Part

#undef SWAPS
#undef SCAL
#undef COPY
#undef AXPY
#undef DOT
#undef GEMV
#undef GER
#undef GEMM

#undef GESV
#undef TRTRS
#undef GELS
#undef SYEV
#undef GEEV
#undef GESVD
#undef GETRF
#undef GETRS
#undef GETRI
#undef POTRF
#undef POTRI
#undef POTRS
#undef GEQRF
#undef ORGQR
#undef ORMQR
#undef PSTRF

#define GESV dgesv_
#define TRTRS dtrtrs_
#define GELS dgels_
#define SYEV dsyev_
#define GEEV dgeev_
#define GESVD dgesvd_
#define GETRF dgetrf_
#define GETRS dgetrs_
#define GETRI dgetri_
#define POTRF dpotrf_
#define POTRI dpotri_
#define POTRS dpotrs_
#define GEQRF dgeqrf_
#define ORGQR dorgqr_
#define ORMQR dormqr_
#define PSTRF dpstrf_

#define SWAPS dswap_
#define SCAL dscal_
#define COPY dcopy_
#define AXPY daxpy_
#define DOT ddot_
#define GEMV dgemv_
#define GER dger_
#define GEMM dgemm_

#define part double
#define Part Double

#include "THGenerateDoubleType.h"

#undef part
#undef Part

#undef CABS
#undef CACOS
#undef CACOSH
#undef CASIN
#undef CASINH
#undef CATAN
#undef CATANH
#undef CCOS
#undef CCOSH
#undef CEXP
#undef CLOG
#undef CPOW
#undef CPROJ
#undef CSIN
#undef CSINH
#undef CSQRT
#undef CTAN
#undef CTANH
#undef CREAL
#undef CIMAG
#undef CARG
#undef CONJ

#undef SWAPS
#undef SCAL
#undef COPY
#undef AXPY
#undef DOT
#undef GEMV
#undef GER
#undef GEMM

#undef GESV
#undef TRTRS
#undef GELS
#undef SYEV
#undef GEEV
#undef GESVD
#undef GETRF
#undef GETRS
#undef GETRI
#undef POTRF
#undef POTRI
#undef POTRS
#undef GEQRF
#undef ORGQR
#undef ORMQR
#undef PSTRF

#undef TH_REAL_IS_REAL

#ifdef THFloatLocalGenerateManyTypes
#undef THFloatLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
