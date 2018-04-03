#ifndef TH_GENERIC_FILE
#error                                                                         \
    "You must define TH_GENERIC_FILE before including THGenerateZDoubleType.h"
#endif

// #if defined(__STDC_IEC_559_COMPLEX__)
// #pragma message "__STDC_IEC_559_COMPLEX__ defined"
// #endif
// #if defined(__STDC_NO_COMPLEX__)
// #pragma message "__STDC_NO_COMPLEX__ defined"
// #endif

#define TH_REAL_IS_COMPLEX

#define CABS cabs
#define CACOS cacos
#define CACOSH cacosh
#define CARG carg
#define CASIN casin
#define CASINH casinh
#define CATAN catan
#define CATANH catanh
#define CCOS ccos
#define CCOSH ccosh
#define CEXP cexp
#define CIMAG cimag
#define CLOG clog
#define CONJ conj
#define CPOW cpow
#define CPROJ cproj
#define CREAL creal
#define CSIN csin
#define CSINH csinh
#define CSQRT csqrt
#define CTAN ctan
#define CTANH ctanh

#define SWAPS zswap_
#define SCAL zscal_
#define COPY zcopy_
#define AXPY zaxpy_
#define DOT zdotc_
#define GEMV zgemv_
#define GER zgerc_
#define GEMM zgemm_

#define GESV zgesv_
#define TRTRS ztrtrs_
#define GELS zgels_
#define SYEV zheev_
#define GEEV zgeev_
#define GESVD zgesvd_
#define GETRF zgetrf_
#define GETRS zgetrs_
#define GETRI zgetri_
#define POTRF zpotrf_
#define POTRI zpotri_
#define POTRS zpotrs_
#define GEQRF zgeqrf_
#define ORGQR zungqr_
#define ORMQR zunmqr_
#define PSTRF zpstrf_

// #pragma message "NOW DOING COMPLEX double"

#define real zx
#define accreal zx
#define part double
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real ZDouble
#define Part Double
#define THInf DBL_MAX
#define TH_REAL_IS_ZDOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef part
#undef Part
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_ZDOUBLE
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#undef CABS
#undef CACOS
#undef CACOSH
#undef CARG
#undef CASIN
#undef CASINH
#undef CATAN
#undef CATANH
#undef CCOS
#undef CCOSH
#undef CEXP
#undef CIMAG
#undef CLOG
#undef CONJ
#undef CPOW
#undef CPROJ
#undef CREAL
#undef CSIN
#undef CSINH
#undef CSQRT
#undef CTAN
#undef CTANH

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

#undef TH_REAL_IS_COMPLEX

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
