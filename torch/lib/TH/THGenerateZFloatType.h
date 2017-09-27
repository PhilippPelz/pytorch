#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateZFloatType.h"
#endif

// #if defined(__STDC_IEC_559_COMPLEX__)
// #pragma message "__STDC_IEC_559_COMPLEX__ defined"
// #endif
// #if defined(__STDC_NO_COMPLEX__)
// #pragma message "__STDC_NO_COMPLEX__ defined"
// #endif

#define TH_REAL_IS_COMPLEX

#define CABS cabsf
#define CACOS cacosf
#define CACOSH cacoshf
#define CARG cargf
#define CASIN casinf
#define CASINH casinhf
#define CATAN catanf
#define CATANH catanhf
#define CCOS ccosf
#define CCOSH ccoshf
#define CEXP cexpf
#define CIMAG cimagf
#define CLOG clogf
#define CONJ conjf
#define CPOW cpowf
#define CPROJ cprojf
#define CREAL crealf
#define CSIN csinf
#define CSINH csinhf
#define CSQRT csqrtf
#define CTAN ctanf
#define CTANH ctanhf

#define SWAPS cswap_
#define SCAL cscal_
#define COPY ccopy_
#define AXPY caxpy_
#define DOT cdotc_
#define GEMV cgemv_
#define GER cgerc_
#define GEMM cgemm_

#define GESV cgesv_
#define TRTRS ctrtrs_
#define GELS cgels_
#define SYEV cheev_
#define GEEV cgeev_
#define GESVD cgesvd_
#define GETRF cgetrf_
#define GETRS cgetrs_
#define GETRI cgetri_
#define POTRF cpotrf_
#define POTRI cpotri_
#define POTRS cpotrs_
#define GEQRF cgeqrf_
#define ORGQR cungqr_
#define ORMQR cunmqr_
#define PSTRF cpstrf_

// #pragma message "NOW DOING COMPLEX FLOAT"

#define real cx
#define accreal cx
#define part float
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real ZFloat
#define Part Float
#define THInf FLT_MAX
#define TH_REAL_IS_ZFLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Part
#undef part
#undef Real
#undef THInf
#undef TH_REAL_IS_ZFLOAT
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
