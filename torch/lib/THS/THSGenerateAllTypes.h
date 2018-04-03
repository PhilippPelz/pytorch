#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateAllTypes.h"
#endif

#define real uint8_t
#define accreal int64_t
#define Real Byte
#define THSInf UINT8_MAX
#define THS_REAL_IS_BYTE
#line 1 THS_GENERIC_FILE
/*#line 1 "THSByteStorage.h"*/
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_BYTE

#define real int8_t
#define accreal int64_t
#define Real Char
#define THSInf INT8_MAX
#define THS_REAL_IS_CHAR
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_CHAR

#define real int16_t
#define accreal int64_t
#define Real Short
#define THSInf INT16_MAX
#define THS_REAL_IS_SHORT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_SHORT

#define real int32_t
#define accreal int64_t
#define Real Int
#define THSInf INT32_MAX
#define THS_REAL_IS_INT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_INT

#define real int64_t
#define accreal int64_t
#define Real Long
#define THSInf INT64_t
#define THS_REAL_IS_LONG
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define THSInf FLT_MAX
#define THS_REAL_IS_FLOAT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THSInf DBL_MAX
#define THS_REAL_IS_DOUBLE
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_DOUBLE

#define CABS(x) cabsf((float _Complex)x)
#define CACOS(x) cacosf((float _Complex)x)
#define CACOSH(x) cacoshf((float _Complex)x)
#define CARG(x) cargf((float _Complex)x)
#define CASIN(x) casinf((float _Complex)x)
#define CASINH(x) casinhf((float _Complex)x)
#define CATAN(x) catanf((float _Complex)x)
#define CATANH(x) catanhf((float _Complex)x)
#define CCOS(x) ccosf((float _Complex)x)
#define CCOSH(x) ccoshf((float _Complex)x)
#define CEXP(x) cexpf((float _Complex)x)
#define CIMAG(x) cimagf((float _Complex)x)
#define CLOG(x) clogf((float _Complex)x)
#define CONJ(x) conjf((float _Complex)x)
#define CPOW(x) cpowf((float _Complex)x)
#define CPROJ(x) cprojf((float _Complex)x)
#define CREAL(x) crealf((float _Complex)x)
#define CSIN(x) csinf((float _Complex)x)
#define CSINH(x) csinhf((float _Complex)x)
#define CSQRT(x) csqrtf((float _Complex)x)
#define CTAN(x) ctanf((float _Complex)x)
#define CTANH(x) ctanhf((float _Complex)x)

#define SWAPS cswap_
#define SCAL cscal_
#define COPY ccopy_
#define AXPY caxpy_
#define DOT cdot_
#define GEMV cgemv_
#define GER cger_
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
#define accreal zx
#define part float
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real ZFloat
#define Part Float
#define THInf FLT_MAX
#define TH_REAL_IS_ZFLOAT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
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

#define CABS(x) cabs((double _Complex)x)
#define CACOS(x) cacos((double _Complex)x)
#define CACOSH(x) cacosh((double _Complex)x)
#define CARG(x) carg((double _Complex)x)
#define CASIN(x) casin((double _Complex)x)
#define CASINH(x) casinh((double _Complex)x)
#define CATAN(x) catan((double _Complex)x)
#define CATANH(x) catanh((double _Complex)x)
#define CCOS(x) ccos((double _Complex)x)
#define CCOSH(x) ccosh((double _Complex)x)
#define CEXP(x) cexp((double _Complex)x)
#define CIMAG(x) cimag((double _Complex)x)
#define CLOG(x) clog((double _Complex)x)
#define CONJ(x) conj((double _Complex)x)
#define CPOW(x) cpow((double _Complex)x)
#define CPROJ(x) cproj((double _Complex)x)
#define CREAL(x) creal((double _Complex)x)
#define CSIN(x) csin((double _Complex)x)
#define CSINH(x) csinh((double _Complex)x)
#define CSQRT(x) csqrt((double _Complex)x)
#define CTAN(x) ctan((double _Complex)x)
#define CTANH(x) ctanh((double _Complex)x)

#define SWAPS zswap_
#define SCAL zscal_
#define COPY zcopy_
#define AXPY zaxpy_
#define DOT zdot_
#define GEMV zgemv_
#define GER zger_
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
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
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

#undef THS_GENERIC_FILE
