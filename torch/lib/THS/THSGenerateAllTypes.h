#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateAllTypes.h"
#endif

#include <complex.h>
#undef I
#define J _Complex_I

#define real unsigned char
#define accreal long
#define Real Byte
#define THSInf UCHAR_MAX
#define THS_REAL_IS_BYTE
#line 1 THS_GENERIC_FILE
/*#line 1 "THSByteStorage.h"*/
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_BYTE

#define real char
#define accreal long
#define Real Char
#define THSInf CHAR_MAX
#define THS_REAL_IS_CHAR
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define THSInf SHRT_MAX
#define THS_REAL_IS_SHORT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define THSInf INT_MAX
#define THS_REAL_IS_INT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define THSInf LONG_MAX
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

typedef float _Complex cx;
typedef double _Complex zx;

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
