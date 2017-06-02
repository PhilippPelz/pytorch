#include "THCBlas.h"
#include "THCGeneral.h"
#include "THCHalf.h"

float THCudaBlas_Sdot(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasSdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Sdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

double THCudaBlas_Ddot(THCState *state, long n, double *x, long incx, double *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    double result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Ddot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

ccx THCudaBlas_Cdot(THCState *state, long n, ccx *x, long incx, ccx *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    cuFloatComplex result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasCdotc(handle, i_n, reinterpret_cast<cuFloatComplex*>(x), i_incx, reinterpret_cast<cuFloatComplex*>(y), i_incy, &result));
    return toCcx(result);
  }

  THError("cublasCdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return ccx(0,0);
}

zcx THCudaBlas_Zdot(THCState *state, long n, zcx *x, long incx, zcx *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    cuDoubleComplex result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    cuDoubleComplex* xc = reinterpret_cast<cuDoubleComplex*>(x);
    cuDoubleComplex* yc = reinterpret_cast<cuDoubleComplex*>(y);
    THCublasCheck(cublasZdotc(handle, i_n, xc, i_incx, yc, i_incy, &result));
    return toCcx(result);
  }

  THError("cublasZdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return zcx(0,0);
}

#ifdef CUDA_HALF_TENSOR
float THCudaBlas_Hdot(THCState *state, long n, half *x, long incx, half *y, long incy)
{
#if CUDA_VERSION >= 8000
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDotEx(handle, i_n, x, CUDA_R_16F, i_incx, y, CUDA_R_16F, i_incy, &result, CUDA_R_32F, CUDA_R_32F));
    return result;
}

  THError("Cublas_Hdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
#else
  THError("Cublas_Hdot requires CUDA 8.0+");
  return 0;
#endif
}
#endif

/* Level 2 */
void THCudaBlas_Sgemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasSgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Sgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Dgemv(THCState *state, char trans, long m, long n, double alpha, double *a, long lda, double *x, long incx, double beta, double *y, long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Dgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Cgemv(THCState *state, char trans, long m, long n,
                              ccx alpha, ccx *a, long lda,
                              ccx *x, long incx, ccx beta,
                              ccx *y, long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    cuFloatComplex* ac = reinterpret_cast<cuFloatComplex*>(a);
    cuFloatComplex* xc = reinterpret_cast<cuFloatComplex*>(x);
    cuFloatComplex* yc = reinterpret_cast<cuFloatComplex*>(y);
    cuFloatComplex* alphac = reinterpret_cast<cuFloatComplex*>(&alpha);
    cuFloatComplex* betac = reinterpret_cast<cuFloatComplex*>(&beta);
    THCublasCheck(cublasCgemv(handle, op, i_m, i_n, alphac, ac, i_lda, xc, i_incx, betac, yc, i_incy));
    return;
  }
  THError("Cublas_Cgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Zgemv(THCState *state, char trans, long m, long n,
                              zcx alpha, zcx *a,
                              long lda, zcx *x, long incx,
                              zcx beta, zcx *y,
                              long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cuDoubleComplex* ac = reinterpret_cast<cuDoubleComplex*>(a);
    cuDoubleComplex* xc = reinterpret_cast<cuDoubleComplex*>(x);
    cuDoubleComplex* yc = reinterpret_cast<cuDoubleComplex*>(y);
    cuDoubleComplex* alphac = reinterpret_cast<cuDoubleComplex*>(&alpha);
    cuDoubleComplex* betac = reinterpret_cast<cuDoubleComplex*>(&beta);

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasZgemv(handle, op, i_m, i_n, alphac, ac, i_lda, xc, i_incx, betac, yc, i_incy));
    return;
  }
  THError("Cublas_Zgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Sger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasSger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dger(THCState *state, long m, long n, double alpha, double *x, long incx, double *y, long incy, double *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasDger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}
void THCudaBlas_Cger(THCState *state, long m, long n, ccx alpha,
                             ccx *x, long incx, ccx *y, long incy,
                             ccx *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cuFloatComplex* ac = reinterpret_cast<cuFloatComplex*>(a);
      cuFloatComplex* xc = reinterpret_cast<cuFloatComplex*>(x);
      cuFloatComplex* yc = reinterpret_cast<cuFloatComplex*>(y);
      cuFloatComplex* alphac = reinterpret_cast<cuFloatComplex*>(&alpha);

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasCgerc(handle, i_m, i_n, alphac, xc, i_incx, yc, i_incy, ac, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Zger(THCState *state, long m, long n,
                             zcx alpha, zcx *x,
                             long incx, zcx *y, long incy,
                             zcx *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cuDoubleComplex* ac = reinterpret_cast<cuDoubleComplex*>(a);
      cuDoubleComplex* xc = reinterpret_cast<cuDoubleComplex*>(x);
      cuDoubleComplex* yc = reinterpret_cast<cuDoubleComplex*>(y);
      cuDoubleComplex* alphac = reinterpret_cast<cuDoubleComplex*>(&alpha);

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasZgerc(handle, i_m, i_n, alphac, xc, i_incx, yc, i_incy, ac, i_lda));
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void adjustLd(char transa, char transb, long m, long n, long k, long *lda, long *ldb, long *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transa_)
  {
    if(m == 1)
      *lda = k;
  }
  else
  {
    if(k == 1)
      *lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasSgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Sgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}
void THCudaBlas_Cgemm(THCState *state, char transa, char transb, long m,
                              long n, long k, ccx alpha, ccx *a,
                              long lda, ccx *b, long ldb, ccx beta,
                              ccx *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cuFloatComplex* ac = reinterpret_cast<cuFloatComplex*>(a);
    cuFloatComplex* bc = reinterpret_cast<cuFloatComplex*>(b);
    cuFloatComplex* cc = reinterpret_cast<cuFloatComplex*>(c);
    cuFloatComplex* alphac = reinterpret_cast<cuFloatComplex*>(&alpha);
    cuFloatComplex* betac = reinterpret_cast<cuFloatComplex*>(&beta);

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasCgemm(handle, opa, opb, i_m, i_n, i_k, alphac, ac, i_lda, bc, i_ldb, betac, cc, i_ldc));
    return;
  }
  THError("Cublas_Cgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

#ifdef CUDA_HALF_TENSOR
// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#  define CUDA_R_16F CUBLAS_DATA_HALF
#endif

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, long m, long n, long k, half alpha, half *a, long lda, half *b, long ldb, half beta, half *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));

    // Check for native Hgemm support
    if (THC_fastHalfInstructions(state)) {
      THCublasCheck(cublasHgemm(handle, opa, opb,
				i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb,
				&beta, c, i_ldc));
    } else {
      // Simulated Hgemm
      float fAlpha = THC_half2float(alpha);
      float fBeta = THC_half2float(beta);

      THCublasCheck(cublasSgemmEx(handle, opa, opb,
				  i_m, i_n, i_k, &fAlpha,
                                  a, CUDA_R_16F, i_lda, b, CUDA_R_16F,
				  i_ldb, &fBeta, c, CUDA_R_16F, i_ldc));
    }

    return;
  }
  THError("Cublas_Hgemm only supports m, n, k, lda, ldb, ldc"
          "with th bound [val] <= %d", INT_MAX);
}
#endif

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, long m, long n, long k, double alpha, double *a, long lda, double *b, long ldb, double beta, double *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Dgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Zgemm(THCState *state, char transa, char transb, long m,
                              long n, long k, zcx alpha,
                              zcx *a, long lda, zcx *b,
                              long ldb, zcx beta,
                              zcx *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cuDoubleComplex* ac = reinterpret_cast<cuDoubleComplex*>(a);
    cuDoubleComplex* bc = reinterpret_cast<cuDoubleComplex*>(b);
    cuDoubleComplex* cc = reinterpret_cast<cuDoubleComplex*>(c);
    cuDoubleComplex* alphac = reinterpret_cast<cuDoubleComplex*>(&alpha);
    cuDoubleComplex* betac = reinterpret_cast<cuDoubleComplex*>(&beta);

    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasZgemm(handle, opa, opb, i_m, i_n, i_k, alphac, ac, i_lda, bc, i_ldb, betac, cc, i_ldc));
    return;
  }
  THError("Cublas_Dgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}


void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             float alpha, const float *a[], long lda, const float *b[], long ldb,
                             float beta, float *c[], long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
}

void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             double alpha, const double *a[], long lda, const double *b[], long ldb,
                             double beta, double *c[], long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
}
void THCudaBlas_CgemmBatched(THCState *state, char transa, char transb,
                                     long m, long n, long k, ccx alpha,
                                     const ccx *a[], long lda,
                                     const ccx *b[], long ldb,
                                     ccx beta, ccx *c[], long ldc,
                                     long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_CgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  const cuFloatComplex** ac = reinterpret_cast<const cuFloatComplex**>(a);
  const cuFloatComplex** bc = reinterpret_cast<const cuFloatComplex**>(b);
  cuFloatComplex** cc = reinterpret_cast<cuFloatComplex**>(c);
  cuFloatComplex* alphac = reinterpret_cast<cuFloatComplex*>(&alpha);
  cuFloatComplex* betac = reinterpret_cast<cuFloatComplex*>(&beta);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasCgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   alphac, ac, (int)lda, bc, (int)ldb, betac, cc, (int)ldc,
                                   (int)batchCount));
}

void THCudaBlas_ZgemmBatched(THCState *state, char transa, char transb,
                                     long m, long n, long k,
                                     zcx alpha,
                                     const zcx *a[], long lda,
                                     const zcx *b[], long ldb,
                                     zcx beta, zcx *c[],
                                     long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_ZgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  const cuDoubleComplex** ac = reinterpret_cast<const cuDoubleComplex**>(a);
  const cuDoubleComplex** bc = reinterpret_cast<const cuDoubleComplex**>(b);
  cuDoubleComplex** cc = reinterpret_cast<cuDoubleComplex**>(c);
  cuDoubleComplex* alphac = reinterpret_cast<cuDoubleComplex*>(&alpha);
  cuDoubleComplex* betac = reinterpret_cast<cuDoubleComplex*>(&beta);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasZgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   alphac, ac, (int)lda, bc, (int)ldb, betac, cc, (int)ldc,
                                   (int)batchCount));
}

/* Inverse */
void THCudaBlas_Sgetrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
}

void THCudaBlas_Dgetrf(THCState *state, int n, double **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
}

void THCudaBlas_Cgetrf(THCState *state, int n, ccx **a, int lda,
                               int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Cgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  cuFloatComplex** ac = reinterpret_cast<cuFloatComplex**>(a);
  THCublasCheck(cublasCgetrfBatched(handle, n, ac, lda, pivot, info, batchSize));
}

void THCudaBlas_Zgetrf(THCState *state, int n, zcx **a,
                               int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Zgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  cuDoubleComplex** ac = reinterpret_cast<cuDoubleComplex**>(a);
  THCublasCheck(cublasZgetrfBatched(handle, n, ac, lda, pivot, info, batchSize));
}

THC_API void THCudaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
}


THC_API void THCudaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
}

THC_API void THCudaBlas_Cgetrs(THCState *state, char transa, int n, int nrhs,
                               const ccx **a, int lda, int *pivot,
                               ccx **b, int ldb, int *info,
                               int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  const cuFloatComplex** ac = reinterpret_cast<const cuFloatComplex**>(a);
  cuFloatComplex** bc = reinterpret_cast<cuFloatComplex**>(b);
  THCublasCheck(cublasCgetrsBatched(handle, opa, n, nrhs, ac, lda, pivot, bc, ldb, info, batchSize));
}


THC_API void THCudaBlas_Zgetrs(THCState *state, char transa, int n, int nrhs,
                               const zcx **a, int lda, int *pivot,
                               zcx **b, int ldb, int *info,
                               int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  const cuDoubleComplex** ac = reinterpret_cast<const cuDoubleComplex**>(a);
  cuDoubleComplex** bc = reinterpret_cast<cuDoubleComplex**>(b);
  THCublasCheck(cublasZgetrsBatched(handle, opa, n, nrhs, ac, lda, pivot, bc, ldb, info, batchSize));
}

void THCudaBlas_Sgetri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
}

void THCudaBlas_Dgetri(THCState *state, int n, const double **a, int lda, int *pivot, double **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
}

void THCudaBlas_Cgetri(THCState *state, int n, const ccx **a,
                               int lda, int *pivot, ccx **c, int ldc,
                               int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  const cuFloatComplex** ac = reinterpret_cast<const cuFloatComplex**>(a);
  cuFloatComplex** cc = reinterpret_cast<cuFloatComplex**>(c);
  THCublasCheck(cublasCgetriBatched(handle, n, ac, lda, pivot, cc, ldc, info, batchSize));
}

void THCudaBlas_Zgetri(THCState *state, int n,
                               const zcx **a, int lda, int *pivot,
                               zcx **c, int ldc, int *info,
                               int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  const cuDoubleComplex** ac = reinterpret_cast<const cuDoubleComplex**>(a);
  cuDoubleComplex** cc = reinterpret_cast<cuDoubleComplex**>(c);
  THCublasCheck(cublasZgetriBatched(handle, n, ac, lda, pivot, cc, ldc, info, batchSize));
}
