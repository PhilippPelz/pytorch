#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else

#ifdef BLAS_F2C
#define ffloat double
#else
#define ffloat float
#endif

TH_EXTERNC void dswap_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void sswap_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dscal_(int *n, double *a, double *x, int *incx);
TH_EXTERNC void sscal_(int *n, float *a, float *x, int *incx);
TH_EXTERNC void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void scopy_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void daxpy_(int *n, double *a, double *x, int *incx, double *y,
                       int *incy);
TH_EXTERNC void saxpy_(int *n, float *a, float *x, int *incx, float *y,
                       int *incy);
TH_EXTERNC double ddot_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
                       int *lda, double *x, int *incx, double *beta, double *y,
                       int *incy);
TH_EXTERNC void sgemv_(char *trans, int *m, int *n, float *alpha, float *a,
                       int *lda, float *x, int *incx, float *beta, float *y,
                       int *incy);
TH_EXTERNC void dger_(int *m, int *n, double *alpha, double *x, int *incx,
                      double *y, int *incy, double *a, int *lda);
TH_EXTERNC void sger_(int *m, int *n, float *alpha, float *x, int *incx,
                      float *y, int *incy, float *a, int *lda);
TH_EXTERNC void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       double *alpha, double *a, int *lda, double *b, int *ldb,
                       double *beta, double *c, int *ldc);
TH_EXTERNC void sgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       float *alpha, float *a, int *lda, float *b, int *ldb,
                       float *beta, float *c, int *ldc);
TH_EXTERNC void zswap_(int *n, double complex *x, int *incx, double complex *y,
                       int *incy);
TH_EXTERNC void cswap_(int *n, float complex *x, int *incx, float complex *y,
                       int *incy);
TH_EXTERNC void zscal_(int *n, double complex *a, double complex *x, int *incx);
TH_EXTERNC void cscal_(int *n, float complex *a, float complex *x, int *incx);
TH_EXTERNC void zcopy_(int *n, double complex *x, int *incx, double complex *y,
                       int *incy);
TH_EXTERNC void ccopy_(int *n, float complex *x, int *incx, float complex *y,
                       int *incy);
TH_EXTERNC float _Complex cdotc_(int *n, float _Complex *x, int *incx, float _Complex *y, int *incy);
TH_EXTERNC double _Complex zdotc_(int *n, double _Complex *x, int *incx, double _Complex *y, int *incy);
TH_EXTERNC void zaxpy_(int *n, double complex *a, double complex *x, int *incx,
                       double complex *y, int *incy);
TH_EXTERNC void caxpy_(int *n, float complex *a, float complex *x, int *incx,
                       float complex *y, int *incy);
TH_EXTERNC void zgemv_(char *trans, int *m, int *n, double complex *alpha,
                       double complex *a, int *lda, double complex *x,
                       int *incx, double complex *beta, double complex *y,
                       int *incy);
TH_EXTERNC void cgemv_(char *trans, int *m, int *n, float complex *alpha,
                       float complex *a, int *lda, float complex *x, int *incx,
                       float complex *beta, float complex *y, int *incy);
TH_EXTERNC void zgerc_(int *m, int *n, double complex *alpha, double complex *x,
                      int *incx, double complex *y, int *incy,
                      double complex *a, int *lda);
TH_EXTERNC void cger_(int *m, int *n, float complex *alpha, float complex *x,
                      int *incx, float complex *y, int *incy,
                      float complex *a, int *lda);
TH_EXTERNC void cgerc_(int *m, int *n, float complex *alpha, float complex *x,
                       int *incx, float complex *y, int *incy, float complex *a,
                       int *lda);
TH_EXTERNC void zgerc_(int *m, int *n, double complex *alpha, double complex *x,
                       int *incx, double complex *y, int *incy, double complex *a,
                       int *lda);
TH_EXTERNC void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       double complex *alpha, double complex *a, int *lda,
                       double complex *b, int *ldb, double complex *beta,
                       double complex *c, int *ldc);
TH_EXTERNC void cgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       float complex *alpha, float complex *a, int *lda,
                       float complex *b, int *ldb, float complex *beta,
                       float complex *c, int *ldc);

void THBlas_(swap)(long n, real *x, long incx, real *y, long incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    SWAPS(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    long i;
    for (i = 0; i < n; i++) {
      real z = x[i * incx];
      x[i * incx] = y[i * incy];
      y[i * incy] = z;
    }
  }
}

void THBlas_(scal)(long n, real a, real *x, long incx) {
  if (n == 1)
    incx = 1;

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((n <= INT_MAX) && (incx <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;

    SCAL(&i_n, &a, x, &i_incx);
    return;
  }
#endif
  {
    long i;
    for (i = 0; i < n; i++)
      x[i * incx] *= a;
  }
}

void THBlas_(copy)(long n, real *x, long incx, real *y, long incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    COPY(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    long i;
    for (i = 0; i < n; i++)
      y[i * incy] = x[i * incx];
  }
}

void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    AXPY(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    long i;
    for (i = 0; i < n; i++)
      y[i * incy] += a * x[i * incx];
  }
}

real THBlas_(dot)(long n, real *x, long incx, real *y, long incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    return (real)DOT(&i_n, x, &i_incx, y, &i_incy);
  }
#endif
  {
    long i;
    real sum = 0;
    for (i = 0; i < n; i++)
      sum += x[i * incx] * y[i * incy];
    return sum;
  }
}

void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda,
                   real *x, long incx, real beta, real *y, long incy) {
  if (n == 1)
    lda = m;

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((m <= INT_MAX) && (n <= INT_MAX) && (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) && (incy > 0) && (incy <= INT_MAX)) {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    GEMV(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return;
  }
#endif
  {
    long i, j;

    if ((trans == 'T') || (trans == 't')) {
      for (i = 0; i < n; i++) {
        real sum = 0;
        real *row_ = a + lda * i;
        for (j = 0; j < m; j++)
          sum += x[j * incx] * row_[j];
        if (beta == 0)
          y[i * incy] = alpha * sum;
        else
          y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    } else {
      if (beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for (j = 0; j < n; j++) {
        real *column_ = a + lda * j;
        real z = alpha * x[j * incx];
        for (i = 0; i < m; i++)
          y[i * incy] += z * column_[i];
      }
    }
  }
}

void THBlas_(ger)(long m, long n, real alpha, real *x, long incx, real *y,
                  long incy, real *a, long lda) {
  if (n == 1)
    lda = m;

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX) &&
      (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    GER(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
    return;
  }
#endif
  {
    long i, j;
    for (j = 0; j < n; j++) {
      real *column_ = a + j * lda;
      real z = alpha * y[j * incy];
      for (i = 0; i < m; i++)
        column_[i] += z * x[i * incx];
    }
  }
}

void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha,
                   real *a, long lda, real *b, long ldb, real beta, real *c,
                   long ldc) {
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if (n == 1)
    ldc = m;

  if (transa_) {
    if (m == 1)
      lda = k;
  } else {
    if (k == 1)
      lda = m;
  }

  if (transb_) {
    if (k == 1)
      ldb = n;
  } else {
    if (n == 1)
      ldb = k;
  }

#if defined(USE_BLAS) &&                                                       \
    defined(TH_REAL_IS_REAL)
  if ((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX) &&
      (ldb <= INT_MAX) && (ldc <= INT_MAX)) {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    GEMM(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb,
         &beta, c, &i_ldc);
    return;
  }
#endif
  {
    long i, j, l;
    if (!transa_ && !transb_) {
      real *a_ = a;
      for (i = 0; i < m; i++) {
        real *b_ = b;
        for (j = 0; j < n; j++) {
          real sum = 0;
          for (l = 0; l < k; l++)
            sum += a_[l * lda] * b_[l];
          b_ += ldb;
          if (beta == 0)
            c[j * ldc + i] = alpha * sum;
          else
            c[j * ldc + i] = beta * c[j * ldc + i] + alpha * sum;
        }
        a_++;
      }
    } else if (transa_ && !transb_) {
      real *a_ = a;
      for (i = 0; i < m; i++) {
        real *b_ = b;
        for (j = 0; j < n; j++) {
          real sum = 0;
          for (l = 0; l < k; l++)
            sum += a_[l] * b_[l];
          b_ += ldb;
          if (beta == 0)
            c[j * ldc + i] = alpha * sum;
          else
            c[j * ldc + i] = beta * c[j * ldc + i] + alpha * sum;
        }
        a_ += lda;
      }
    } else if (!transa_ && transb_) {
      real *a_ = a;
      for (i = 0; i < m; i++) {
        real *b_ = b;
        for (j = 0; j < n; j++) {
          real sum = 0;
          for (l = 0; l < k; l++)
            sum += a_[l * lda] * b_[l * ldb];
          b_++;
          if (beta == 0)
            c[j * ldc + i] = alpha * sum;
          else
            c[j * ldc + i] = beta * c[j * ldc + i] + alpha * sum;
        }
        a_++;
      }
    } else {
      real *a_ = a;
      for (i = 0; i < m; i++) {
        real *b_ = b;
        for (j = 0; j < n; j++) {
          real sum = 0;
          for (l = 0; l < k; l++)
            sum += a_[l] * b_[l * ldb];
          b_++;
          if (beta == 0)
            c[j * ldc + i] = alpha * sum;
          else
            c[j * ldc + i] = beta * c[j * ldc + i] + alpha * sum;
        }
        a_ += lda;
      }
    }
  }
}

#endif
