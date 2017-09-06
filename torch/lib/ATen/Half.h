#pragma once
#include <complex.h>
typedef float _Complex cx;
typedef double _Complex zx;
#include <stdint.h>
#ifdef AT_CUDA_ENABLED
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

namespace at {

template <typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

#if defined(__GNUC__)
#define AT_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define AT_ALIGN(n) __declspec(align(n))
#else
#define AT_ALIGN(n)
#endif

typedef struct AT_ALIGN(2) {
  unsigned short x;
#ifdef AT_CUDA_ENABLED
#if CUDA_VERSION < 9000
  operator half() { return half{x}; }
#else
  operator half() {
    __half_raw x_raw;
    x_raw.x = x;
    return half(x_raw);
  }
#endif
#endif
  operator double();
} Half;

template <> Half convert(double f);
template <> double convert(Half f);
template <> Half convert(int64_t f);
template <> int64_t convert(Half f);

template <> zx convert(uint8_t f);
template <> uint8_t convert(zx f);
template <> zx convert(int8_t f);
template <> int8_t convert(zx f);
template <> zx convert(double f);
template <> double convert(zx f);
template <> zx convert(float f);
template <> float convert(zx f);
template <> zx convert(cx f);
template <> cx convert(zx f);
template <> zx convert(int f);
template <> int convert(zx f);
template <> zx convert(int64_t f);
template <> int64_t convert(zx f);
template <> zx convert(int16_t f);
template <> int16_t convert(zx f);
template <> zx convert(Half f);
template <> Half convert(zx f);

template <> cx convert(uint8_t f);
template <> uint8_t convert(cx f);
template <> cx convert(int8_t f);
template <> int8_t convert(cx f);
template <> cx convert(double f);
template <> double convert(cx f);
template <> cx convert(float f);
template <> float convert(cx f);
template <> cx convert(int f);
template <> int convert(cx f);
template <> cx convert(int64_t f);
template <> int64_t convert(cx f);
template <> cx convert(int16_t f);
template <> int16_t convert(cx f);
template <> cx convert(Half f);
template <> Half convert(cx f);
// _(uint8_t, Byte, i)
// _(int8_t, Char, i)
// _(double, Double, d)
// _(float, Float, d)
// _(zx, ZDouble, c)
// _(cx, ZFloat, c)
// _(int, Int, i)
// _(int64_t, Long, i)
// _(int16_t, Short, i)
// _(Half, Half, d)

inline Half::operator double() { return convert<double, Half>(*this); }
#ifdef AT_CUDA_ENABLED
template <> half convert(double d);
#endif

template <typename To, typename From> static inline To HalfFix(From h) {
  return To{h.x};
}

#ifdef AT_CUDA_ENABLED
#if CUDA_VERSION >= 9000
template <> inline __half HalfFix<__half, Half>(Half h) {
  __half_raw raw;
  raw.x = h.x;
  return __half{raw};
}

template <> inline Half HalfFix<Half, __half>(__half h) {
  __half_raw raw(h);
  return Half{raw.x};
}
#endif
#endif
}
