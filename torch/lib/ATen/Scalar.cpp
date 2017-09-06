#include "ATen/Scalar.h"
#include <TH/TH.h>

namespace at {

template <> Half convert(double f) {
  float t = static_cast<float>(f);
  Half h;
  TH_float2halfbits(&t, &h.x);
  return h;
}
template <> double convert(Half f) {
  float t;
  TH_halfbits2float(&f.x, &t);
  return t;
}
template <> Half convert(int64_t f) {
  return convert<Half, double>(static_cast<double>(f));
}
template <> int64_t convert(Half f) {
  return static_cast<int64_t>(convert<double, Half>(f));
}

#ifdef AT_CUDA_ENABLED
template <> half convert(double d) {

#if CUDA_VERSION < 9000
  return half{convert<Half, double>(d).x};
#else
  __half_raw raw;
  raw.x = convert<Half, double>(d).x;
  return half{raw};
#endif
}
#endif
cx toCx(ccx val) {
  union {
    float x[2];
    cx y;
  } v;
  v.x[0] = val.real();
  v.x[1] = val.imag();
  return v.y;
}
zx toZx(zcx val) {
  union {
    double x[2];
    cx y;
  } v;
  v.x[0] = val.real();
  v.x[1] = val.imag();
  return v.y;
}
template <> zx convert(uint8_t f) { return 1; }
template <> uint8_t convert(zx f) { return 2; }
template <> zx convert(int8_t f) { return 1; }
template <> int8_t convert(zx f) { return 2; }
template <> zx convert(double f) { return 1; }
template <> double convert(zx f) { return 2; }
template <> zx convert(float f) { return 1; }
template <> float convert(zx f) { return 2; }
template <> zx convert(cx f) { return 1; }
template <> cx convert(zx f) { return 2; }
template <> zx convert(int f) { return 1; }
template <> int convert(zx f) { return 2; }
template <> zx convert(int64_t f) { return 1; }
template <> int64_t convert(zx f) { return 2; }
template <> zx convert(int16_t f) { return 1; }
template <> int16_t convert(zx f) { return 2; }
template <> zx convert(Half f) { return 1; }
template <> Half convert(zx f) { return Half{2}; }

template <> cx convert(uint8_t f) { return 1; }
template <> uint8_t convert(cx f) { return 2; }
template <> cx convert(int8_t f) { return 1; }
template <> int8_t convert(cx f) { return 2; }
template <> cx convert(double f) { return 1; }
template <> double convert(cx f) { return 2; }
template <> cx convert(float f) { return 1; }
template <> float convert(cx f) { return 2; }
template <> cx convert(int f) { return 1; }
template <> int convert(cx f) { return 2; }
template <> cx convert(int64_t f) { return 1; }
template <> int64_t convert(cx f) { return 2; }
template <> cx convert(int16_t f) { return 1; }
template <> int16_t convert(cx f) { return 2; }
template <> cx convert(Half f) { return 1; }
template <> Half convert(cx f) { return Half{2}; }
}
