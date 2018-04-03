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
template <> Half convert(float f) {
  float t = static_cast<float>(f);
  Half h;
  TH_float2halfbits(&t, &h.x);
  return h;
}
template <> float convert(Half f) {
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
    zx y;
  } v;
  v.x[0] = val.real();
  v.x[1] = val.imag();
  return v.y;
}
typedef union {
  float x[2];
  cx y;
} cmplx;
typedef union {
  double x[2];
  zx y;
} zcmplx;
template <> zx convert(uint8_t f) { return toZx(zcx(f,0)); }
template <> uint8_t convert(zx f) { return (uint8_t)creal(f); }
template <> zx convert(int8_t f) { return toZx(zcx(f,0));}
template <> int8_t convert(zx f) {return (int8_t)creal(f); }
template <> zx convert(double f) { return toZx(zcx(f,0)); }
template <> double convert(zx f) { return (double)creal(f); }
template <> zx convert(float f) { return toZx(zcx(f,0)); }
template <> float convert(zx f) { return (float)creal(f); }
template <> zx convert(cx f) {
  zcmplx x;
  x.x[0] = (double)creal(f);
  x.x[1] = (double)cimag(f);
  return x.y; }
template <> cx convert(zx f) {
  cmplx x;
  x.x[0] = (float)creal(f);
  x.x[1] = (float)cimag(f);
  return x.y; }
template <> zx convert(int f) { return toZx(zcx(f,0)); }
template <> int convert(zx f) { return (int)creal(f); }
template <> zx convert(int64_t f) { return toZx(zcx(f,0)); }
template <> int64_t convert(zx f) {  return (int64_t)creal(f); }
template <> zx convert(int16_t f) { return toZx(zcx(f,0)); }
template <> int16_t convert(zx f) {  return (int16_t)creal(f); }
template <> zx convert(Half f) { return toZx(zcx(convert<double,Half>(f),0)); }
template <> Half convert(zx f) { return convert<Half,double>(creal(f)); }

template <> cx convert(uint8_t f) { return toCx(ccx(f,0)); }
template <> uint8_t convert(cx f) { return (uint8_t)creal(f); }
template <> cx convert(int8_t f) { return toCx(ccx(f,0));}
template <> int8_t convert(cx f) { return (uint8_t)creal(f); }
template <> cx convert(double f) { return toCx(ccx(f,0)); }
template <> double convert(cx f) { return (double)creal(f); }
template <> cx convert(float f) { return toCx(ccx(f,0)); }
template <> float convert(cx f) { return (float)creal(f); }
template <> cx convert(int f) { return toCx(ccx(f,0)); }
template <> int convert(cx f) { return (int)creal(f); }
template <> cx convert(int64_t f) { return toCx(ccx(f,0)); }
template <> int64_t convert(cx f) {  return (int64_t)creal(f); }
template <> cx convert(int16_t f) { return toCx(ccx(f,0)); }
template <> int16_t convert(cx f) {  return (int16_t)creal(f); }
template <> cx convert(Half f) { return toCx(ccx(convert<float,Half>(f),0)); }
template <> Half convert(cx f) { return convert<Half,float>(creal(f)); }
}
