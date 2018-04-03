#pragma once

#include "Traits.hpp"
#include <THC/THCHalf.h>

#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

namespace thpp {

template<>
struct type_traits<half> {
  static constexpr Type type = Type::HALF;
  static constexpr bool is_floating_point = true;
  static constexpr bool is_complex_double = false;
  static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<ccx> {
  static constexpr Type type = Type::ZFLOAT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;
  static constexpr bool is_complex_float = true;
  static constexpr bool is_cuda = true;
};

template<>
struct type_traits<zcx> {
  static constexpr Type type = Type::ZDOUBLE;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = true;
  static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = true;
};
} // namespace thpp
