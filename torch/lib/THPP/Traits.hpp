#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "Storage.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

#include <thrust/complex.h>
typedef thrust::complex<float> ccx;
typedef thrust::complex<double> zcx;

namespace thpp {

template<typename T>
struct type_traits {};

template<typename real>
struct interface_traits {
  using one = typename std::conditional<type_traits<real>::is_cuda, thrust::complex<double>,  double _Complex>::type;

  using three = typename std::conditional<type_traits<real>::is_cuda,
    thrust::complex<float>, float _Complex>::type;

  using four = typename std::conditional<type_traits<real>::is_floating_point,
    double, long long>::type;

  using scalar_type = typename std::conditional<type_traits<real>::is_complex_double,
    one,
    typename std::conditional<type_traits<real>::is_complex_float, three, four >::type
  >::type;
  using tensor_interface_type = TensorScalarInterface<scalar_type>;
  using storage_interface_type = StorageScalarInterface<scalar_type>;
};

template<>
struct type_traits<char> {
  static constexpr Type type = Type::CHAR;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;
  static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<unsigned char> {
  static constexpr Type type = Type::UCHAR;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<float> {
  static constexpr Type type = Type::FLOAT;
  static constexpr bool is_floating_point = true;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<double> {
  static constexpr Type type = Type::DOUBLE;
  static constexpr bool is_floating_point = true;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<float _Complex> {
  static constexpr Type type = Type::ZFLOAT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;
  static constexpr bool is_complex_float = true;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<double _Complex> {
  static constexpr Type type = Type::ZDOUBLE;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_float = false;
  static constexpr bool is_complex_double = true;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<short> {
  static constexpr Type type = Type::SHORT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<unsigned short> {
  static constexpr Type type = Type::USHORT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<int> {
  static constexpr Type type = Type::INT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<unsigned int> {
  static constexpr Type type = Type::UINT;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<long> {
  static constexpr Type type = Type::LONG;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<unsigned long> {
  static constexpr Type type = Type::ULONG;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<long long> {
  static constexpr Type type = Type::LONG_LONG;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<>
struct type_traits<unsigned long long> {
  static constexpr Type type = Type::ULONG_LONG;
  static constexpr bool is_floating_point = false;
  static constexpr bool is_complex_double = false;static constexpr bool is_complex_float = false;
  static constexpr bool is_cuda = false;
};

template<typename T>
struct type_traits<const T> : type_traits<T> {};

} // namespace thpp
