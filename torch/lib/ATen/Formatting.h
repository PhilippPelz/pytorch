#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include <iostream>

namespace at {

std::ostream &operator<<(std::ostream &out, IntList list);
std::ostream &operator<<(std::ostream &out, Backend b);
std::ostream &operator<<(std::ostream &out, ScalarType t);
std::ostream &print(std::ostream &stream, const Tensor &tensor,
                    int64_t linesize);
static inline std::ostream &operator<<(std::ostream &out, const Tensor &t) {
  return print(out, t, 80);
}
static inline void print(const Tensor &t, int64_t linesize = 80) {
  print(std::cout, t, linesize);
}

static inline std::ostream &operator<<(std::ostream &out, Scalar s) {
  s = s.local();
  return out << (s.isFloatingPoint()
                     ? (s.isComplexFloatingPoint() ? 1 : s.toDouble())
                     : s.toLong());
}
}
