#pragma once

// ${generated_comment}

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

#include <ATen/ATen.h>
namespace torch { namespace autograd {

using at::Scalar;
using at::Tensor;
using at::IntList;

${autograd_function_declarations}

}} // namespace torch::autograd
