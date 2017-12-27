#include <Python.h>
#include <structmember.h>

#define THP_HOST_HALF

#include <TH/THMath.h>
#include <stack>
#include <stdbool.h>
#include <tuple>
#include <vector>

#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/copy_utils.h"

//generic_include TH torch/csrc/generic/Tensor.cpp
