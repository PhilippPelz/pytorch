#ifndef THCP_UTILS_H
#define THCP_UTILS_H

#define THCPUtils_(NAME) TH_CONCAT_4(THCP,Real,Utils_,NAME)

#define THCStoragePtr  TH_CONCAT_3(THC,Real,StoragePtr)
#define THCTensorPtr   TH_CONCAT_3(THC,Real,TensorPtr)
#define THCPStoragePtr TH_CONCAT_3(THCP,Real,StoragePtr)
#define THCPTensorPtr  TH_CONCAT_3(THCP,Real,TensorPtr)
#define THCPPartTensorPtr  TH_CONCAT_3(THCP,Part,TensorPtr)

#define THCSTensorPtr  TH_CONCAT_3(THCS,Real,TensorPtr)
#define THCSPTensorPtr TH_CONCAT_3(THCSP,Real,TensorPtr)

static Py_complex thrust_ccomplex_to_py_complex(ccx cx) {
  Py_complex x;
  x.real = cx.real();
  x.imag = cx.imag();
  return x;
}
static Py_complex thrust_zcomplex_to_py_complex(zcx cx) {
  Py_complex x;
  x.real = cx.real();
  x.imag = cx.imag();
  return x;
}

static ccx py_complex_to_thrust_ccomplex(Py_complex x) {
  return ccx((float)x.real,(float)x.imag);
}

static zcx py_complex_to_thrust_zcomplex(Py_complex x) {
  return zcx((double)x.real,((double)x.imag));
}

#define THCPUtils_unpackReal_ZCOMPLEX(object)                                   \
  (PyComplex_Check(object)                                                     \
       ? py_complex_to_thrust_zcomplex(PyComplex_AsCComplex(object))                \
       : (throw std::runtime_error("Could not parse complex"), 0))

#define THCPUtils_unpackReal_CCOMPLEX(object)                                   \
  (PyComplex_Check(object)                                                     \
       ? py_complex_to_thrust_ccomplex(PyComplex_AsCComplex(object))                \
       : (throw std::runtime_error("Could not parse complex"), 0))

#define THCPUtils_newReal_ZCOMPLEX(value)                                       \
 PyComplex_FromCComplex(thrust_zcomplex_to_py_complex(value))

#define THCPUtils_newReal_CCOMPLEX(value)                                       \
 PyComplex_FromCComplex(thrust_ccomplex_to_py_complex(value))

#define THCPZDoubleUtils_checkReal(object) THPUtils_checkReal_COMPLEX(object)
#define THCPZDoubleUtils_unpackReal(object) THCPUtils_unpackReal_ZCOMPLEX(object)
#define THCPZDoubleUtils_newReal(value) THCPUtils_newReal_ZCOMPLEX(value)
#define THCPZDoubleUtils_checkAccreal(object) THPUtils_checkReal_COMPLEX(object)
#define THCPZDoubleUtils_unpackAccreal(object)                                  \
 THCPUtils_unpackReal_ZCOMPLEX(object)
#define THCPZDoubleUtils_newAccreal(value) THCPUtils_newReal_ZCOMPLEX(value)

#define THCPZDoubleUtils_checkPart(object) THCPDoubleUtils_checkReal(object)
#define THCPZDoubleUtils_unpackPart(object) THCPDoubleUtils_unpackReal(object)
#define THCPZDoubleUtils_newPart(value) THCPDoubleUtils_newReal(value)

#define THCPZFloatUtils_checkReal(object) THPUtils_checkReal_COMPLEX(object)
#define THCPZFloatUtils_unpackReal(object) THCPUtils_unpackReal_CCOMPLEX(object)
#define THCPZFloatUtils_newReal(value) THCPUtils_newReal_CCOMPLEX(value)
#define THCPZFloatUtils_checkAccreal(object) THPUtils_checkReal_COMPLEX(object)
#define THCPZFloatUtils_unpackAccreal(object)                                   \
  THCPUtils_unpackReal_ZCOMPLEX(object)
#define THCPZFloatUtils_newAccreal(value) THCPUtils_newReal_CCOMPLEX(value)

#define THCPZFloatUtils_checkPart(object) THCPFloatUtils_checkReal(object)
#define THCPZFloatUtils_unpackPart(object) THCPFloatUtils_unpackReal(object)
#define THCPZFloatUtils_newPart(value) THCPFloatUtils_newReal(value)

#define THCPDoubleUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)
#define THCPDoubleUtils_unpackReal(object)                                      \
  (double) THPUtils_unpackReal_FLOAT(object)
#define THCPDoubleUtils_newReal(value) THPUtils_newReal_FLOAT(value)
#define THCPDoubleUtils_checkAccreal(object) THPUtils_checkReal_FLOAT(object)
#define THCPDoubleUtils_unpackAccreal(object)                                   \
  (double) THPUtils_unpackReal_FLOAT(object)
#define THCPDoubleUtils_newAccreal(value) THPUtils_newReal_FLOAT(value)

#define THCPDoubleUtils_checkPart(object) THPUtils_checkReal_FLOAT(object)
#define THCPDoubleUtils_unpackPart(object)                                      \
  (double) THPUtils_unpackReal_FLOAT(object)
#define THCPDoubleUtils_newPart(value) THPUtils_newReal_FLOAT(value)

#define THCPFloatUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)
#define THCPFloatUtils_unpackReal(object)                                       \
  (float) THPUtils_unpackReal_FLOAT(object)
#define THCPFloatUtils_newReal(value) THPUtils_newReal_FLOAT(value)
#define THCPFloatUtils_checkAccreal(object) THPUtils_checkReal_FLOAT(object)
#define THCPFloatUtils_unpackAccreal(object)                                    \
  (double) THPUtils_unpackReal_FLOAT(object)
#define THCPFloatUtils_newAccreal(value) THPUtils_newReal_FLOAT(value)

#define THCPFloatUtils_checkPart(object) THPUtils_checkReal_FLOAT(object)
#define THCPFloatUtils_unpackPart(object)                                       \
  (float) THPUtils_unpackReal_FLOAT(object)
#define THCPFloatUtils_newPart(value) THPUtils_newReal_FLOAT(value)

#define THCPHalfUtils_checkReal(object) THPUtils_checkReal_FLOAT(object)
#ifndef THP_HOST_HALF

#define THCPHalfUtils_unpackReal(object)                                        \
  (half) THC_float2half(THPUtils_unpackReal_FLOAT(object))
#define THCPHalfUtils_newReal(value) PyFloat_FromDouble(THC_half2float(value))

#define THCPHalfUtils_unpackPart(object)                                        \
  (half) THC_float2half(THPUtils_unpackReal_FLOAT(object))
#define THCPHalfUtils_newPart(value) PyFloat_FromDouble(THC_half2float(value))

#else

#define THCPHalfUtils_unpackReal(object)                                        \
  TH_float2half(THPUtils_unpackReal_FLOAT(object))
#define THCPHalfUtils_newReal(value) PyFloat_FromDouble(TH_half2float(value))

#define THCPHalfUtils_unpackPart(object)                                        \
  (half) THC_float2half(THPUtils_unpackReal_FLOAT(object))
#define THCPHalfUtils_newPart(value) PyFloat_FromDouble(THC_half2float(value))

#endif

#define THCPHalfUtils_checkPart(object) THPUtils_checkReal_FLOAT(object)


#define THCPHalfUtils_checkAccreal(object) THPUtils_checkReal_FLOAT(object)
#define THCPHalfUtils_unpackAccreal(object)                                     \
  (double) THPUtils_unpackReal_FLOAT(object)
#define THCPHalfUtils_newAccreal(value) THPUtils_newReal_FLOAT(value)

#define THCPLongUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THCPLongUtils_unpackReal(object) (long) THPUtils_unpackReal_INT(object)
#define THCPLongUtils_newReal(value) THPUtils_newReal_INT(value)
#define THCPLongUtils_checkAccreal(object) THPUtils_checkReal_INT(object)
#define THCPLongUtils_unpackAccreal(object)                                     \
  (long) THPUtils_unpackReal_INT(object)
#define THCPLongUtils_newAccreal(value) THPUtils_newReal_INT(value)

#define THCPLongUtils_checkPart(object) THPUtils_checkReal_INT(object)
#define THCPLongUtils_unpackPart(object) (long) THPUtils_unpackReal_INT(object)
#define THCPLongUtils_newPart(value) THPUtils_newReal_INT(value)

#define THCPIntUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THCPIntUtils_unpackReal(object) (int) THPUtils_unpackReal_INT(object)
#define THCPIntUtils_newReal(value) THPUtils_newReal_INT(value)
#define THCPIntUtils_checkAccreal(object) THPUtils_checkReal_INT(object)
#define THCPIntUtils_unpackAccreal(object) (long) THPUtils_unpackReal_INT(object)
#define THCPIntUtils_newAccreal(value) THPUtils_newReal_INT(value)

#define THCPIntUtils_checkPart(object) THPUtils_checkReal_INT(object)
#define THCPIntUtils_unpackPart(object) (int) THPUtils_unpackReal_INT(object)
#define THCPIntUtils_newPart(value) THPUtils_newReal_INT(value)

#define THCPShortUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THCPShortUtils_unpackReal(object) (short) THPUtils_unpackReal_INT(object)
#define THCPShortUtils_newReal(value) THPUtils_newReal_INT(value)
#define THCPShortUtils_checkAccreal(object) THPUtils_checkReal_INT(object)
#define THCPShortUtils_unpackAccreal(object)                                    \
  (long) THPUtils_unpackReal_INT(object)
#define THCPShortUtils_newAccreal(value) THPUtils_newReal_INT(value)

#define THCPShortUtils_checkPart(object) THPUtils_checkReal_INT(object)
#define THCPShortUtils_unpackPart(object) (short) THPUtils_unpackReal_INT(object)
#define THCPShortUtils_newPart(value) THPUtils_newReal_INT(value)


#define THCPCharUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THCPCharUtils_unpackReal(object) (char) THPUtils_unpackReal_INT(object)
#define THCPCharUtils_newReal(value) THPUtils_newReal_INT(value)
#define THCPCharUtils_checkAccreal(object) THPUtils_checkReal_INT(object)
#define THCPCharUtils_unpackAccreal(object)                                     \
  (long) THPUtils_unpackReal_INT(object)
#define THCPCharUtils_newAccreal(value) THPUtils_newReal_INT(value)

#define THCPCharUtils_checkPart(object) THPUtils_checkReal_INT(object)
#define THCPCharUtils_unpackPart(object) (char) THPUtils_unpackReal_INT(object)
#define THCPCharUtils_newPart(value) THPUtils_newReal_INT(value)

#define THCPByteUtils_checkReal(object) THPUtils_checkReal_INT(object)
#define THCPByteUtils_unpackReal(object)                                        \
  (unsigned char) THPUtils_unpackReal_INT(object)
#define THCPByteUtils_newReal(value) THPUtils_newReal_INT(value)
#define THCPByteUtils_checkAccreal(object) THPUtils_checkReal_INT(object)
#define THCPByteUtils_unpackAccreal(object)                                     \
  (long) THPUtils_unpackReal_INT(object)
#define THCPByteUtils_newAccreal(value) THPUtils_newReal_INT(value)

#define THCPByteUtils_checkPart(object) THPUtils_checkReal_INT(object)
#define THCPByteUtils_unpackPart(object)                                        \
  (unsigned char) THPUtils_unpackReal_INT(object)
#define THCPByteUtils_newPart(value) THPUtils_newReal_INT(value)

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/utils.h"
#include <THC/THCGenerateAllTypes.h>

#endif
