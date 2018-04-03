#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else

#if defined(THD_GENERIC_FILE) || defined(TH_REAL_IS_HALF)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif

struct THPStorage;
struct THPTensor;
struct THSPTensor;

typedef class THPPointer<THStorage> THStoragePtr;
typedef class THPPointer<THTensor> THTensorPtr;
typedef class THPPointer<THPStorage> THPStoragePtr;
typedef class THPPointer<THPTensor> THPTensorPtr;

#if GENERATE_SPARSE
typedef class THPPointer<THSTensor> THSTensorPtr;
typedef class THPPointer<THSPTensor> THSPTensorPtr;
#endif

#if (!defined(THC_GENERIC_FILE) || defined(THC_REAL_IS_HALF) ||                \
     defined(THC_REAL_IS_ZDOUBLE) || defined(THC_REAL_IS_ZFLOAT)) &&           \
    (!defined(THD_GENERIC_FILE))
template <> struct THPUtils_typeTraits<real> {

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) ||                 \
    defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) ||               \
    defined(THC_REAL_IS_HALF)
  static constexpr char *python_type_str = "float";

#elif defined(TH_REAL_IS_ZFLOAT) || defined(TH_REAL_IS_ZDOUBLE) ||             \
    defined(THC_REAL_IS_ZFLOAT) || defined(THC_REAL_IS_ZDOUBLE)

  static constexpr char *python_type_str = "complex";
#else

  static constexpr char *python_type_str = "int";
#endif
};
#endif

#undef GENERATE_SPARSE

#endif // TH_GENERIC_FILE
