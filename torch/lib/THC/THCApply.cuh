#ifndef THC_APPLY_INC
#define THC_APPLY_INC

#include "THCTensorCopy.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorTypeUtils.cuh"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
#define THC_APPLY_THREADS_PER_BLOCK 32 * 16

template <typename Op,
          typename Ta,
          typename IndexType,
          int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelPointwiseApply1(TensorInfo<Ta, IndexType> a,
                      IndexType totalElements,
                      Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

    op(&a.data[aOffset]);
  }
}

template <typename Op,
          typename Ta, typename Tb,
          typename IndexType,
          int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelPointwiseApply2(TensorInfo<Ta, IndexType> a,
                      TensorInfo<Tb, IndexType> b,
                      IndexType totalElements,
                      Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<Tb, IndexType, BDims>::get(linearIndex, b);

    op(&a.data[aOffset], &b.data[bOffset]);
  }
}

template <typename Op,
          typename Ta, typename Tb, typename Tc,
          typename IndexType,
          int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelPointwiseApply3(TensorInfo<Ta, IndexType> a,
                      TensorInfo<Tb, IndexType> b,
                      TensorInfo<Tc, IndexType> c,
                      IndexType totalElements,
                      Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<Tb, IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<Tc, IndexType, CDims>::get(linearIndex, c);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
  }
}
// TensorInfo<float, unsigned long>,
// TensorInfo<float, unsigned long>,
// TensorInfo<float, unsigned long>,
// TensorInfo<float, unsigned int>,
// TensorInfo<ccx, unsigned int>,
// unsigned long,
// const THNN_CudaZFloatModProj_renorm_bg
template <typename Op,
          typename Ta, typename Tb, typename Tc, typename Td, typename Te,
          typename IndexType,
          int ADims, int BDims, int CDims, int DDims, int EDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelPointwiseApply5(TensorInfo<Ta, IndexType> a,
                      TensorInfo<Tb, IndexType> b,
                      TensorInfo<Tc, IndexType> c,
                      TensorInfo<Td, IndexType> d,
                      TensorInfo<Te, IndexType> e,
                      IndexType totalElements,
                      const Op& op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<Ta,IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<Tb,IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<Tc,IndexType, CDims>::get(linearIndex, c);

    const IndexType dOffset =
      IndexToOffset<Td,IndexType, DDims>::get(linearIndex, d);

    const IndexType eOffset =
      IndexToOffset<Te,IndexType, EDims>::get(linearIndex, e);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset], &d.data[dOffset], &e.data[eOffset]);
  }
}

inline dim3 getApplyBlock() {
  return dim3(THC_APPLY_THREADS_PER_BLOCK);
}

inline bool getApplyGrid(THCState* state, ptrdiff_t totalElements, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);

  if (curDevice == -1) {
    return false;
  }

  // Assume a reasonable number of SMs if no state is available
  int numSM =
    state ? THCState_getCurrentDeviceProperties(state)->multiProcessorCount : 15;

  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
  // which seems to be a good sweetspot for latency hiding
  grid = dim3(min((long long) THCCeilDiv(totalElements,
                                         (ptrdiff_t) THC_APPLY_THREADS_PER_BLOCK),
                  4LL * numSM));
  return true;
}

template <typename TensorTypeA,
          typename Op>
bool THC_pointwiseApply1(THCState* state,
                         TensorTypeA* a,
                         const Op& op,
                         TensorArgType aType = ReadWrite) {
  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                            \
  kernelPointwiseApply1<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,   \
                        TYPE, A>                                        \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      aInfo, (TYPE) totalElements, op);

#define HANDLE_A_CASE(TYPE, A)                  \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, -2);                    \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, 1);                   \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, 2);                   \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, -1);                  \
        break;                                  \
      }                                         \
    }                                           \
  }

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      kernelPointwiseApply1<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            unsigned long, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    } else {
      kernelPointwiseApply1<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            unsigned long, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  return true;
}

template <typename TensorTypeA,
          typename TensorTypeB,
          typename Op>
bool THC_pointwiseApply2(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly) {
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                         \
  kernelPointwiseApply2<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        typename TensorUtils<TensorTypeB>::DataType,    \
                        TYPE, A, B>                                     \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, (TYPE) totalElements, op);

#define HANDLE_B_CASE(TYPE, A, B)               \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, -2);                 \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, 1);                \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, 2);                \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, -1);               \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B)               \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B);               \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B);              \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B);              \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B);             \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);
    bInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned long> bInfo =
      getTensorInfo<TensorTypeB, unsigned long>(state, b);
    bInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      kernelPointwiseApply2<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            unsigned long, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    } else {
      kernelPointwiseApply2<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            unsigned long, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  return true;
}

template <typename TensorTypeA,
          typename TensorTypeB,
          typename TensorTypeC,
          typename TensorTypeD,
          typename TensorTypeE,
          typename Op>
bool THC_pointwiseApply5(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         TensorTypeC* c,
                         TensorTypeD* d,
                         TensorTypeE* e,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly,
                         TensorArgType cType = ReadOnly,
                         TensorArgType dType = ReadOnly,
                         TensorArgType eType = ReadOnly) {
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b) ||
      totalElements != TensorUtils<TensorTypeC>::getNumElements(state, c) ||
      totalElements != TensorUtils<TensorTypeD>::getNumElements(state, d) ||
      totalElements != TensorUtils<TensorTypeE>::getNumElements(state, e)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeC>::getDims(state, c) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeD>::getDims(state, d) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeE>::getDims(state, e) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;
  TensorTypeC* oldC = NULL;
  TensorTypeD* oldD = NULL;
  TensorTypeE* oldE = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }
  if (cType == ReadWrite &&
      TensorUtils<TensorTypeC>::overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = TensorUtils<TensorTypeC>::newContiguous(state, c);
  }
  if (dType == ReadWrite &&
      TensorUtils<TensorTypeD>::overlappingIndices(state, d)) {
    // Must perform in contiguous space
    oldD = d;
    d = TensorUtils<TensorTypeD>::newContiguous(state, d);
  }
  if (eType == ReadWrite &&
      TensorUtils<TensorTypeE>::overlappingIndices(state, e)) {
    // Must perform in contiguous space
    oldE = e;
    e = TensorUtils<TensorTypeE>::newContiguous(state, e);
  }

#define HANDLE_CASE(TYPE, A, B, C, D, E)                                      \
  kernelPointwiseApply5<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        typename TensorUtils<TensorTypeB>::DataType,    \
                        typename TensorUtils<TensorTypeC>::DataType,    \
                        typename TensorUtils<TensorTypeD>::DataType,    \
                        typename TensorUtils<TensorTypeE>::DataType,    \
                        TYPE, A, B, C, D, E>                                  \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, cInfo, dInfo, eInfo, (TYPE) totalElements, op);

      #define HANDLE_E_CASE(TYPE, A, B, C, D, E)             \
        {                                              \
          if (eInfo.isContiguous()) {                  \
            HANDLE_CASE(TYPE, A, B, C, D,-2);               \
          } else {                                     \
            switch (E) {                               \
              case 1:                                  \
                HANDLE_CASE(TYPE, A, B,C, D, 1);            \
                break;                                 \
              case 2:                                  \
                HANDLE_CASE(TYPE, A, B,C, D, 2);            \
                break;                                 \
              case 3:                                  \
                HANDLE_CASE(TYPE, A, B,C, D, 3);            \
                break;                                 \
              default:                                 \
                HANDLE_CASE(TYPE, A, B,C, D, -1);           \
                break;                                 \
            }                                          \
          }                                            \
        }

      #define HANDLE_D_CASE(TYPE, A, B, C, D, E)             \
        {                                              \
          if (dInfo.isContiguous()) {                  \
            HANDLE_E_CASE(TYPE, A, B, C,-2, E);               \
          } else {                                     \
            switch (D) {                               \
              case 1:                                  \
                HANDLE_E_CASE(TYPE, A, B,C, 1, E);            \
                break;                                 \
              case 2:                                  \
                HANDLE_E_CASE(TYPE, A, B,C, 2, E);            \
                break;                                 \
              case 3:                                  \
                HANDLE_E_CASE(TYPE, A, B,C, 3, E);            \
                break;                                 \
              default:                                 \
                HANDLE_E_CASE(TYPE, A, B,C, -1, E);           \
                break;                                 \
            }                                          \
          }                                            \
        }

      #define HANDLE_C_CASE(TYPE, A, B, C, D, E)             \
        {                                              \
          if (cInfo.isContiguous()) {                  \
            HANDLE_D_CASE(TYPE, A, B, -2, D, E);               \
          } else {                                     \
            switch (C) {                               \
              case 1:                                  \
                HANDLE_D_CASE(TYPE, A, B, 1, D, E);            \
                break;                                 \
              case 2:                                  \
                HANDLE_D_CASE(TYPE, A, B, 2, D, E);            \
                break;                                 \
              case 3:                                  \
                HANDLE_D_CASE(TYPE, A, B, 3, D, E);            \
                break;                                 \
              default:                                 \
                HANDLE_D_CASE(TYPE, A, B, -1, D, E);           \
                break;                                 \
            }                                          \
          }                                            \
        }

      #define HANDLE_B_CASE(TYPE, A, B, C, D, E)                 \
        {                                                  \
          if (bInfo.isContiguous()) {                      \
            HANDLE_C_CASE(TYPE, A, -2, C, D, E);                 \
          } else {                                         \
            switch (B) {                                   \
              case 1:                                      \
                HANDLE_C_CASE(TYPE, A, 1, C, D, E);              \
                break;                                     \
              case 2:                                      \
                HANDLE_C_CASE(TYPE, A, 2, C, D, E);              \
                break;                                     \
              case 3:                                      \
                HANDLE_C_CASE(TYPE, A, 3, C, D, E);              \
                break;                                     \
              default:                                     \
                HANDLE_C_CASE(TYPE, A, -1, C, D, E);             \
                break;                                     \
            }                                              \
          }                                                \
        }

      #define HANDLE_A_CASE(TYPE, A, B, C, D, E)                 \
        {                                                  \
          if (aInfo.isContiguous()) {                      \
            HANDLE_B_CASE(TYPE, -2, B, C, D, E);                 \
          } else {                                         \
            switch (A) {                                   \
              case 1:                                      \
                HANDLE_B_CASE(TYPE, 1, B, C, D, E);              \
                break;                                     \
              case 2:                                      \
                HANDLE_B_CASE(TYPE, 2, B, C, D, E);              \
                break;                                     \
              case 3:                                      \
                HANDLE_B_CASE(TYPE, 3, B, C, D, E);              \
                break;                                     \
              default:                                     \
                HANDLE_B_CASE(TYPE, -1, B, C, D, E);             \
                break;                                     \
            }                                              \
          }                                                \
        }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b) &&
      TensorUtils<TensorTypeC>::canUse32BitIndexMath(state, c) &&
      TensorUtils<TensorTypeD>::canUse32BitIndexMath(state, d) &&
      TensorUtils<TensorTypeE>::canUse32BitIndexMath(state, e)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned int> cInfo =
      getTensorInfo<TensorTypeC, unsigned int>(state, c);
    cInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeD>::DataType, unsigned int> dInfo =
      getTensorInfo<TensorTypeD, unsigned int>(state, d);
    dInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeE>::DataType, unsigned int> eInfo =
      getTensorInfo<TensorTypeE, unsigned int>(state, e);
    eInfo.collapseDims();
    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims, dInfo.dims, eInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned long> bInfo =
      getTensorInfo<TensorTypeB, unsigned long>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned long> cInfo =
      getTensorInfo<TensorTypeC, unsigned long>(state, c);
    cInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeD>::DataType, unsigned long> dInfo =
      getTensorInfo<TensorTypeD, unsigned long>(state, d);
    dInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeE>::DataType, unsigned long> eInfo =
      getTensorInfo<TensorTypeE, unsigned long>(state, e);
    eInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous() && dInfo.isContiguous() && eInfo.isContiguous()) {
      kernelPointwiseApply5<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            typename TensorUtils<TensorTypeD>::DataType,
                            typename TensorUtils<TensorTypeE>::DataType,
                            unsigned long, -2, -2, -2, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, dInfo, eInfo, (unsigned long) totalElements, op);
    } else {
      kernelPointwiseApply5<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            typename TensorUtils<TensorTypeD>::DataType,
                            typename TensorUtils<TensorTypeE>::DataType,
                            unsigned long, -1, -1, -1, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, dInfo, eInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_E_CASE
#undef HANDLE_D_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeC>::copyIgnoringOverlaps(state, oldC, c);
    TensorUtils<TensorTypeC>::free(state, c);
    c = oldC;
  }

  if (oldD) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeD>::copyIgnoringOverlaps(state, oldD, d);
    TensorUtils<TensorTypeD>::free(state, d);
    d = oldD;
  }

  if (oldE) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeE>::copyIgnoringOverlaps(state, oldE, e);
    TensorUtils<TensorTypeE>::free(state, e);
    e = oldE;
  }

  return true;
}

template <typename TensorTypeA,
          typename TensorTypeB,
          typename TensorTypeC,
          typename Op>
bool THC_pointwiseApply3(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         TensorTypeC* c,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly,
                         TensorArgType cType = ReadOnly) {
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b) ||
      totalElements != TensorUtils<TensorTypeC>::getNumElements(state, c)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeC>::getDims(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;
  TensorTypeC* oldC = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }
  if (cType == ReadWrite &&
      TensorUtils<TensorTypeC>::overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = TensorUtils<TensorTypeC>::newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  kernelPointwiseApply3<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        typename TensorUtils<TensorTypeB>::DataType,    \
                        typename TensorUtils<TensorTypeC>::DataType,    \
                        TYPE, A, B, C>                                  \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, cInfo, (TYPE) totalElements, op);

#define HANDLE_C_CASE(TYPE, A, B, C)            \
  {                                             \
    if (cInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, B, -2);              \
    } else {                                    \
      switch (C) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, B, 1);             \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, B, 2);             \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, B, -1);            \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)            \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_C_CASE(TYPE, A, -2, C);            \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_C_CASE(TYPE, A, 1, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_C_CASE(TYPE, A, 2, C);           \
        break;                                  \
        default:                                \
        HANDLE_C_CASE(TYPE, A, -1, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)            \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B, C);            \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B, C);           \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b) &&
      TensorUtils<TensorTypeC>::canUse32BitIndexMath(state, c)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned int> cInfo =
      getTensorInfo<TensorTypeC, unsigned int>(state, c);
    cInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned long> bInfo =
      getTensorInfo<TensorTypeB, unsigned long>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned long> cInfo =
      getTensorInfo<TensorTypeC, unsigned long>(state, c);
    cInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      kernelPointwiseApply3<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            unsigned long, -2, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    } else {
      kernelPointwiseApply3<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            unsigned long, -1, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeC>::copyIgnoringOverlaps(state, oldC, c);
    TensorUtils<TensorTypeC>::free(state, c);
    c = oldC;
  }

  return true;
}

#undef THC_APPLY_THREADS_PER_BLOCK

#endif // THC_APPLY_INC
