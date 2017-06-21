#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.c"
#else

void THStorage_(rawCopy)(THStorage *storage, real *src)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = src[i];
}

void THStorage_(copy)(THStorage *storage, THStorage *src)
{
  THArgCheck(storage->size == src->size, 2, "size mismatch");
  THStorage_(rawCopy)(storage, src->data);
}

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC) \
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  ptrdiff_t i;                                                        \
  for(i = 0; i < storage->size; i++)                                  \
    storage->data[i] = (real)src->data[i];                            \
}

#define IMPLEMENT_THStorage_COPY_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = (real)TH_half2float(src->data[i]);		\
}

#define IMPLEMENT_THStorage_COPY_TO_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = TH_float2half((float)(src->data[i]));		\
}

#define IMPLEMENT_THStorage_COPY_TO_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = src->data[i];		\
}

/*
Maybe copy from a complex storage to twice big storage is allowed, but is not allowed for tensor
This is because there could probably some networks use a complex output, but the network it self 
is a real network 
*/

#define IMPLEMENT_THStorage_COPY_FROM_DOUBLE_COMPLEX(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == 2 * src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
  { \
    storage->data[2*i] = (real) creal(src->data[i]);		\
    storage->data[2*i+1] = (real) cimag(src->data[i]);  \
  } \
}

#define IMPLEMENT_THStorage_COPY_FROM_FLOAT_COMPLEX(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == 2 * src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
  { \
    storage->data[2*i] = (real) crealf(src->data[i]);		\
    storage->data[2*i+1] = (real) cimagf(src->data[i]);  \
  } \
}

#define IMPLEMENT_THStorage_COPY_TO_COMPLEX(TYPENAMESRC)    \
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(2 * storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
  { \
    storage->data[i] = (part)(src->data[2*i]) + J * (part)(src->data[2*i+1]);   \
  } \
}

#define IMPLEMENT_THStorage_COPY_FROM_HALF_TO_COMPLEX(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == 2 * src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size; i++)					\
    storage->data[i] = (part)TH_half2float(src->data[2*i]) + (part)TH_half2float(src->data[2*i+1]);		\
}

#if defined(TH_REAL_IS_COMPLEX)

IMPLEMENT_THStorage_COPY(ZFloat)
IMPLEMENT_THStorage_COPY(ZDouble)

IMPLEMENT_THStorage_COPY_TO_COMPLEX(Byte)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Char)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Short)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Int)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Long)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Float)
IMPLEMENT_THStorage_COPY_TO_COMPLEX(Double)

IMPLEMENT_THStorage_COPY_FROM_HALF_TO_COMPLEX(Half)
#else

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THStorage_COPY(Byte)
IMPLEMENT_THStorage_COPY(Char)
IMPLEMENT_THStorage_COPY(Short)
IMPLEMENT_THStorage_COPY(Int)
IMPLEMENT_THStorage_COPY(Long)
IMPLEMENT_THStorage_COPY(Float)
IMPLEMENT_THStorage_COPY(Double)
IMPLEMENT_THStorage_COPY_FROM_FLOAT_COMPLEX(ZFloat)
IMPLEMENT_THStorage_COPY_FROM_DOUBLE_COMPLEX(ZDouble)
IMPLEMENT_THStorage_COPY_FROM_HALF(Half)
#else
/* only allow pass-through for Half */
IMPLEMENT_THStorage_COPY_TO_FROM_HALF(Half)
IMPLEMENT_THStorage_COPY_TO_HALF(Byte)
IMPLEMENT_THStorage_COPY_TO_HALF(Char)
IMPLEMENT_THStorage_COPY_TO_HALF(Short)
IMPLEMENT_THStorage_COPY_TO_HALF(Int)
IMPLEMENT_THStorage_COPY_TO_HALF(Long)
IMPLEMENT_THStorage_COPY_TO_HALF(Float)
IMPLEMENT_THStorage_COPY_TO_HALF(Double)
#endif

#endif

#endif
