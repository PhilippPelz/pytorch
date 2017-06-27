#include "undef_macros.h"

#define THStoragePtr THCStoragePtr
#define THPStoragePtr THCPStoragePtr
#define THTensorPtr THCTensorPtr
#define THPTensorPtr THCPTensorPtr
#define THPPartTensorPtr THCPPartTensorPtr

#define THStorage THCStorage
#define THStorage_(NAME) THCStorage_(NAME)
#define THTensor THCTensor
#define THTensor_(NAME) THCTensor_(NAME)
#define THPartTensor THCPartTensor
#define THPartTensor_(NAME) THCPartTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THPStorage THCPStorage
#define THPStorageBaseStr THCPStorageBaseStr
#define THPStorageStr THCPStorageStr
#define THPStorageClass THCPStorageClass
#define THPStorageType THCPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THPTensor THCPTensor
#define THPTensorStr THCPTensorStr
#define THPTensorBaseStr THCPTensorBaseStr
#define THPTensorClass THCPTensorClass
#define THPTensorType THCPTensorType

#define THPPartTensor_(NAME) TH_CONCAT_4(THCP,Part,Tensor_,NAME)
#define THPPartTensor_stateless_(NAME) TH_CONCAT_4(THCP,Part,Tensor_stateless_,NAME)
#define THPPartTensor THCPPartTensor
#define THPPartTensorStr THCPPartTensorStr
#define THPPartTensorBaseStr THCPPartTensorBaseStr
#define THPPartTensorClass THCPPartTensorClass
#define THPPartTensorType THCPPartTensorType

#define THPUtils_(NAME) TH_CONCAT_4(THCP, Real, Utils_, NAME)

#define THPTensorStatelessType THCPTensorStatelessType
#define THPTensorStateless THCPTensorStateless

#define THSTensorPtr THCSTensorPtr
#define THSPTensorPtr THCSPTensorPtr
#define THSTensor THCSTensor
#define THSTensor_(NAME) THCSTensor_(NAME)

#define THSPTensor_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_,NAME)
#define THSPTensor_stateless_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_stateless_,NAME)
#define THSPTensor THCSPTensor
#define THSPTensorStr THCSPTensorStr
#define THSPTensorBaseStr THCSPTensorBaseStr
#define THSPTensorClass THCSPTensorClass
#define THSPTensorType THCSPTensorType

#define THSPTensorStatelessType THCSPTensorStatelessType
#define THSPTensorStateless THCSPTensorStateless


#define LIBRARY_STATE_NOARGS state
#define LIBRARY_STATE state,
#define LIBRARY_STATE_TYPE THCState*,
#define TH_GENERIC_FILE THC_GENERIC_FILE

#define THHostTensor TH_CONCAT_3(TH,Real,Tensor)
#define THHostTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define THHostStorage TH_CONCAT_3(TH,Real,Storage)
#define THHostStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)
