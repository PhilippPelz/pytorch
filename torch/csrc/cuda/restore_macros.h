
#define THTensor                    TH_CONCAT_3(TH,Real,Tensor)
#define THTensor_(NAME)             TH_CONCAT_4(TH,Real,Tensor_,NAME)

#define THPTensor                   TH_CONCAT_3(THP,Real,Tensor)
#define THPTensorStr                TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define THPTensorClass              TH_CONCAT_3(THP,Real,TensorClass)
#define THPTensor_(NAME)            TH_CONCAT_4(THP,Real,Tensor_,NAME)

#define THPPartTensor                   TH_CONCAT_3(THP,Part,Tensor)
#define THPPartTensorStr                TH_CONCAT_STRING_3(torch.,Part,Tensor)
#define THPPartTensorClass              TH_CONCAT_3(THP,Part,TensorClass)
#define THPPartTensor_(NAME)            TH_CONCAT_4(THP,Part,Tensor_,NAME)

#define THPStorage TH_CONCAT_3(THP,Real,Storage)
#define THPStorageStr TH_CONCAT_STRING_3(torch.,Real,Storage)
#define THPStorageClass TH_CONCAT_3(THP,Real,StorageClass)
#define THPStorage_(NAME) TH_CONCAT_4(THP,Real,Storage_,NAME)
#define THPUtils_(NAME) TH_CONCAT_4(THP, Real, Utils_, NAME)

#ifdef _THP_CORE
#define THStoragePtr TH_CONCAT_3(TH,Real,StoragePtr)
#define THTensorPtr  TH_CONCAT_3(TH,Real,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Real,TensorPtr)
#define THPPartTensorPtr  TH_CONCAT_3(THP,Part,TensorPtr)
#endif
