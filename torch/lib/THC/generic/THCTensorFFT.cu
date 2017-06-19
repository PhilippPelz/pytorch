#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorFFT.cu"
#else

#if defined(THC_REAL_IS_ZFLOAT) || defined(THC_REAL_IS_ZDOUBLE)

// THC_API int THCState_getNumCuFFTPlans(THCState* state);
// THC_API cufftHandle* THCState_getCuFFTPlan(THCState* state,int batch, int n1, int n2, int n3) ;
void THCTensor_(fftnbase)(THCState *state, THCTensor *self, THCTensor *result, int direction) {
	int ndim = THCTensor_(nDimension)(state, result);
	int batch = 1;
	int *fft_dims = (int*)malloc(ndim*sizeof(int));
	for (int i = 0; i < ndim; i++) {
		fft_dims[i] = (int) THCTensor_(size)(state, self, i);
	}
	cufftHandle plan;
	cufftSafeCall(cufftPlanMany(&plan, ndim, fft_dims, NULL, 1, 0, NULL, 1, 0, cufftname, batch));
	cufftSafeCall(cufftSetStream(plan, THCState_getCurrentStream(state)));
	cufftSafeCall(cufft(plan, (cureal *)THCTensor_(data)(state, self), (cureal *)THCTensor_(data)(state, result), direction));
	cufftDestroy(plan);
	free(fft_dims);
}

// takes the first dimension as batch dimension
void THCTensor_(fftnBatchedbase)(THCState *state, THCTensor *self, THCTensor *result, int direction) {
	int ndim = THCTensor_(nDimension)(state, self) -1;
	int batch = THCTensor_(size)(state, self, 0);
	int *fft_dims = (int*)malloc(ndim*sizeof(int));
	FILE *f;
	//f = fopen("/home/philipp/fftnBatchedbase.log", "a+");
	//fprintf(f,"fftnBatchedbase start" );
	for (int i = 1; i <= ndim ; i++) {
		fft_dims[i - 1] = (int) THCTensor_(size)(state, self, i);
		//fprintf(f,"fft_dims[i - 1] = %d",fft_dims[i - 1]);
	}
	cufftHandle handle;
	cufftSafeCall(cufftPlanMany(&handle, ndim, fft_dims, NULL, 1, 0, NULL, 1, 0, cufftname, batch));
	//fprintf(f,"cufftPlanMany\n");
	//fclose(f);
	cufftSafeCall(cufftSetStream(handle, THCState_getCurrentStream(state)));
	//fprintf(f,"cufftSetStream\n");
	cufftSafeCall(cufft(handle, (cureal *)THCTensor_(data)(state, self), (cureal *)THCTensor_(data)(state, result), direction));
	//fprintf(f,"cufft\n");
	cufftDestroy(handle);

	free(fft_dims);
}

void THCTensor_(fftnBatched)(THCState *state, THCTensor *result, THCTensor *self) {
	FILE *f;
	f = fopen("/home/philipp/fftnBatched.log", "a+");
	//fprintf(f,"fftnBatchedbase done\n");
	//fclose(f);
	THCTensor_(fftnBatchedbase)(state, self, result, CUFFT_FORWARD);

	THCTensor_(mul)(state, result, result, ccx(1 / sqrt(THCTensor_(nElement)(state, result)),0));
}

void THCTensor_(ifftnBatched)(THCState *state, THCTensor *self, THCTensor *result) {
	THCTensor_(fftnBatchedbase)(state, self, result, CUFFT_INVERSE);
	THCTensor_(mul)(state, result, result, ccx(1 / sqrt(THCTensor_(nElement)(state, result)),0));
}

void THCTensor_(fft)(THCState *state, THCTensor *result, THCTensor *self) {
	if(THCTensor_(nDimension)(state, self) < 1)
		THError("tensor must at least have dimension 1\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	if (!THCTensor_(isSameSizeAs)(state, self, result))
    THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-1; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}

	THLongStorage *new_self_size = THLongStorage_newWithSize2( self_batch_dim, THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);

	THLongStorage *new_result_size = THLongStorage_newWithSize2( self_batch_dim, THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);

	THCTensor_(fftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(fft2)(THCState *state, THCTensor *result, THCTensor *self) {
	FILE *f;
	f = fopen("/home/philipp/x.log", "a+"); // a+ (create + append) option will allow appending which is useful in a log file
	if (f == NULL) { /* Something is wrong   */}
	//fprintf(f,"fft2\n");
	if(THCTensor_(nDimension)(state, self) < 2)
		THError("tensor must at least have dimension 2\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	int res_ndim = THCTensor_(nDimension)(state, result);
	fprintf(f,"(self_dim,res_dim) = (%d,%d)\n",self_ndim,res_ndim);
	fclose(f);
	// if (!THCTensor_(isSameSizeAs)(state, self, result))
  //   THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-2; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}
	//fprintf(f,"self_batch_dim = %d\n",self_batch_dim);
  //fprintf(f,"dim1 = %d\n",THCTensor_(size)(state, self, self_ndim-2));
	//fprintf(f,"dim2 = %d\n",THCTensor_(size)(state, self, self_ndim-1));
	THLongStorage *new_self_size = THLongStorage_newWithSize3( self_batch_dim, THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	//fprintf(f,"after THLongStorage_newWithSize3\n");
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);
	//fprintf(f,"after newView\n");

	THLongStorage *new_result_size = THLongStorage_newWithSize3( self_batch_dim, THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	//fprintf(f,"after THLongStorage_newWithSize3\n");
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);
	//fprintf(f,"after newView\n");

	THCTensor_(fftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(fft3)(THCState *state, THCTensor *result, THCTensor *self) {
	if(THCTensor_(nDimension)(state, self) < 3)
		THError("tensor must at least have dimension 3\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	if (!THCTensor_(isSameSizeAs)(state, self, result))
    THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-3; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}

	THLongStorage *new_self_size = THLongStorage_newWithSize4( self_batch_dim, THCTensor_(size)(state, self, self_ndim-3),THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);

	THLongStorage *new_result_size = THLongStorage_newWithSize4( self_batch_dim, THCTensor_(size)(state, self, self_ndim-3),THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);

	THCTensor_(fftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(fftn)(THCState *state, THCTensor *result, THCTensor *self) {
	THCTensor_(fftnbase)(state, self, result, CUFFT_FORWARD);
	THCTensor_(mul)(state, result, result, ccx(1 / sqrt(THCTensor_(nElement)(state, result)),0));
}

void THCTensor_(ifft)(THCState *state, THCTensor *result, THCTensor *self) {
	if(THCTensor_(nDimension)(state, self) < 1)
		THError("tensor must at least have dimension 1\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	if (!THCTensor_(isSameSizeAs)(state, self, result))
    THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-1; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}

	THLongStorage *new_self_size = THLongStorage_newWithSize2( self_batch_dim, THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);

	THLongStorage *new_result_size = THLongStorage_newWithSize2( self_batch_dim, THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);

	THCTensor_(ifftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(ifft2)(THCState *state, THCTensor *result, THCTensor *self) {
	if(THCTensor_(nDimension)(state, self) < 2)
		THError("tensor must at least have dimension 2\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	if (!THCTensor_(isSameSizeAs)(state, self, result))
    THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-2; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}

	THLongStorage *new_self_size = THLongStorage_newWithSize3( self_batch_dim, THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);

	THLongStorage *new_result_size = THLongStorage_newWithSize3( self_batch_dim, THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);

	THCTensor_(ifftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(ifft3)(THCState *state, THCTensor *result, THCTensor *self) {
	if(THCTensor_(nDimension)(state, self) < 3)
		THError("tensor must at least have dimension 3\n");
	int self_ndim = THCTensor_(nDimension)(state, self);
	if (!THCTensor_(isSameSizeAs)(state, self, result))
    THError("self_ndim must be equal result_ndim\n");
	int self_batch_dim = 1;
	for(int i = 0; i< self_ndim-3; i++){
		self_batch_dim *= THCTensor_(size)(state, self, i);
	}

	THLongStorage *new_self_size = THLongStorage_newWithSize4( self_batch_dim, THCTensor_(size)(state, self, self_ndim-3),THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_self = THCTensor_(newView)(state, self, new_self_size);

	THLongStorage *new_result_size = THLongStorage_newWithSize4( self_batch_dim, THCTensor_(size)(state, self, self_ndim-3),THCTensor_(size)(state, self, self_ndim-2),THCTensor_(size)(state, self, self_ndim-1));
	THCTensor *new_result = THCTensor_(newView)(state, result, new_result_size);

	THCTensor_(ifftnBatched)(state,new_self,new_result);
	THLongStorage_free(new_self_size);
	THLongStorage_free(new_result_size);
	THCTensor_(free)(state,new_result);
	THCTensor_(free)(state,new_self);
}

void THCTensor_(ifftn)(THCState *state, THCTensor *result, THCTensor *self) {
	THCTensor_(fftnbase)(state, self, result, CUFFT_INVERSE);
	THCTensor_(mul)(state, result, result, ccx(1 / sqrt(THCTensor_(nElement)(state, result)),0));
}

#endif
#endif
