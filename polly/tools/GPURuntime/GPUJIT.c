/******************** GPUJIT.cpp - GPUJIT Execution Engine ********************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file implements GPUJIT, a ptx string execution engine for GPU.       */
/*                                                                            */
/******************************************************************************/

#include "GPUJIT.h"
#include <dlfcn.h>
#include <stdio.h>

/* Dynamic library handles for the CUDA and CUDA runtime library. */
static void *HandleCuda;
static void *HandleCudaRT;

/* Type-defines of function pointer to CUDA driver APIs. */
typedef CUresult CUDAAPI CuMemAllocFcnTy(CUdeviceptr *, size_t);
static CuMemAllocFcnTy *CuMemAllocFcnPtr;

typedef CUresult CUDAAPI CuFuncSetBlockShapeFcnTy(CUfunction, int, int, int);
static CuFuncSetBlockShapeFcnTy *CuFuncSetBlockShapeFcnPtr;

typedef CUresult CUDAAPI CuParamSetvFcnTy(CUfunction, int, void *,
                                          unsigned int);
static CuParamSetvFcnTy *CuParamSetvFcnPtr;

typedef CUresult CUDAAPI CuParamSetSizeFcnTy(CUfunction, unsigned int);
static CuParamSetSizeFcnTy *CuParamSetSizeFcnPtr;

typedef CUresult CUDAAPI CuLaunchGridFcnTy(CUfunction, int, int);
static CuLaunchGridFcnTy *CuLaunchGridFcnPtr;

typedef CUresult CUDAAPI CuMemcpyDtoHFcnTy(void *, CUdeviceptr, size_t);
static CuMemcpyDtoHFcnTy *CuMemcpyDtoHFcnPtr;

typedef CUresult CUDAAPI CuMemcpyHtoDFcnTy(CUdeviceptr, const void *, size_t);
static CuMemcpyHtoDFcnTy *CuMemcpyHtoDFcnPtr;

typedef CUresult CUDAAPI CuMemFreeFcnTy(CUdeviceptr);
static CuMemFreeFcnTy *CuMemFreeFcnPtr;

typedef CUresult CUDAAPI CuModuleUnloadFcnTy(CUmodule);
static CuModuleUnloadFcnTy *CuModuleUnloadFcnPtr;

typedef CUresult CUDAAPI CuCtxDestroyFcnTy(CUcontext);
static CuCtxDestroyFcnTy *CuCtxDestroyFcnPtr;

typedef CUresult CUDAAPI CuInitFcnTy(unsigned int);
static CuInitFcnTy *CuInitFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetCountFcnTy(int *);
static CuDeviceGetCountFcnTy *CuDeviceGetCountFcnPtr;

typedef CUresult CUDAAPI CuCtxCreateFcnTy(CUcontext *, unsigned int, CUdevice);
static CuCtxCreateFcnTy *CuCtxCreateFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetFcnTy(CUdevice *, int);
static CuDeviceGetFcnTy *CuDeviceGetFcnPtr;

typedef CUresult CUDAAPI CuModuleLoadDataExFcnTy(CUmodule *, const void *,
                                                 unsigned int, CUjit_option *,
                                                 void **);
static CuModuleLoadDataExFcnTy *CuModuleLoadDataExFcnPtr;

typedef CUresult CUDAAPI CuModuleGetFunctionFcnTy(CUfunction *, CUmodule,
                                                  const char *);
static CuModuleGetFunctionFcnTy *CuModuleGetFunctionFcnPtr;

typedef CUresult CUDAAPI CuDeviceComputeCapabilityFcnTy(int *, int *, CUdevice);
static CuDeviceComputeCapabilityFcnTy *CuDeviceComputeCapabilityFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetNameFcnTy(char *, int, CUdevice);
static CuDeviceGetNameFcnTy *CuDeviceGetNameFcnPtr;

/* Type-defines of function pointer ot CUDA runtime APIs. */
typedef cudaError_t CUDARTAPI CudaEventCreateFcnTy(cudaEvent_t *);
static CudaEventCreateFcnTy *CudaEventCreateFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventRecordFcnTy(cudaEvent_t,
                                                   cudaStream_t);
static CudaEventRecordFcnTy *CudaEventRecordFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventSynchronizeFcnTy(cudaEvent_t);
static CudaEventSynchronizeFcnTy *CudaEventSynchronizeFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventElapsedTimeFcnTy(float *, cudaEvent_t,
                                                        cudaEvent_t);
static CudaEventElapsedTimeFcnTy *CudaEventElapsedTimeFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventDestroyFcnTy(cudaEvent_t);
static CudaEventDestroyFcnTy *CudaEventDestroyFcnPtr;

typedef cudaError_t CUDARTAPI CudaThreadSynchronizeFcnTy(void);
static CudaThreadSynchronizeFcnTy *CudaThreadSynchronizeFcnPtr;

static void *getAPIHandle(void *Handle, const char *FuncName) {
  char *Err;
  void *FuncPtr;
  dlerror();
  FuncPtr = dlsym(Handle, FuncName);
  if ((Err = dlerror()) != 0) {
    fprintf(stdout, "Load CUDA driver API failed: %s. \n", Err);
    return 0;
  }
  return FuncPtr;
}

static int initialDeviceAPILibraries() {
  HandleCuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!HandleCuda) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  HandleCudaRT = dlopen("libcudart.so", RTLD_LAZY);
  if (!HandleCudaRT) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  return 1;
}

static int initialDeviceAPIs() {
  if (initialDeviceAPILibraries() == 0)
    return 0;

  /* Get function pointer to CUDA Driver APIs.
   *
   * Note that compilers conforming to the ISO C standard are required to
   * generate a warning if a conversion from a void * pointer to a function
   * pointer is attempted as in the following statements. The warning
   * of this kind of cast may not be emitted by clang and new versions of gcc
   * as it is valid on POSIX 2008.
   */
  CuFuncSetBlockShapeFcnPtr =
    (CuFuncSetBlockShapeFcnTy *) getAPIHandle(HandleCuda,
                                              "cuFuncSetBlockShape");

  CuParamSetvFcnPtr = (CuParamSetvFcnTy *) getAPIHandle(HandleCuda,
                                                        "cuParamSetv");

  CuParamSetSizeFcnPtr = (CuParamSetSizeFcnTy *) getAPIHandle(HandleCuda,
                                                              "cuParamSetSize");

  CuLaunchGridFcnPtr = (CuLaunchGridFcnTy *) getAPIHandle(HandleCuda,
                                                          "cuLaunchGrid");

  CuMemAllocFcnPtr = (CuMemAllocFcnTy *) getAPIHandle(HandleCuda,
                                                      "cuMemAlloc_v2");

  CuMemFreeFcnPtr = (CuMemFreeFcnTy *) getAPIHandle(HandleCuda, "cuMemFree_v2");

  CuMemcpyDtoHFcnPtr = (CuMemcpyDtoHFcnTy *) getAPIHandle(HandleCuda,
                                                          "cuMemcpyDtoH_v2");

  CuMemcpyHtoDFcnPtr = (CuMemcpyHtoDFcnTy *) getAPIHandle(HandleCuda,
                                                          "cuMemcpyHtoD_v2");

  CuModuleUnloadFcnPtr = (CuModuleUnloadFcnTy *) getAPIHandle(HandleCuda,
                                                              "cuModuleUnload");

  CuCtxDestroyFcnPtr = (CuCtxDestroyFcnTy *) getAPIHandle(HandleCuda,
                                                          "cuCtxDestroy");

  CuInitFcnPtr = (CuInitFcnTy *) getAPIHandle(HandleCuda, "cuInit");

  CuDeviceGetCountFcnPtr = (CuDeviceGetCountFcnTy *) getAPIHandle(HandleCuda,
                                                            "cuDeviceGetCount");

  CuDeviceGetFcnPtr = (CuDeviceGetFcnTy *) getAPIHandle(HandleCuda,
                                                        "cuDeviceGet");

  CuCtxCreateFcnPtr = (CuCtxCreateFcnTy *) getAPIHandle(HandleCuda,
                                                        "cuCtxCreate_v2");

  CuModuleLoadDataExFcnPtr =
    (CuModuleLoadDataExFcnTy *) getAPIHandle(HandleCuda, "cuModuleLoadDataEx");

  CuModuleGetFunctionFcnPtr =
    (CuModuleGetFunctionFcnTy *)getAPIHandle(HandleCuda, "cuModuleGetFunction");

  CuDeviceComputeCapabilityFcnPtr =
    (CuDeviceComputeCapabilityFcnTy *)getAPIHandle(HandleCuda,
                                                   "cuDeviceComputeCapability");

  CuDeviceGetNameFcnPtr =
    (CuDeviceGetNameFcnTy *) getAPIHandle(HandleCuda, "cuDeviceGetName");

  /* Get function pointer to CUDA Runtime APIs. */
  CudaEventCreateFcnPtr =
    (CudaEventCreateFcnTy *) getAPIHandle(HandleCudaRT, "cudaEventCreate");

  CudaEventRecordFcnPtr =
    (CudaEventRecordFcnTy *) getAPIHandle(HandleCudaRT, "cudaEventRecord");

  CudaEventSynchronizeFcnPtr =
    (CudaEventSynchronizeFcnTy *) getAPIHandle(HandleCudaRT,
                                               "cudaEventSynchronize");

  CudaEventElapsedTimeFcnPtr =
    (CudaEventElapsedTimeFcnTy *) getAPIHandle(HandleCudaRT,
                                               "cudaEventElapsedTime");

  CudaEventDestroyFcnPtr =
    (CudaEventDestroyFcnTy *) getAPIHandle(HandleCudaRT, "cudaEventDestroy");

  CudaThreadSynchronizeFcnPtr =
    (CudaThreadSynchronizeFcnTy *) getAPIHandle(HandleCudaRT,
                                                "cudaThreadSynchronize");

  return 1;
}

void polly_initDevice(CUcontext *Context, CUdevice *Device) {
  int Major = 0, Minor = 0, DeviceID = 0;
  char DeviceName[256];
  int DeviceCount = 0;

  /* Get API handles. */
  if (initialDeviceAPIs() == 0) {
    fprintf(stdout, "Getting the \"handle\" for the CUDA driver API failed.\n");
    exit(-1);
  }

  if (CuInitFcnPtr(0) != CUDA_SUCCESS) {
    fprintf(stdout, "Initializing the CUDA driver API failed.\n");
    exit(-1);
  }

  /* Get number of devices that supports CUDA. */
  CuDeviceGetCountFcnPtr(&DeviceCount);
  if (DeviceCount == 0) {
    fprintf(stdout, "There is no device supporting CUDA.\n");
    exit(-1);
  }

  /* We select the 1st device as default. */
  CuDeviceGetFcnPtr(Device, 0);

  /* Get compute capabilities and the device name. */
  CuDeviceComputeCapabilityFcnPtr(&Major, &Minor, *Device);
  CuDeviceGetNameFcnPtr(DeviceName, 256, *Device);
  fprintf(stderr, "> Running on GPU device %d : %s.\n", DeviceID, DeviceName);

  /* Create context on the device. */
  CuCtxCreateFcnPtr(Context, 0, *Device);
}

void polly_getPTXModule(void *PTXBuffer, CUmodule *Module) {
  if(CuModuleLoadDataExFcnPtr(Module, PTXBuffer, 0, 0, 0) != CUDA_SUCCESS) {
    fprintf(stdout, "Loading ptx assembly text failed.\n");
    exit(-1);
  }
}

void polly_getPTXKernelEntry(const char *KernelName, CUmodule *Module,
                             CUfunction *Kernel) {
  /* Locate the kernel entry point. */
  if(CuModuleGetFunctionFcnPtr(Kernel, *Module, KernelName)
     !=  CUDA_SUCCESS) {
    fprintf(stdout, "Loading kernel function failed.\n");
    exit(-1);
  }
}

void polly_startTimerByCudaEvent(cudaEvent_t *StartTimer,
                                 cudaEvent_t *StopTimer) {
  CudaEventCreateFcnPtr(StartTimer);
  CudaEventCreateFcnPtr(StopTimer);
  CudaEventRecordFcnPtr(*StartTimer, 0);
}

void polly_stopTimerByCudaEvent(cudaEvent_t *StartTimer,
                                cudaEvent_t *StopTimer, float *ElapsedTimes) {
  CudaEventRecordFcnPtr(*StopTimer, 0);
  CudaEventSynchronizeFcnPtr(*StopTimer);
  CudaEventElapsedTimeFcnPtr(ElapsedTimes, *StartTimer, *StopTimer );
  CudaEventDestroyFcnPtr(*StartTimer);
  CudaEventDestroyFcnPtr(*StopTimer);
  fprintf(stderr, "Processing time: %f (ms).\n", *ElapsedTimes);
}

void polly_allocateMemoryForHostAndDevice(void **PtrHostData,
                                          CUdeviceptr *PtrDevData,
                                          int MemSize) {
  if ((*PtrHostData = (int *)malloc(MemSize)) == 0) {
    fprintf(stdout, "Could not allocate host memory.\n");
    exit(-1);
  }
  CuMemAllocFcnPtr(PtrDevData, MemSize);
}

void polly_copyFromHostToDevice(CUdeviceptr DevData, void *HostData,
                                int MemSize) {
  CuMemcpyHtoDFcnPtr(DevData, HostData, MemSize);
}

void polly_copyFromDeviceToHost(void *HostData, CUdeviceptr DevData,
                                int MemSize) {
  if(CuMemcpyDtoHFcnPtr(HostData, DevData, MemSize) != CUDA_SUCCESS) {
    fprintf(stdout, "Copying results from device to host memory failed.\n");
    exit(-1);
  }
}

void polly_setKernelParameters(CUfunction *Kernel, int BlockWidth,
                               int BlockHeight, CUdeviceptr DevData) {
  int ParamOffset = 0;
  CuFuncSetBlockShapeFcnPtr(*Kernel, BlockWidth, BlockHeight, 1);
  CuParamSetvFcnPtr(*Kernel, ParamOffset, &DevData, sizeof(DevData));
  ParamOffset += sizeof(DevData);
  CuParamSetSizeFcnPtr(*Kernel, ParamOffset);
}

void polly_launchKernel(CUfunction *Kernel, int GridWidth, int GridHeight) {
  if (CuLaunchGridFcnPtr(*Kernel, GridWidth, GridHeight) != CUDA_SUCCESS) {
    fprintf(stdout, "Launching CUDA kernel failed.\n");
    exit(-1);
  }
  CudaThreadSynchronizeFcnPtr();
  fprintf(stdout, "CUDA kernel launched.\n");
}

void polly_cleanupGPGPUResources(void *HostData, CUdeviceptr DevData,
                                 CUmodule *Module, CUcontext *Context) {
  if (HostData) {
    free(HostData);
    HostData = 0;
  }

  if (DevData) {
    CuMemFreeFcnPtr(DevData);
    DevData = 0;
  }

  if (*Module) {
    CuModuleUnloadFcnPtr(*Module);
    *Module = 0;
  }

  if (*Context) {
    CuCtxDestroyFcnPtr(*Context);
    *Context = 0;
  }

  dlclose(HandleCuda);
  dlclose(HandleCudaRT);
}
