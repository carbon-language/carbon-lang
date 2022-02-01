//===--- cuda/dynamic_cuda/cuda.pp ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of cuda api by calling into cuda library via dlopen
// Does the dlopen/dlsym calls as part of the call to cuInit
//
//===----------------------------------------------------------------------===//

#include "cuda.h"
#include "Debug.h"
#include "dlwrap.h"

#include <string>
#include <unordered_map>

#include <dlfcn.h>

DLWRAP_INITIALIZE();

DLWRAP_INTERNAL(cuInit, 1);

DLWRAP(cuCtxGetDevice, 1);
DLWRAP(cuDeviceGet, 2);
DLWRAP(cuDeviceGetAttribute, 3);
DLWRAP(cuDeviceGetCount, 1);
DLWRAP(cuFuncGetAttribute, 3);

// Device info
DLWRAP(cuDeviceGetName, 3);
DLWRAP(cuDeviceTotalMem, 2);
DLWRAP(cuDriverGetVersion, 1);

DLWRAP(cuGetErrorString, 2);
DLWRAP(cuLaunchKernel, 11);

DLWRAP(cuMemAlloc, 2);
DLWRAP(cuMemAllocHost, 2);
DLWRAP(cuMemAllocManaged, 3);

DLWRAP(cuMemcpyDtoDAsync, 4);
DLWRAP(cuMemcpyDtoH, 3);
DLWRAP(cuMemcpyDtoHAsync, 4);
DLWRAP(cuMemcpyHtoD, 3);
DLWRAP(cuMemcpyHtoDAsync, 4);

DLWRAP(cuMemFree, 1);
DLWRAP(cuMemFreeHost, 1);
DLWRAP(cuModuleGetFunction, 3);
DLWRAP(cuModuleGetGlobal, 4);

DLWRAP(cuModuleUnload, 1);
DLWRAP(cuStreamCreate, 2);
DLWRAP(cuStreamDestroy, 1);
DLWRAP(cuStreamSynchronize, 1);
DLWRAP(cuCtxSetCurrent, 1);
DLWRAP(cuDevicePrimaryCtxRelease, 1);
DLWRAP(cuDevicePrimaryCtxGetState, 3);
DLWRAP(cuDevicePrimaryCtxSetFlags, 2);
DLWRAP(cuDevicePrimaryCtxRetain, 2);
DLWRAP(cuModuleLoadDataEx, 5);

DLWRAP(cuDeviceCanAccessPeer, 3);
DLWRAP(cuCtxEnablePeerAccess, 2);
DLWRAP(cuMemcpyPeerAsync, 6);

DLWRAP(cuCtxGetLimit, 2);
DLWRAP(cuCtxSetLimit, 2);

DLWRAP(cuEventCreate, 2);
DLWRAP(cuEventRecord, 2);
DLWRAP(cuStreamWaitEvent, 3);
DLWRAP(cuEventSynchronize, 1);
DLWRAP(cuEventDestroy, 1);

DLWRAP_FINALIZE();

#ifndef DYNAMIC_CUDA_PATH
#define DYNAMIC_CUDA_PATH "libcuda.so"
#endif

#define TARGET_NAME CUDA
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

static bool checkForCUDA() {
  // return true if dlopen succeeded and all functions found

  // Prefer _v2 versions of functions if found in the library
  std::unordered_map<std::string, const char *> TryFirst = {
      {"cuMemAlloc", "cuMemAlloc_v2"},
      {"cuMemFree", "cuMemFree_v2"},
      {"cuMemcpyDtoH", "cuMemcpyDtoH_v2"},
      {"cuMemcpyHtoD", "cuMemcpyHtoD_v2"},
      {"cuStreamDestroy", "cuStreamDestroy_v2"},
      {"cuModuleGetGlobal", "cuModuleGetGlobal_v2"},
      {"cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"},
      {"cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"},
      {"cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"},
      {"cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"},
      {"cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"},
  };

  const char *CudaLib = DYNAMIC_CUDA_PATH;
  void *DynlibHandle = dlopen(CudaLib, RTLD_NOW);
  if (!DynlibHandle) {
    DP("Unable to load library '%s': %s!\n", CudaLib, dlerror());
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);

    auto It = TryFirst.find(Sym);
    if (It != TryFirst.end()) {
      const char *First = It->second;
      void *P = dlsym(DynlibHandle, First);
      if (P) {
        DP("Implementing %s with dlsym(%s) -> %p\n", Sym, First, P);
        *dlwrap::pointer(I) = P;
        continue;
      }
    }

    void *P = dlsym(DynlibHandle, Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, CudaLib);
      return false;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);

    *dlwrap::pointer(I) = P;
  }

  return true;
}

CUresult cuInit(unsigned X) {
  // Note: Called exactly once from cuda rtl.cpp in a global constructor so
  // does not need to handle being called repeatedly or concurrently
  if (!checkForCUDA()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return dlwrap_cuInit(X);
}
