/******************************************************************************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file defines GPUJIT.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef GPUJIT_H_
#define GPUJIT_H_

#include <cuda.h>
#include <cuda_runtime.h>

void polly_initDevice(CUcontext *Context, CUdevice *Device);
void polly_getPTXModule(void *PTXBuffer, CUmodule *Module);
void polly_getPTXKernelEntry(const char *KernelName,
                             CUmodule *Module,
                             CUfunction *Kernel);
void polly_startTimerByCudaEvent(cudaEvent_t *StartTimer,
                                 cudaEvent_t *StopTimer);
void polly_stopTimerByCudaEvent(cudaEvent_t *StartTimer, cudaEvent_t *StopTimer,
                                float *ElapsedTimes);
void polly_copyFromHostToDevice(CUdeviceptr DevData, void *HostData,
                                int MemSize);
void polly_copyFromDeviceToHost(void *HostData, CUdeviceptr DevData,
                                int MemSize);
void polly_allocateMemoryForHostAndDevice(void **PtrHostData,
                                          CUdeviceptr *PtrDevData,
                                          int MemSize);
void polly_setKernelParameters(CUfunction *Kernel, int BlockWidth,
                               int BlockHeight, CUdeviceptr DevData);
void polly_launchKernel(CUfunction *Kernel, int GridWidth, int GridHeight);
void polly_cleanupGPGPUResources(void *HostData, CUdeviceptr DevData,
                                 CUmodule *Module, CUcontext *Context);
#endif /* GPUJIT_H_ */
