/******************************************************************************/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file defines GPUJIT.                                                 */
/*                                                                            */
/******************************************************************************/

#ifndef GPUJIT_H_
#define GPUJIT_H_
#include "stddef.h"

/*
 * The following demostrates how we can use the GPURuntime library to
 * execute a GPU kernel.
 *
 * char KernelString[] = "\n\
 *   .version 1.4\n\
 *   .target sm_10, map_f64_to_f32\n\
 *   .entry _Z8myKernelPi (\n\
 *   .param .u64 __cudaparm__Z8myKernelPi_data)\n\
 *   {\n\
 *     .reg .u16 %rh<4>;\n\
 *     .reg .u32 %r<5>;\n\
 *     .reg .u64 %rd<6>;\n\
 *     cvt.u32.u16     %r1, %tid.x;\n\
 *     mov.u16         %rh1, %ctaid.x;\n\
 *     mov.u16         %rh2, %ntid.x;\n\
 *     mul.wide.u16    %r2, %rh1, %rh2;\n\
 *     add.u32         %r3, %r1, %r2;\n\
 *     ld.param.u64    %rd1, [__cudaparm__Z8myKernelPi_data];\n\
 *     cvt.s64.s32     %rd2, %r3;\n\
 *     mul.wide.s32    %rd3, %r3, 4;\n\
 *     add.u64         %rd4, %rd1, %rd3;\n\
 *     st.global.s32   [%rd4+0], %r3;\n\
 *     exit;\n\
 *   }\n\
 * ";
 *
 * const char *Entry = "_Z8myKernelPi";
 *
 * int main() {
 *   PollyGPUFunction *Kernel;
 *   PollyGPUContext *Context;
 *   PollyGPUDevicePtr *DevArray;
 *   int *HostData;
 *   int MemSize;
 *
 *   int GridX = 8;
 *   int GridY = 8;
 *
 *   int BlockX = 16;
 *   int BlockY = 16;
 *   int BlockZ = 1;
 *
 *   MemSize = 256*64*sizeof(int);
 *   Context = polly_initContext();
 *   DevArray = polly_allocateMemoryForDevice(MemSize);
 *   Kernel = polly_getKernel(KernelString, KernelName);
 *
 *   void *Params[1];
 *   void *DevPtr = polly_getDevicePtr(DevArray)
 *   Params[0] = &DevPtr;
 *
 *   polly_launchKernel(Kernel, GridX, GridY, BlockX, BlockY, BlockZ, Params);
 *
 *   polly_copyFromDeviceToHost(HostData, DevData, MemSize);
 *   polly_freeKernel(Kernel);
 *   polly_freeDeviceMemory(DevArray);
 *   polly_freeContext(Context);
 * }
 *
 */

typedef enum PollyGPURuntimeT {
  RUNTIME_NONE,
  RUNTIME_CUDA,
  RUNTIME_CL
} PollyGPURuntime;

typedef struct PollyGPUContextT PollyGPUContext;
typedef struct PollyGPUFunctionT PollyGPUFunction;
typedef struct PollyGPUDevicePtrT PollyGPUDevicePtr;

typedef struct OpenCLContextT OpenCLContext;
typedef struct OpenCLKernelT OpenCLKernel;
typedef struct OpenCLDevicePtrT OpenCLDevicePtr;

typedef struct CUDAContextT CUDAContext;
typedef struct CUDAKernelT CUDAKernel;
typedef struct CUDADevicePtrT CUDADevicePtr;

PollyGPUContext *polly_initContextCUDA();
PollyGPUContext *polly_initContextCL();
PollyGPUFunction *polly_getKernel(const char *BinaryBuffer,
                                  const char *KernelName);
void polly_freeKernel(PollyGPUFunction *Kernel);
void polly_copyFromHostToDevice(void *HostData, PollyGPUDevicePtr *DevData,
                                long MemSize);
void polly_copyFromDeviceToHost(PollyGPUDevicePtr *DevData, void *HostData,
                                long MemSize);
void polly_synchronizeDevice();
void polly_launchKernel(PollyGPUFunction *Kernel, unsigned int GridDimX,
                        unsigned int GridDimY, unsigned int BlockSizeX,
                        unsigned int BlockSizeY, unsigned int BlockSizeZ,
                        void **Parameters);
void polly_freeDeviceMemory(PollyGPUDevicePtr *Allocation);
void polly_freeContext(PollyGPUContext *Context);

// Note that polly_{malloc/free}Managed are currently not used by Polly.
// We use them in COSMO by replacing all malloc with polly_mallocManaged and all
// frees with cudaFree, so we can get managed memory "automatically".
// Needless to say, this is a hack.
// Please make sure that this code is not present in Polly when 2018 rolls in.
// If this is still present, ping Siddharth Bhat <siddu.druid@gmail.com>
void *polly_mallocManaged(size_t size);
void polly_freeManaged(void *mem);
#endif /* GPUJIT_H_ */
