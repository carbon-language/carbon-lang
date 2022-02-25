//===--------- Mapping.h - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_MAPPING_H
#define OMPTARGET_MAPPING_H

#include "Types.h"

namespace _OMP {

namespace mapping {

#pragma omp declare target

inline constexpr uint32_t MaxThreadsPerTeam = 1024;

#pragma omp end declare target

/// Initialize the mapping machinery.
void init(bool IsSPMD);

/// Return true if the kernel is executed in SPMD mode.
bool isSPMDMode();

/// Return true if the kernel is executed in generic mode.
bool isGenericMode();

/// Return true if the executing thread is the main thread in generic mode.
/// These functions will lookup state and it is required that that is OK for the
/// thread and location. See also `isInitialThreadInLevel0` for a stateless
/// alternative for certain situations, e.g. during initialization.
bool isMainThreadInGenericMode();
bool isMainThreadInGenericMode(bool IsSPMD);

/// Return true if this thread is the initial thread in parallel level 0.
///
/// The thread for which this returns true should be used for single threaded
/// initialization tasks. We pick a special thread to ensure there are no
/// races between the initialization and the first read of initialized state.
bool isInitialThreadInLevel0(bool IsSPMD);

/// Return true if the executing thread has the lowest Id of the active threads
/// in the warp.
bool isLeaderInWarp();

/// Return a mask describing all active threads in the warp.
LaneMaskTy activemask();

/// Return a mask describing all threads with a smaller Id in the warp.
LaneMaskTy lanemaskLT();

/// Return a mask describing all threads with a larget Id in the warp.
LaneMaskTy lanemaskGT();

/// Return the thread Id in the warp, in [0, getWarpSize()).
uint32_t getThreadIdInWarp();

/// Return the thread Id in the block, in [0, getBlockSize()).
uint32_t getThreadIdInBlock();

/// Return the warp id in the block.
uint32_t getWarpId();

/// Return the warp size, thus number of threads in the warp.
uint32_t getWarpSize();

/// Return the number of warps in the block.
uint32_t getNumberOfWarpsInBlock();

/// Return the block Id in the kernel, in [0, getKernelSize()).
uint32_t getBlockId();

/// Return the block size, thus number of threads in the block.
///
/// Note: The version taking \p IsSPMD mode explicitly can be used during the
/// initialization of the target region, that is before `mapping::isSPMDMode()`
/// can be called by any thread other than the main one.
uint32_t getBlockSize();
uint32_t getBlockSize(bool IsSPMD);

/// Return the number of blocks in the kernel.
uint32_t getNumberOfBlocks();

/// Return the kernel size, thus number of threads in the kernel.
uint32_t getKernelSize();

/// Return the number of processing elements on the device.
uint32_t getNumberOfProcessorElements();

} // namespace mapping

} // namespace _OMP

#endif
