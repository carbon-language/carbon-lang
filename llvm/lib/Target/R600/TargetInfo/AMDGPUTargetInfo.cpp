//===-- TargetInfo/AMDGPUTargetInfo.cpp - TargetInfo for AMDGPU -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

/// \brief The target for the AMDGPU backend
Target llvm::TheAMDGPUTarget;

/// \brief Extern function to initialize the targets for the AMDGPU backend
extern "C" void LLVMInitializeR600TargetInfo() {
  RegisterTarget<Triple::r600, false>
    R600(TheAMDGPUTarget, "r600", "AMD GPUs HD2XXX-HD6XXX");
}
