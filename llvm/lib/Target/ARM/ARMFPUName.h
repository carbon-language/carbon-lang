//===-- ARMFPUName.h - List of the ARM FPU names ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMFPUNAME_H
#define LLVM_LIB_TARGET_ARM_ARMFPUNAME_H

namespace llvm {
namespace ARM {

enum FPUKind {
  INVALID_FPU = 0

#define ARM_FPU_NAME(NAME, ID) , ID
#include "ARMFPUName.def"
};

} // namespace ARM
} // namespace llvm

#endif
