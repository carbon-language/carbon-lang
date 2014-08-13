//===-- ARMArchName.h - List of the ARM arch names --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMARCHNAME_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMARCHNAME_H

namespace llvm {
namespace ARM {

enum ArchKind {
  INVALID_ARCH = 0

#define ARM_ARCH_NAME(NAME, ID, DEFAULT_CPU_NAME, DEFAULT_CPU_ARCH) , ID
#define ARM_ARCH_ALIAS(NAME, ID) /* empty */
#include "ARMArchName.def"
};

} // namespace ARM
} // namespace llvm

#endif
