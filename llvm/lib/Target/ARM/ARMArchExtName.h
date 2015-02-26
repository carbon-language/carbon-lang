//===-- ARMArchExtName.h - List of the ARM Extension names ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMARCHEXTNAME_H
#define LLVM_LIB_TARGET_ARM_ARMARCHEXTNAME_H

namespace llvm {
namespace ARM {

enum ArchExtKind {
  INVALID_ARCHEXT = 0

#define ARM_ARCHEXT_NAME(NAME, ID) , ID
#include "ARMArchExtName.def"
};

} // namespace ARM
} // namespace llvm

#endif
