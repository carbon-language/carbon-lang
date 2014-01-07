//===-- AArch64TargetObjectFile.h - AArch64 Object Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file deals with any AArch64 specific requirements on object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AARCH64_TARGETOBJECTFILE_H
#define LLVM_TARGET_AARCH64_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

  /// AArch64LinuxTargetObjectFile - This implementation is used for linux
  /// AArch64.
  class AArch64LinuxTargetObjectFile : public TargetLoweringObjectFileELF {
    virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  };

} // end namespace llvm

#endif
