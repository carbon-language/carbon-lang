//===- SkeletonJITInfo.h - Skeleton impl of JIT interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the skeleton implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SKELETONJITINFO_H
#define SKELETONJITINFO_H

#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class TargetMachine;
  class IntrinsicLowering;

  class SkeletonJITInfo : public TargetJITInfo {
    TargetMachine &TM;
  public:
    SkeletonJITInfo(TargetMachine &tm) : TM(tm) {}

    /// addPassesToJITCompile - Add passes to the specified pass manager to
    /// implement a fast dynamic compiler for this target.  Return true if this
    /// is not supported for this target.
    ///
    virtual void addPassesToJITCompile(FunctionPassManager &PM);

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);
  };
}

#endif
