//===- PowerPCJITInfo.h - PowerPC impl. of the JIT interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_JITINFO_H
#define POWERPC_JITINFO_H

#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class TargetMachine;

  class PowerPCJITInfo : public TargetJITInfo {
  protected:
    TargetMachine &TM;
  public:
    PowerPCJITInfo(TargetMachine &tm) : TM(tm) {}

    /// addPassesToJITCompile - Add passes to the specified pass manager to
    /// implement a fast dynamic compiler for this target.  Return true if this
    /// is not supported for this target.
    ///
    virtual void addPassesToJITCompile(FunctionPassManager &PM);
  };
}

#endif
