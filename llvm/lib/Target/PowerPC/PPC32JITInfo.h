//===- PPC32JITInfo.h - PowerPC/Darwin JIT interface --------*- C++ -*-===//
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

#ifndef POWERPC_DARWIN_JITINFO_H
#define POWERPC_DARWIN_JITINFO_H

#include "PowerPCJITInfo.h"

namespace llvm {
  class TargetMachine;
  class IntrinsicLowering;

  class PPC32JITInfo : public PowerPCJITInfo {
  public:
    PPC32JITInfo(TargetMachine &tm) : PowerPCJITInfo(tm) {}

    virtual void *emitFunctionStub(void *Fn, MachineCodeEmitter &MCE);
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);
  };
}

#endif
