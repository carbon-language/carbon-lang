//===- ARMJITInfo.h - ARM implementation of the JIT interface  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMJITINFO_H
#define ARMJITINFO_H

#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
  class ARMTargetMachine;

  class ARMJITInfo : public TargetJITInfo {
    ARMTargetMachine &TM;
  public:
    explicit ARMJITInfo(ARMTargetMachine &tm) : TM(tm) {useGOT = 0;}

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);

    /// emitFunctionStub - Use the specified MachineCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE);

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);
  };
}

#endif
