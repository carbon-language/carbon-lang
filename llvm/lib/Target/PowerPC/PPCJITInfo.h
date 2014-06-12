//===-- PPCJITInfo.h - PowerPC impl. of the JIT interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_JITINFO_H
#define POWERPC_JITINFO_H

#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
class PPCSubtarget;
class PPCJITInfo : public TargetJITInfo {
protected:
  PPCSubtarget &Subtarget;
  bool is64Bit;

public:
  PPCJITInfo(PPCSubtarget &STI);

  StubLayout getStubLayout() override;
  void *emitFunctionStub(const Function *F, void *Fn,
                         JITCodeEmitter &JCE) override;
  LazyResolverFn getLazyResolverFunction(JITCompilerFn) override;
  void relocate(void *Function, MachineRelocation *MR, unsigned NumRelocs,
                unsigned char *GOTBase) override;

  /// replaceMachineCodeForFunction - Make it so that calling the function
  /// whose machine code is at OLD turns into a call to NEW, perhaps by
  /// overwriting OLD with a branch to NEW.  This is used for self-modifying
  /// code.
  ///
  void replaceMachineCodeForFunction(void *Old, void *New) override;
};
}

#endif
