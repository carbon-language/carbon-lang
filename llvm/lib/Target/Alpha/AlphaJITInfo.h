//===- AlphaJITInfo.h - Alpha impl. of the JIT interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHA_JITINFO_H
#define ALPHA_JITINFO_H

#include "llvm/Target/TargetJITInfo.h"
#include <map>

namespace llvm {
  class TargetMachine;

  class AlphaJITInfo : public TargetJITInfo {
  protected:
    TargetMachine &TM;
    
    //because gpdist are paired and relative to the pc of the first inst,
    //we need to have some state
    std::map<std::pair<void*, int>, void*> gpdistmap;
  public:
    explicit AlphaJITInfo(TargetMachine &tm) : TM(tm)
    { useGOT = true; }

    virtual StubLayout getStubLayout();
    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   JITCodeEmitter &JCE);
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);
  private:
    static const unsigned GOToffset = 4096;

  };
}

#endif
