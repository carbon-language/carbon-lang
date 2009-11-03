//===-------------- llvm/CodeGen/ProcessImplicitDefs.h ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_PROCESSIMPLICITDEFS_H
#define LLVM_CODEGEN_PROCESSIMPLICITDEFS_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

  class MachineInstr;
  class TargetInstrInfo;

  /// Process IMPLICIT_DEF instructions and make sure there is one implicit_def
  /// for each use. Add isUndef marker to implicit_def defs and their uses.
  class ProcessImplicitDefs : public MachineFunctionPass {
  private:

    bool CanTurnIntoImplicitDef(MachineInstr *MI, unsigned Reg,
                                unsigned OpIdx, const TargetInstrInfo *tii_);

  public:
    static char ID;

    ProcessImplicitDefs() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    virtual bool runOnMachineFunction(MachineFunction &fn);
  };

}

#endif // LLVM_CODEGEN_PROCESSIMPLICITDEFS_H
