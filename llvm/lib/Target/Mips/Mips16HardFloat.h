//===---- Mips16HardFloat.h for Mips16 Hard Float                  --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a phase which implements part of the floating point
// interoperability between Mips16 and Mips32 code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPS16HARDFLOAT_H
#define LLVM_LIB_TARGET_MIPS_MIPS16HARDFLOAT_H

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsTargetMachine.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace llvm {

class Mips16HardFloat : public ModulePass {

public:
  static char ID;

  Mips16HardFloat(MipsTargetMachine &TM_) : ModulePass(ID),
    TM(TM_), Subtarget(TM.getSubtarget<MipsSubtarget>()) {
  }

  const char *getPassName() const override {
    return "MIPS16 Hard Float Pass";
  }

  bool runOnModule(Module &M) override;

protected:
  /// Keep a pointer to the MipsSubtarget around so that we can make the right
  /// decision when generating code for different targets.
  const TargetMachine &TM;
  const MipsSubtarget &Subtarget;

};

ModulePass *createMips16HardFloat(MipsTargetMachine &TM);

}
#endif
