//===---- MipsOs16.h for Mips Option -Os16                         --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an optimization phase for the MIPS target.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsTargetMachine.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"



#ifndef MIPSOS16_H
#define MIPSOS16_H

using namespace llvm;

namespace llvm {

class MipsOs16 : public ModulePass {

public:
  static char ID;

  MipsOs16() : ModulePass(ID) {

  }

  const char *getPassName() const override {
    return "MIPS Os16 Optimization";
  }

  bool runOnModule(Module &M) override;

};

ModulePass *createMipsOs16(MipsTargetMachine &TM);

}

#endif
