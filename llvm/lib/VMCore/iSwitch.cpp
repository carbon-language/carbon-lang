//===-- iSwitch.cpp - Implement the Switch instruction -----------*- C++ -*--=//
//
// This file implements the Switch instruction...
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"
#include "llvm/BasicBlock.h"

SwitchInst::SwitchInst(Value *V, BasicBlock *DefDest) 
  : TerminatorInst(Instruction::Switch) {
  assert(V && DefDest);
  Operands.push_back(Use(V, this));
  Operands.push_back(Use(DefDest, this));
}

SwitchInst::SwitchInst(const SwitchInst &SI) 
  : TerminatorInst(Instruction::Switch) {
  Operands.reserve(SI.Operands.size());

  for (unsigned i = 0, E = SI.Operands.size(); i != E; i+=2) {
    Operands.push_back(Use(SI.Operands[i], this));
    Operands.push_back(Use(SI.Operands[i+1], this));
  }
}

void SwitchInst::dest_push_back(Constant *OnVal, BasicBlock *Dest) {
  Operands.push_back(Use((Value*)OnVal, this));
  Operands.push_back(Use((Value*)Dest, this));
}
