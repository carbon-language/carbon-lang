//===-- iReturn.cpp - Implement the Return instruction -----------*- C++ -*--=//
//
// This file implements the Return instruction...
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"

ReturnInst::ReturnInst(Value *V)
  : TerminatorInst(Instruction::Ret), Val(V, this) {
}

ReturnInst::ReturnInst(const ReturnInst &RI)
  : TerminatorInst(Instruction::Ret), Val(RI.Val, this) {
}

void ReturnInst::dropAllReferences() {
  Val = 0;
}

bool ReturnInst::setOperand(unsigned i, Value *V) { 
  if (i) return false; 
  Val = V;
  return true;
}
