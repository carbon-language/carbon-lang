//===-- iSwitch.cpp - Implement the Switch instruction -----------*- C++ -*--=//
//
// This file implements the Switch instruction...
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"
#include "llvm/BasicBlock.h"
#ifndef NDEBUG
#include "llvm/Type.h"
#endif

SwitchInst::SwitchInst(Value *V, BasicBlock *DefV) 
  : TerminatorInst(Instruction::Switch), 
    DefaultDest(DefV, this), Val(V, this) {
  assert(Val && DefV);
}

SwitchInst::SwitchInst(const SwitchInst &SI) 
  : TerminatorInst(Instruction::Switch), DefaultDest(SI.DefaultDest), 
    Val(SI.Val) {

  for (dest_const_iterator I = SI.Destinations.begin(), 
	                 end = SI.Destinations.end(); I != end; ++I)
    Destinations.push_back(dest_value(ConstPoolUse(I->first, this), 
				      BasicBlockUse(I->second, this)));
}


void SwitchInst::dest_push_back(ConstPoolVal *OnVal, BasicBlock *Dest) {
  Destinations.push_back(dest_value(ConstPoolUse(OnVal, this), 
                                    BasicBlockUse(Dest, this)));
}

void SwitchInst::dropAllReferences() {
  Val = 0;
  DefaultDest = 0;
  Destinations.clear();
}

const BasicBlock *SwitchInst::getSuccessor(unsigned idx) const {
  if (idx == 0) return DefaultDest;
  if (idx > Destinations.size()) return 0;
  return Destinations[idx-1].second;
}

unsigned SwitchInst::getNumOperands() const {
  return 2+Destinations.size();
}

const Value *SwitchInst::getOperand(unsigned i) const {
  if (i == 0) return Val;
  else if (i == 1) return DefaultDest;

  unsigned slot = (i-2) >> 1;
  if (slot >= Destinations.size()) return 0;
  
  if (i & 1) return Destinations[slot].second;
  return Destinations[slot].first;
}

bool SwitchInst::setOperand(unsigned i, Value *V) {
  if (i == 0) { Val = V; return true; }
  else if (i == 1) { 
    assert(V->getType() == Type::LabelTy); 
    DefaultDest = (BasicBlock*)V;
    return true; 
  }

  unsigned slot = (i-2) >> 1;
  if (slot >= Destinations.size()) return 0;

  if (i & 1) {
    assert(V->getType() == Type::LabelTy);
    Destinations[slot].second = (BasicBlock*)V;
  } else {
    // TODO: assert constant
    Destinations[slot].first = (ConstPoolVal*)V;
  }
  return true;
}
