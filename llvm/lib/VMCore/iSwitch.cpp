//===-- iSwitch.cpp - Implement the Switch instruction --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Switch instruction...
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"
#include "llvm/BasicBlock.h"
using namespace llvm;

SwitchInst::SwitchInst(Value *V, BasicBlock *DefaultDest,
                       Instruction *InsertBefore) 
  : TerminatorInst(Instruction::Switch, InsertBefore) {
  assert(V && DefaultDest);
  Operands.push_back(Use(V, this));
  Operands.push_back(Use(DefaultDest, this));
}

SwitchInst::SwitchInst(Value *V, BasicBlock *DefaultDest,
                       BasicBlock *InsertAtEnd) 
  : TerminatorInst(Instruction::Switch, InsertAtEnd) {
  assert(V && DefaultDest);
  Operands.push_back(Use(V, this));
  Operands.push_back(Use(DefaultDest, this));
}

SwitchInst::SwitchInst(const SwitchInst &SI) 
  : TerminatorInst(Instruction::Switch) {
  Operands.reserve(SI.Operands.size());

  for (unsigned i = 0, E = SI.Operands.size(); i != E; i+=2) {
    Operands.push_back(Use(SI.Operands[i], this));
    Operands.push_back(Use(SI.Operands[i+1], this));
  }
}

/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(Constant *OnVal, BasicBlock *Dest) {
  Operands.push_back(Use((Value*)OnVal, this));
  Operands.push_back(Use((Value*)Dest, this));
}

/// removeCase - This method removes the specified successor from the switch
/// instruction.  Note that this cannot be used to remove the default
/// destination (successor #0).
///
void SwitchInst::removeCase(unsigned idx) {
  assert(idx != 0 && "Cannot remove the default case!");
  assert(idx*2 < Operands.size() && "Successor index out of range!!!");
  Operands.erase(Operands.begin()+idx*2, Operands.begin()+(idx+1)*2);  
}
