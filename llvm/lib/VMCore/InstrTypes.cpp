//===-- InstrTypes.cpp - Implement Instruction subclasses --------*- C++ -*--=//
//
// This file implements 
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"
#include <algorithm>  // find

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

TerminatorInst::TerminatorInst(unsigned iType) 
  : Instruction(Type::VoidTy, iType, "") {
}


//===----------------------------------------------------------------------===//
//                            MethodArgument Class
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void MethodArgument::setName(const string &name) {
  Method *P;
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable()->insert(this);
}


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const Type *Ty, const string &name) 
  : Instruction(Ty, Instruction::PHINode, name) {
}

PHINode::PHINode(const PHINode &PN) 
  : Instruction(PN.getType(), Instruction::PHINode) {
  Operands.reserve(PN.Operands.size());
  for (unsigned i = 0; i < PN.Operands.size(); i+=2) {
    Operands.push_back(Use(PN.Operands[i], this));
    Operands.push_back(Use(PN.Operands[i+1], this));
  }
}

void PHINode::addIncoming(Value *D, BasicBlock *BB) {
  Operands.push_back(Use(D, this));
  Operands.push_back(Use(BB, this));
}

// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(const BasicBlock *BB) {
  op_iterator Idx = find(Operands.begin(), Operands.end(), (const Value*)BB);
  assert(Idx != Operands.end() && "BB not in PHI node!");
  --Idx;  // Back up to value prior to Basic block
  Value *Removed = *Idx;
  Operands.erase(Idx, Idx+2);  // Erase Value and BasicBlock
  return Removed;
}
