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
#include <algorithm>

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
  
  for (unsigned i = 0; i < PN.IncomingValues.size(); i++)
    IncomingValues.push_back(Use(PN.IncomingValues[i], this));
}

void PHINode::dropAllReferences() {
  IncomingValues.clear();
}

bool PHINode::setOperand(unsigned i, Value *Val) {
  assert(Val && "PHI node must only reference nonnull definitions!");
  if (i >= IncomingValues.size()) return false;

  IncomingValues[i] = Val;
  return true;
}

void PHINode::addIncoming(Value *D) {
  IncomingValues.push_back(Use(D, this));
}

