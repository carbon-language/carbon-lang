//===-- Instruction.cpp - Implement the Instruction class --------*- C++ -*--=//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"

Instruction::Instruction(const Type *ty, unsigned it, const std::string &Name) 
  : User(ty, Value::InstructionVal, Name) {
  Parent = 0;
  iType = it;
}

// Specialize setName to take care of symbol table majik
void Instruction::setName(const std::string &name, SymbolTable *ST) {
  BasicBlock *P = 0; Function *PP = 0;
  assert((ST == 0 || !getParent() || !getParent()->getParent() || 
	  ST == getParent()->getParent()->getSymbolTable()) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && (PP = P->getParent()) && hasName())
    PP->getSymbolTable()->remove(this);
  Value::setName(name);
  if (PP && hasName()) PP->getSymbolTableSure()->insert(this);
}
