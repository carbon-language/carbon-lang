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


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case Invoke: return "invoke";
    
  // Standard binary operators...
  case Add: return "add";
  case Sub: return "sub";
  case Mul: return "mul";
  case Div: return "div";
  case Rem: return "rem";

  // Logical operators...
  case And: return "and";
  case Or : return "or";
  case Xor: return "xor";

  // SetCC operators...
  case SetLE:  return "setle";
  case SetGE:  return "setge";
  case SetLT:  return "setlt";
  case SetGT:  return "setgt";
  case SetEQ:  return "seteq";
  case SetNE:  return "setne";
    
  // Memory instructions...
  case Malloc:        return "malloc";
  case Free:          return "free";
  case Alloca:        return "alloca";
  case Load:          return "load";
  case Store:         return "store";
  case GetElementPtr: return "getelementptr";
    
  // Other instructions...
  case PHINode: return "phi";
  case Cast:    return "cast";
  case Call:    return "call";
  case Shl:     return "shl";
  case Shr:     return "shr";
    
  default: return "<Invalid operator> ";
  }
  
  return 0;
}
