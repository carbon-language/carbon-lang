//===-- Instruction.cpp - Implement the Instruction class --------*- C++ -*--=//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"
#include "Support/LeakDetector.h"

Instruction::Instruction(const Type *ty, unsigned it, const std::string &Name,
                         Instruction *InsertBefore)
  : User(ty, Value::InstructionVal, Name) {
  Parent = 0;
  iType = it;

  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    assert(InsertBefore->getParent() &&
           "Instruction to insert before is not in a basic block!");
    InsertBefore->getParent()->getInstList().insert(InsertBefore, this);
  }
}

void Instruction::setParent(BasicBlock *P) {
  if (getParent())
    LeakDetector::addGarbageObject(this);

  Parent = P;

  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

// Specialize setName to take care of symbol table majik
void Instruction::setName(const std::string &name, SymbolTable *ST) {
  BasicBlock *P = 0; Function *PP = 0;
  assert((ST == 0 || !getParent() || !getParent()->getParent() || 
	  ST == &getParent()->getParent()->getSymbolTable()) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && (PP = P->getParent()) && hasName())
    PP->getSymbolTable().remove(this);
  Value::setName(name);
  if (PP && hasName()) PP->getSymbolTable().insert(this);
}


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case Invoke: return "invoke";
  case Unwind: return "unwind";
    
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
  case VarArg:  return "va_arg";

  default: return "<Invalid operator> ";
  }
  
  return 0;
}


/// isAssociative - Return true if the instruction is associative:
///
///   Associative operators satisfy:  x op (y op z) === (x op y) op z)
///
/// In LLVM, the Add, Mul, And, Or, and Xor operators are associative, when not
/// applied to floating point types.
///
bool Instruction::isAssociative(unsigned Opcode, const Type *Ty) {
  if (Opcode == Add || Opcode == Mul ||
      Opcode == And || Opcode == Or || Opcode == Xor) {
    // Floating point operations do not associate!
    return !Ty->isFloatingPoint();
  }
  return 0;
}

/// isCommutative - Return true if the instruction is commutative:
///
///   Commutative operators satistify: (x op y) === (y op x)
///
/// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
/// applied to any type.
///
bool Instruction::isCommutative(unsigned op) {
  switch (op) {
  case Add:
  case Mul:
  case And: 
  case Or:
  case Xor:
  case SetEQ:
  case SetNE:
    return true;
  default:
    return false;
  }
}


/// isTrappingInstruction - Return true if the instruction may trap.
///
bool Instruction::isTrapping(unsigned op) {
  switch(op) {
  case Div:
  case Rem:
  case Load:
  case Store:
  case Call:
  case Invoke:
    return true;
  default:
    return false;
  }
}
