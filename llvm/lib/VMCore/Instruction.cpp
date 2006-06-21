//===-- Instruction.cpp - Implement the Instruction class -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

Instruction::Instruction(const Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         const std::string &Name, Instruction *InsertBefore)
  : User(ty, Value::InstructionVal + it, Ops, NumOps, Name), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    assert(InsertBefore->getParent() &&
           "Instruction to insert before is not in a basic block!");
    InsertBefore->getParent()->getInstList().insert(InsertBefore, this);
  }
}

Instruction::Instruction(const Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         const std::string &Name, BasicBlock *InsertAtEnd)
  : User(ty, Value::InstructionVal + it, Ops, NumOps, Name), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // append this instruction into the basic block
  assert(InsertAtEnd && "Basic block to append to may not be NULL!");
  InsertAtEnd->getInstList().push_back(this);
}

// Out of line virtual method, so the vtable, etc has a home.
Instruction::~Instruction() {
  assert(Parent == 0 && "Instruction still linked in the program!");
}


void Instruction::setOpcode(unsigned opc) {
  setValueType(Value::InstructionVal + opc);
}

void Instruction::setParent(BasicBlock *P) {
  if (getParent()) {
    if (!P) LeakDetector::addGarbageObject(this);
  } else {
    if (P) LeakDetector::removeGarbageObject(this);
  }

  Parent = P;
}

void Instruction::removeFromParent() {
  getParent()->getInstList().remove(this);
}

void Instruction::eraseFromParent() {
  getParent()->getInstList().erase(this);
}

/// moveBefore - Unlink this instruction from its current basic block and
/// insert it into the basic block that MovePos lives in, right before
/// MovePos.
void Instruction::moveBefore(Instruction *MovePos) {
  MovePos->getParent()->getInstList().splice(MovePos,getParent()->getInstList(),
                                             this);
}


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case Invoke: return "invoke";
  case Unwind: return "unwind";
  case Unreachable: return "unreachable";

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
  case PHI:     return "phi";
  case Cast:    return "cast";
  case Select:  return "select";
  case Call:    return "call";
  case Shl:     return "shl";
  case Shr:     return "shr";
  case VAArg:   return "va_arg";
  case ExtractElement: return "extractelement";
  case InsertElement: return "insertelement";
  case ShuffleVector: return "shufflevector";

  default: return "<Invalid operator> ";
  }

  return 0;
}

/// isIdenticalTo - Return true if the specified instruction is exactly
/// identical to the current one.  This means that all operands match and any
/// extra information (e.g. load is volatile) agree.
bool Instruction::isIdenticalTo(Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i) != I->getOperand(i))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(this))
    return LI->isVolatile() == cast<LoadInst>(I)->isVolatile();
  if (const StoreInst *SI = dyn_cast<StoreInst>(this))
    return SI->isVolatile() == cast<StoreInst>(I)->isVolatile();
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall() == cast<CallInst>(I)->isTailCall();
  return true;
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
///   Commutative operators satisfy: (x op y) === (y op x)
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

/// isRelational - Return true if the instruction is a Set* instruction:
///
bool Instruction::isRelational(unsigned op) {
  switch (op) {
  case SetEQ:
  case SetNE:
  case SetLT:
  case SetGT:
  case SetLE:
  case SetGE:
    return true;
  }
  return false;
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
