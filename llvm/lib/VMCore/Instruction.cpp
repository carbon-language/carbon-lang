//===-- Instruction.cpp - Implement the Instruction class --------*- C++ -*--=//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/CodeGen/MachineInstr.h"

Instruction::Instruction(const Type *ty, unsigned it, const string &Name) 
  : User(ty, Value::InstructionVal, Name), 
    machineInstrVec(new MachineCodeForVMInstr) {
  Parent = 0;
  iType = it;
}

Instruction::~Instruction() {
  assert(getParent() == 0 && "Instruction still embedded in basic block!");
  delete machineInstrVec;
}

// Specialize setName to take care of symbol table majik
void Instruction::setName(const string &name, SymbolTable *ST) {
  BasicBlock *P = 0; Method *PP = 0;
  assert((ST == 0 || !getParent() || !getParent()->getParent() || 
	  ST == getParent()->getParent()->getSymbolTable()) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && (PP = P->getParent()) && hasName())
    PP->getSymbolTable()->remove(this);
  Value::setName(name);
  if (PP && hasName()) PP->getSymbolTableSure()->insert(this);
}

void Instruction::addMachineInstruction(MachineInstr* minstr) {
  machineInstrVec->push_back(minstr);
}

#if 0
// Dont make this inline because you would need to include
// MachineInstr.h in Instruction.h, which creates a circular
// sequence of forward declarations.  Trying to fix that will
// cause a serious circularity in link order.
// 
const vector<Value*> &Instruction::getTempValuesForMachineCode() const {
  return machineInstrVec->getTempValues();
}
#endif

void Instruction::dropAllReferences() {
  machineInstrVec->dropAllReferences();
  User::dropAllReferences();
}
