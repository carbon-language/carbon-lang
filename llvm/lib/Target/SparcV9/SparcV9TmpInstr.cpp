//===- SparcV9TmpInstr.cpp - SparcV9 Intermediate Value class -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Methods of class for temporary intermediate values used within the current
// SparcV9 backend.
//	
//===----------------------------------------------------------------------===//

#include "SparcV9TmpInstr.h"
#include "llvm/Support/LeakDetector.h"

namespace llvm {

TmpInstruction::TmpInstruction(Value *s1, Value *s2, const std::string &name)
  : Instruction(s1->getType(), Instruction::UserOp1, name) {
  Operands.push_back(Use(s1, this));  // s1 must be non-null
  if (s2)
    Operands.push_back(Use(s2, this));

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               Value *s1, Value *s2, const std::string &name)
  : Instruction(s1->getType(), Instruction::UserOp1, name) {
  mcfi.addTemp(this);

  Operands.push_back(Use(s1, this));  // s1 must be non-null
  if (s2)
    Operands.push_back(Use(s2, this));

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

// Constructor that requires the type of the temporary to be specified.
// Both S1 and S2 may be NULL.
TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               const Type *Ty, Value *s1, Value* s2,
                               const std::string &name)
  : Instruction(Ty, Instruction::UserOp1, name) {
  mcfi.addTemp(this);

  if (s1) 
    Operands.push_back(Use(s1, this));
  if (s2)
    Operands.push_back(Use(s2, this));

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

} // end namespace llvm
