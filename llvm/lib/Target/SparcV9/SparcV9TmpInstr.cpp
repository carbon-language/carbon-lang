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
#include "llvm/Type.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

TmpInstruction::TmpInstruction(const TmpInstruction &TI)
  : Instruction(TI.getType(), TI.getOpcode(), Ops, TI.getNumOperands()) {
  if (TI.getNumOperands()) {
    Ops[0].init(TI.Ops[0], this);
    if (TI.getNumOperands() == 2)
      Ops[1].init(TI.Ops[1], this);
    else
      assert(0 && "Bad # operands to TmpInstruction!");
  }
}

TmpInstruction::TmpInstruction(Value *s1, Value *s2, const std::string &name)
  : Instruction(s1->getType(), Instruction::UserOp1, Ops, 1+(s2 != 0), name) {
  Ops[0].init(s1, this);  // s1 must be non-null
  if (s2)
    Ops[1].init(s2, this);

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               Value *s1, Value *s2, const std::string &name)
  : Instruction(s1->getType(), Instruction::UserOp1, Ops, 1+(s2 != 0), name) {
  mcfi.addTemp(this);

  Ops[0].init(s1, this);  // s1 must be non-null
  if (s2)
    Ops[1].init(s2, this);

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

// Constructor that requires the type of the temporary to be specified.
// Both S1 and S2 may be NULL.
TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               const Type *Ty, Value *s1, Value* s2,
                               const std::string &name)
  : Instruction(Ty, Instruction::UserOp1, Ops, (s1 != 0)+(s2 != 0), name) {
  mcfi.addTemp(this);

  assert((s1 != 0 || s2 == 0) &&
         "s2 cannot be non-null if s1 is non-null!");
  if (s1) {
    Ops[0].init(s1, this);
    if (s2)
      Ops[1].init(s2, this);
  }

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}
