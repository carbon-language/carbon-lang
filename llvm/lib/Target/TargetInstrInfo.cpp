//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

/// findTiedToSrcOperand - Returns the operand that is tied to the specified
/// dest operand. Returns -1 if there isn't one.
int TargetInstrDescriptor::findTiedToSrcOperand(unsigned OpNum) const {
  for (unsigned i = 0, e = numOperands; i != e; ++i) {
    if (i == OpNum)
      continue;
    if (getOperandConstraint(i, TOI::TIED_TO) == (int)OpNum)
      return i;
  }
  return -1;
}


TargetInstrInfo::TargetInstrInfo(const TargetInstrDescriptor* Desc,
                                 unsigned numOpcodes)
  : desc(Desc), NumOpcodes(numOpcodes) {
}

TargetInstrInfo::~TargetInstrInfo() {
}

// commuteInstruction - The default implementation of this method just exchanges
// operand 1 and 2.
MachineInstr *TargetInstrInfo::commuteInstruction(MachineInstr *MI) const {
  assert(MI->getOperand(1).isRegister() && MI->getOperand(2).isRegister() &&
         "This only knows how to commute register operands so far");
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  if (Reg1IsKill)
    MI->getOperand(2).setIsKill();
  else
    MI->getOperand(2).unsetIsKill();
  if (Reg2IsKill)
    MI->getOperand(1).setIsKill();
  else
    MI->getOperand(1).unsetIsKill();
  return MI;
}

bool TargetInstrInfo::PredicateInstruction(MachineInstr *MI,
                                const std::vector<MachineOperand> &Pred) const {
  bool MadeChange = false;
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  if (TID->Flags & M_PREDICABLE) {
    for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if ((TID->OpInfo[i].Flags & M_PREDICATE_OPERAND)) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isRegister()) {
          MO.setReg(Pred[j].getReg());
          MadeChange = true;
        } else if (MO.isImmediate()) {
          MO.setImm(Pred[j].getImmedValue());
          MadeChange = true;
        } else if (MO.isMachineBasicBlock()) {
          MO.setMachineBasicBlock(Pred[j].getMachineBasicBlock());
          MadeChange = true;
        }
        ++j;
      }
    }
  }
  return MadeChange;
}

bool TargetInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  if (TID->Flags & M_TERMINATOR_FLAG) {
    // Conditional branch is a special case.
    if ((TID->Flags & M_BRANCH_FLAG) != 0 && (TID->Flags & M_BARRIER_FLAG) == 0)
      return true;
    if ((TID->Flags & M_PREDICABLE) == 0)
      return true;
    return !isPredicated(MI);
  }
  return false;
}
