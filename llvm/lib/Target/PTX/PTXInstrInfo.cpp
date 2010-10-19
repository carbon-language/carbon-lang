//===- PTXInstrInfo.cpp - PTX Instruction Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#include "PTXGenInstrInfo.inc"

PTXInstrInfo::PTXInstrInfo(PTXTargetMachine &_TM)
  : TargetInstrInfoImpl(PTXInsts, array_lengthof(PTXInsts)),
    RI(_TM, *this), TM(_TM) {}

static const struct map_entry {
  const TargetRegisterClass *cls;
  const int opcode;
} map[] = {
  { &PTX::RRegs32RegClass, PTX::MOVrr },
  { &PTX::PredsRegClass,   PTX::MOVpp }
};

void PTXInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL,
                               unsigned DstReg, unsigned SrcReg,
                               bool KillSrc) const {
  for (int i = 0, e = sizeof(map)/sizeof(map[0]); i != e; ++ i)
    if (PTX::RRegs32RegClass.contains(DstReg, SrcReg)) {
      BuildMI(MBB, I, DL,
              get(PTX::MOVrr), DstReg).addReg(SrcReg, getKillRegState(KillSrc));
      return;
    }

  llvm_unreachable("Impossible reg-to-reg copy");
}

bool PTXInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                unsigned DstReg, unsigned SrcReg,
                                const TargetRegisterClass *DstRC,
                                const TargetRegisterClass *SrcRC,
                                DebugLoc DL) const {
  if (DstRC != SrcRC)
    return false;

  for (int i = 0, e = sizeof(map)/sizeof(map[0]); i != e; ++ i)
    if (DstRC == map[i].cls) {
      MachineInstr *MI = BuildMI(MBB, I, DL, get(map[i].opcode),
                                 DstReg).addReg(SrcReg);
      if (MI->findFirstPredOperandIdx() == -1) {
        MI->addOperand(MachineOperand::CreateReg(0, false));
        MI->addOperand(MachineOperand::CreateImm(/*IsInv=*/0));
      }
      return true;
    }

  return false;
}

bool PTXInstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned &SrcReg, unsigned &DstReg,
                               unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  switch (MI.getOpcode()) {
    default:
      return false;
    case PTX::MOVpp:
    case PTX::MOVrr:
      assert(MI.getNumOperands() >= 2 &&
             MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
             "Invalid register-register move instruction");
      SrcSubIdx = DstSubIdx = 0; // No sub-registers
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
  }
}
