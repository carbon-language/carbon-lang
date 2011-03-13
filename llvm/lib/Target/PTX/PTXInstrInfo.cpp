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
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

using namespace llvm;

#include "PTXGenInstrInfo.inc"

PTXInstrInfo::PTXInstrInfo(PTXTargetMachine &_TM)
  : TargetInstrInfoImpl(PTXInsts, array_lengthof(PTXInsts)),
    RI(_TM, *this), TM(_TM) {}

static const struct map_entry {
  const TargetRegisterClass *cls;
  const int opcode;
} map[] = {
  { &PTX::RRegu16RegClass, PTX::MOVU16rr },
  { &PTX::RRegu32RegClass, PTX::MOVU32rr },
  { &PTX::RRegu64RegClass, PTX::MOVU64rr },
  { &PTX::RRegf32RegClass, PTX::MOVF32rr },
  { &PTX::RRegf64RegClass, PTX::MOVF64rr },
  { &PTX::PredsRegClass,   PTX::MOVPREDrr }
};

void PTXInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL,
                               unsigned DstReg, unsigned SrcReg,
                               bool KillSrc) const {
  for (int i = 0, e = sizeof(map)/sizeof(map[0]); i != e; ++ i) {
    if (map[i].cls->contains(DstReg, SrcReg)) {
      const TargetInstrDesc &TID = get(map[i].opcode);
      MachineInstr *MI = BuildMI(MBB, I, DL, TID, DstReg).
        addReg(SrcReg, getKillRegState(KillSrc));
      AddDefaultPredicate(MI);
      return;
    }
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
      const TargetInstrDesc &TID = get(map[i].opcode);
      MachineInstr *MI = BuildMI(MBB, I, DL, TID, DstReg).addReg(SrcReg);
      AddDefaultPredicate(MI);
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
    case PTX::MOVU16rr:
    case PTX::MOVU32rr:
    case PTX::MOVU64rr:
    case PTX::MOVF32rr:
    case PTX::MOVF64rr:
    case PTX::MOVPREDrr:
      assert(MI.getNumOperands() >= 2 &&
             MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
             "Invalid register-register move instruction");
      SrcSubIdx = DstSubIdx = 0; // No sub-registers
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
  }
}

// predicate support

bool PTXInstrInfo::isPredicated(const MachineInstr *MI) const {
  int i = MI->findFirstPredOperandIdx();
  if (i == -1)
    llvm_unreachable("missing predicate operand");
  return MI->getOperand(i).getReg() ||
         MI->getOperand(i+1).getImm() != PTX::PRED_IGNORE;
}

bool PTXInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  return !isPredicated(MI) && get(MI->getOpcode()).isTerminator();
}

bool PTXInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Pred) const {
  if (Pred.size() < 2)
    llvm_unreachable("lesser than 2 predicate operands are provided");

  int i = MI->findFirstPredOperandIdx();
  if (i == -1)
    llvm_unreachable("missing predicate operand");

  MI->getOperand(i).setReg(Pred[0].getReg());
  MI->getOperand(i+1).setImm(Pred[1].getImm());

  return true;
}

bool PTXInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  // TODO Implement SubsumesPredicate
  // Returns true if the first specified predicate subsumes the second,
  // e.g. GE subsumes GT.
  return false;
}


bool PTXInstrInfo::
DefinesPredicate(MachineInstr *MI,
                 std::vector<MachineOperand> &Pred) const {
  // TODO Implement DefinesPredicate
  // If the specified instruction defines any predicate or condition code
  // register(s) used for predication, returns true as well as the definition
  // predicate(s) by reference.
  return false;
}

// static helper routines

MachineSDNode *PTXInstrInfo::
GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                  DebugLoc dl, EVT VT, SDValue Op1) {
  SDValue predReg = DAG->getRegister(0, MVT::i1);
  SDValue predOp = DAG->getTargetConstant(PTX::PRED_IGNORE, MVT::i1);
  SDValue ops[] = { Op1, predReg, predOp };
  return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
}

MachineSDNode *PTXInstrInfo::
GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                  DebugLoc dl, EVT VT, SDValue Op1, SDValue Op2) {
  SDValue predReg = DAG->getRegister(0, MVT::i1);
  SDValue predOp = DAG->getTargetConstant(PTX::PRED_IGNORE, MVT::i1);
  SDValue ops[] = { Op1, Op2, predReg, predOp };
  return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
}

void PTXInstrInfo::AddDefaultPredicate(MachineInstr *MI) {
  if (MI->findFirstPredOperandIdx() == -1) {
    MI->addOperand(MachineOperand::CreateReg(0, /*IsDef=*/false));
    MI->addOperand(MachineOperand::CreateImm(PTX::PRED_IGNORE));
  }
}
