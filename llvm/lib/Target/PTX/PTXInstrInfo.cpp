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

#define DEBUG_TYPE "ptx-instrinfo"

#include "PTX.h"
#include "PTXInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#include "PTXGenInstrInfo.inc"

PTXInstrInfo::PTXInstrInfo(PTXTargetMachine &_TM)
  : TargetInstrInfoImpl(PTXInsts, array_lengthof(PTXInsts)),
    RI(_TM, *this), TM(_TM) {}

static const struct map_entry {
  const TargetRegisterClass *cls;
  const int opcode;
} map[] = {
  { &PTX::RegI16RegClass, PTX::MOVU16rr },
  { &PTX::RegI32RegClass, PTX::MOVU32rr },
  { &PTX::RegI64RegClass, PTX::MOVU64rr },
  { &PTX::RegF32RegClass, PTX::MOVF32rr },
  { &PTX::RegF64RegClass, PTX::MOVF64rr },
  { &PTX::RegPredRegClass,   PTX::MOVPREDrr }
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
  return i != -1 && MI->getOperand(i).getReg() != PTX::NoRegister;
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
  const MachineOperand &PredReg1 = Pred1[0];
  const MachineOperand &PredReg2 = Pred2[0];
  if (PredReg1.getReg() != PredReg2.getReg())
    return false;

  const MachineOperand &PredOp1 = Pred1[1];
  const MachineOperand &PredOp2 = Pred2[1];
  if (PredOp1.getImm() != PredOp2.getImm())
    return false;

  return true;
}

bool PTXInstrInfo::
DefinesPredicate(MachineInstr *MI,
                 std::vector<MachineOperand> &Pred) const {
  // If an instruction sets a predicate register, it defines a predicate.

  // TODO supprot 5-operand format of setp instruction

  if (MI->getNumOperands() < 1)
    return false;

  const MachineOperand &MO = MI->getOperand(0);

  if (!MO.isReg() || RI.getRegClass(MO.getReg()) != &PTX::RegPredRegClass)
    return false;

  Pred.push_back(MO);
  Pred.push_back(MachineOperand::CreateImm(PTX::PRED_NORMAL));
  return true;
}

// branch support

bool PTXInstrInfo::
AnalyzeBranch(MachineBasicBlock &MBB,
              MachineBasicBlock *&TBB,
              MachineBasicBlock *&FBB,
              SmallVectorImpl<MachineOperand> &Cond,
              bool AllowModify) const {
  // TODO implement cases when AllowModify is true

  if (MBB.empty())
    return true;

  MachineBasicBlock::const_iterator iter = MBB.end();
  const MachineInstr& instLast1 = *--iter;
  const TargetInstrDesc &desc1 = instLast1.getDesc();
  // for special case that MBB has only 1 instruction
  const bool IsSizeOne = MBB.size() == 1;
  // if IsSizeOne is true, *--iter and instLast2 are invalid
  // we put a dummy value in instLast2 and desc2 since they are used
  const MachineInstr& instLast2 = IsSizeOne ? instLast1 : *--iter;
  const TargetInstrDesc &desc2 = IsSizeOne ? desc1 : instLast2.getDesc();

  DEBUG(dbgs() << "\n");
  DEBUG(dbgs() << "AnalyzeBranch: opcode: " << instLast1.getOpcode() << "\n");
  DEBUG(dbgs() << "AnalyzeBranch: MBB:    " << MBB.getName().str() << "\n");
  DEBUG(dbgs() << "AnalyzeBranch: TBB:    " << TBB << "\n");
  DEBUG(dbgs() << "AnalyzeBranch: FBB:    " << FBB << "\n");

  // this block ends with no branches
  if (!IsAnyKindOfBranch(instLast1)) {
    DEBUG(dbgs() << "AnalyzeBranch: ends with no branch\n");
    return false;
  }

  // this block ends with only an unconditional branch
  if (desc1.isUnconditionalBranch() &&
      // when IsSizeOne is true, it "absorbs" the evaluation of instLast2
      (IsSizeOne || !IsAnyKindOfBranch(instLast2))) {
    DEBUG(dbgs() << "AnalyzeBranch: ends with only uncond branch\n");
    TBB = GetBranchTarget(instLast1);
    return false;
  }

  // this block ends with a conditional branch and
  // it falls through to a successor block
  if (desc1.isConditionalBranch() &&
      IsAnySuccessorAlsoLayoutSuccessor(MBB)) {
    DEBUG(dbgs() << "AnalyzeBranch: ends with cond branch and fall through\n");
    TBB = GetBranchTarget(instLast1);
    int i = instLast1.findFirstPredOperandIdx();
    Cond.push_back(instLast1.getOperand(i));
    Cond.push_back(instLast1.getOperand(i+1));
    return false;
  }

  // when IsSizeOne is true, we are done
  if (IsSizeOne)
    return true;

  // this block ends with a conditional branch
  // followed by an unconditional branch
  if (desc2.isConditionalBranch() &&
      desc1.isUnconditionalBranch()) {
    DEBUG(dbgs() << "AnalyzeBranch: ends with cond and uncond branch\n");
    TBB = GetBranchTarget(instLast2);
    FBB = GetBranchTarget(instLast1);
    int i = instLast2.findFirstPredOperandIdx();
    Cond.push_back(instLast2.getOperand(i));
    Cond.push_back(instLast2.getOperand(i+1));
    return false;
  }

  // branch cannot be understood
  DEBUG(dbgs() << "AnalyzeBranch: cannot be understood\n");
  return true;
}

unsigned PTXInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  unsigned count = 0;
  while (!MBB.empty())
    if (IsAnyKindOfBranch(MBB.back())) {
      MBB.pop_back();
      ++count;
    } else
      break;
  DEBUG(dbgs() << "RemoveBranch: MBB:   " << MBB.getName().str() << "\n");
  DEBUG(dbgs() << "RemoveBranch: remove " << count << " branch inst\n");
  return count;
}

unsigned PTXInstrInfo::
InsertBranch(MachineBasicBlock &MBB,
             MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  DEBUG(dbgs() << "InsertBranch: MBB: " << MBB.getName().str() << "\n");
  DEBUG(if (TBB) dbgs() << "InsertBranch: TBB: " << TBB->getName().str()
                        << "\n";
        else     dbgs() << "InsertBranch: TBB: (NULL)\n");
  DEBUG(if (FBB) dbgs() << "InsertBranch: FBB: " << FBB->getName().str()
                        << "\n";
        else     dbgs() << "InsertBranch: FBB: (NULL)\n");
  DEBUG(dbgs() << "InsertBranch: Cond size: " << Cond.size() << "\n");

  assert(TBB && "TBB is NULL");

  if (FBB) {
    BuildMI(&MBB, DL, get(PTX::BRAdp))
      .addMBB(TBB).addReg(Cond[0].getReg()).addImm(Cond[1].getImm());
    BuildMI(&MBB, DL, get(PTX::BRAd))
      .addMBB(FBB).addReg(PTX::NoRegister).addImm(PTX::PRED_NORMAL);
    return 2;
  } else if (Cond.size()) {
    BuildMI(&MBB, DL, get(PTX::BRAdp))
      .addMBB(TBB).addReg(Cond[0].getReg()).addImm(Cond[1].getImm());
    return 1;
  } else {
    BuildMI(&MBB, DL, get(PTX::BRAd))
      .addMBB(TBB).addReg(PTX::NoRegister).addImm(PTX::PRED_NORMAL);
    return 1;
  }
}

// Memory operand folding for spills
void PTXInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MII,
                                       unsigned SrcReg, bool isKill, int FrameIdx,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  MachineInstr& MI = *MII;
  DebugLoc DL = MI.getDebugLoc();

  DEBUG(dbgs() << "storeRegToStackSlot: " << MI);

  int OpCode;

  // Select the appropriate opcode based on the register class
  if (RC == PTX::RegI16RegisterClass) {
    OpCode = PTX::STACKSTOREI16;
  }
  else if (RC == PTX::RegI32RegisterClass) {
    OpCode = PTX::STACKSTOREI32;
  }
  else if (RC == PTX::RegI64RegisterClass) {
    OpCode = PTX::STACKSTOREI32;
  }
  else if (RC == PTX::RegF32RegisterClass) {
    OpCode = PTX::STACKSTOREF32;
  }
  else if (RC == PTX::RegF64RegisterClass) {
    OpCode = PTX::STACKSTOREF64;
  }

  // Build the store instruction (really a mov)
  MachineInstrBuilder MIB = BuildMI(MBB, MII, DL, get(OpCode));
  MIB.addImm(FrameIdx);
  MIB.addReg(SrcReg);

  AddDefaultPredicate(MIB);
}

void PTXInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MII,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  MachineInstr& MI = *MII;
  DebugLoc DL = MI.getDebugLoc();

  DEBUG(dbgs() << "loadRegToStackSlot: " << MI);

  int OpCode;

  // Select the appropriate opcode based on the register class
  if (RC == PTX::RegI16RegisterClass) {
    OpCode = PTX::STACKLOADI16;
  }
  else if (RC == PTX::RegI32RegisterClass) {
    OpCode = PTX::STACKLOADI32;
  }
  else if (RC == PTX::RegI64RegisterClass) {
    OpCode = PTX::STACKLOADI32;
  }
  else if (RC == PTX::RegF32RegisterClass) {
    OpCode = PTX::STACKLOADF32;
  }
  else if (RC == PTX::RegF64RegisterClass) {
    OpCode = PTX::STACKLOADF64;
  }

  // Build the load instruction (really a mov)
  MachineInstrBuilder MIB = BuildMI(MBB, MII, DL, get(OpCode));
  MIB.addReg(DestReg);
  MIB.addImm(FrameIdx);

  AddDefaultPredicate(MIB);
}

// static helper routines

MachineSDNode *PTXInstrInfo::
GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                  DebugLoc dl, EVT VT, SDValue Op1) {
  SDValue predReg = DAG->getRegister(PTX::NoRegister, MVT::i1);
  SDValue predOp = DAG->getTargetConstant(PTX::PRED_NORMAL, MVT::i32);
  SDValue ops[] = { Op1, predReg, predOp };
  return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
}

MachineSDNode *PTXInstrInfo::
GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                  DebugLoc dl, EVT VT, SDValue Op1, SDValue Op2) {
  SDValue predReg = DAG->getRegister(PTX::NoRegister, MVT::i1);
  SDValue predOp = DAG->getTargetConstant(PTX::PRED_NORMAL, MVT::i32);
  SDValue ops[] = { Op1, Op2, predReg, predOp };
  return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
}

void PTXInstrInfo::AddDefaultPredicate(MachineInstr *MI) {
  if (MI->findFirstPredOperandIdx() == -1) {
    MI->addOperand(MachineOperand::CreateReg(PTX::NoRegister, /*IsDef=*/false));
    MI->addOperand(MachineOperand::CreateImm(PTX::PRED_NORMAL));
  }
}

bool PTXInstrInfo::IsAnyKindOfBranch(const MachineInstr& inst) {
  const TargetInstrDesc &desc = inst.getDesc();
  return desc.isTerminator() || desc.isBranch() || desc.isIndirectBranch();
}

bool PTXInstrInfo::
IsAnySuccessorAlsoLayoutSuccessor(const MachineBasicBlock& MBB) {
  for (MachineBasicBlock::const_succ_iterator
      i = MBB.succ_begin(), e = MBB.succ_end(); i != e; ++i)
    if (MBB.isLayoutSuccessor((const MachineBasicBlock*) &*i))
      return true;
  return false;
}

MachineBasicBlock *PTXInstrInfo::GetBranchTarget(const MachineInstr& inst) {
  // FIXME So far all branch instructions put destination in 1st operand
  const MachineOperand& target = inst.getOperand(0);
  assert(target.isMBB() && "FIXME: detect branch target operand");
  return target.getMBB();
}
