//===- MBlazeInstrInfo.cpp - MBlaze Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MBlazeInstrInfo.h"
#include "MBlazeTargetMachine.h"
#include "MBlazeMachineFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "MBlazeGenInstrInfo.inc"

using namespace llvm;

MBlazeInstrInfo::MBlazeInstrInfo(MBlazeTargetMachine &tm)
  : TargetInstrInfoImpl(MBlazeInsts, array_lengthof(MBlazeInsts)),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MBlazeInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::LWI) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned MBlazeInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const {
  if (MI->getOpcode() == MBlaze::SWI) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MBlazeInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(MBlaze::NOP));
}

void MBlazeInstrInfo::
copyPhysReg(MachineBasicBlock &MBB,
            MachineBasicBlock::iterator I, DebugLoc DL,
            unsigned DestReg, unsigned SrcReg,
            bool KillSrc) const {
  llvm::BuildMI(MBB, I, DL, get(MBlaze::ADD), DestReg)
    .addReg(SrcReg, getKillRegState(KillSrc)).addReg(MBlaze::R0);
}

void MBlazeInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::SWI)).addReg(SrcReg,getKillRegState(isKill))
    .addImm(0).addFrameIndex(FI);
}

void MBlazeInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  BuildMI(MBB, I, DL, get(MBlaze::LWI), DestReg)
      .addImm(0).addFrameIndex(FI);
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//
unsigned MBlazeInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  // Can only insert uncond branches so far.
  assert(Cond.empty() && !FBB && TBB && "Can only handle uncond branches!");
  BuildMI(&MBB, DL, get(MBlaze::BRI)).addMBB(TBB);
  return 1;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned MBlazeInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  MBlazeFunctionInfo *MBlazeFI = MF->getInfo<MBlazeFunctionInfo>();
  unsigned GlobalBaseReg = MBlazeFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalBaseReg = RegInfo.createVirtualRegister(MBlaze::GPRRegisterClass);
  BuildMI(FirstMBB, MBBI, DebugLoc(), TII->get(TargetOpcode::COPY),
          GlobalBaseReg).addReg(MBlaze::R20);
  RegInfo.addLiveIn(MBlaze::R20);

  MBlazeFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}
