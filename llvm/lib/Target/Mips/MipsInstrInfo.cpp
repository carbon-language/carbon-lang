//===- MipsInstrInfo.cpp - Mips Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsInstrInfo.h"
#include "MipsTargetMachine.h"
#include "MipsMachineFunction.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/STLExtras.h"

#define GET_INSTRINFO_CTOR
#include "MipsGenInstrInfo.inc"

using namespace llvm;

MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm)
  : MipsGenInstrInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}


const MipsRegisterInfo &MipsInstrInfo::getRegisterInfo() const { 
  return RI;
}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MipsInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  if ((MI->getOpcode() == Mips::LW) || (MI->getOpcode() == Mips::LWC1) ||
      (MI->getOpcode() == Mips::LDC1)) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
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
unsigned MipsInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  if ((MI->getOpcode() == Mips::SW) || (MI->getOpcode() == Mips::SWC1) ||
      (MI->getOpcode() == Mips::SDC1)) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MipsInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const
{
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(Mips::NOP));
}

void MipsInstrInfo::
copyPhysReg(MachineBasicBlock &MBB,
            MachineBasicBlock::iterator I, DebugLoc DL,
            unsigned DestReg, unsigned SrcReg,
            bool KillSrc) const {
  bool DestCPU = Mips::CPURegsRegClass.contains(DestReg);
  bool SrcCPU  = Mips::CPURegsRegClass.contains(SrcReg);

  // CPU-CPU is the most common.
  if (DestCPU && SrcCPU) {
    BuildMI(MBB, I, DL, get(Mips::ADDu), DestReg).addReg(Mips::ZERO)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Copy to CPU from other registers.
  if (DestCPU) {
    if (Mips::CCRRegClass.contains(SrcReg))
      BuildMI(MBB, I, DL, get(Mips::CFC1), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    else if (Mips::FGR32RegClass.contains(SrcReg))
      BuildMI(MBB, I, DL, get(Mips::MFC1), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    else if (SrcReg == Mips::HI)
      BuildMI(MBB, I, DL, get(Mips::MFHI), DestReg);
    else if (SrcReg == Mips::LO)
      BuildMI(MBB, I, DL, get(Mips::MFLO), DestReg);
    else
      llvm_unreachable("Copy to CPU from invalid register");
    return;
  }

  // Copy to other registers from CPU.
  if (SrcCPU) {
    if (Mips::CCRRegClass.contains(DestReg))
      BuildMI(MBB, I, DL, get(Mips::CTC1), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    else if (Mips::FGR32RegClass.contains(DestReg))
      BuildMI(MBB, I, DL, get(Mips::MTC1), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    else if (DestReg == Mips::HI)
      BuildMI(MBB, I, DL, get(Mips::MTHI))
        .addReg(SrcReg, getKillRegState(KillSrc));
    else if (DestReg == Mips::LO)
      BuildMI(MBB, I, DL, get(Mips::MTLO))
        .addReg(SrcReg, getKillRegState(KillSrc));
    else
      llvm_unreachable("Copy from CPU to invalid register");
    return;
  }

  if (Mips::FGR32RegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(Mips::FMOV_S32), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (Mips::AFGR64RegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(Mips::FMOV_D32), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (Mips::CCRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(Mips::MOVCCRToCCR), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  llvm_unreachable("Cannot copy registers");
}

void MipsInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, DL, get(Mips::SW)).addReg(SrcReg, getKillRegState(isKill))
                                      .addFrameIndex(FI).addImm(0);
  else if (RC == Mips::FGR32RegisterClass)
    BuildMI(MBB, I, DL, get(Mips::SWC1)).addReg(SrcReg, getKillRegState(isKill))
                                        .addFrameIndex(FI).addImm(0);
  else if (RC == Mips::AFGR64RegisterClass) {
    BuildMI(MBB, I, DL, get(Mips::SDC1))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI).addImm(0);
  } else
    llvm_unreachable("Register class not handled!");
}

void MipsInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const
{
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, DL, get(Mips::LW), DestReg).addFrameIndex(FI).addImm(0);
  else if (RC == Mips::FGR32RegisterClass)
    BuildMI(MBB, I, DL, get(Mips::LWC1), DestReg).addFrameIndex(FI).addImm(0);
  else if (RC == Mips::AFGR64RegisterClass) {
    BuildMI(MBB, I, DL, get(Mips::LDC1), DestReg).addFrameIndex(FI).addImm(0);
  } else
    llvm_unreachable("Register class not handled!");
}

MachineInstr*
MipsInstrInfo::emitFrameIndexDebugValue(MachineFunction &MF, int FrameIx,
                                        uint64_t Offset, const MDNode *MDPtr,
                                        DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Mips::DBG_VALUE))
    .addFrameIndex(FrameIx).addImm(0).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

static unsigned GetAnalyzableBrOpc(unsigned Opc) {
  return (Opc == Mips::BEQ  || Opc == Mips::BNE  || Opc == Mips::BGTZ ||
          Opc == Mips::BGEZ || Opc == Mips::BLTZ || Opc == Mips::BLEZ ||
          Opc == Mips::BC1T || Opc == Mips::BC1F || Opc == Mips::J) ? Opc : 0;
}

/// GetOppositeBranchOpc - Return the inverse of the specified
/// opcode, e.g. turning BEQ to BNE.
unsigned Mips::GetOppositeBranchOpc(unsigned Opc)
{
  switch (Opc) {
  default: llvm_unreachable("Illegal opcode!");
  case Mips::BEQ  : return Mips::BNE;
  case Mips::BNE  : return Mips::BEQ;
  case Mips::BGTZ : return Mips::BLEZ;
  case Mips::BGEZ : return Mips::BLTZ;
  case Mips::BLTZ : return Mips::BGEZ;
  case Mips::BLEZ : return Mips::BGTZ;
  case Mips::BC1T : return Mips::BC1F;
  case Mips::BC1F : return Mips::BC1T;
  }
}

static void AnalyzeCondBr(const MachineInstr* Inst, unsigned Opc,
                          MachineBasicBlock *&BB,
                          SmallVectorImpl<MachineOperand>& Cond) {
  assert(GetAnalyzableBrOpc(Opc) && "Not an analyzable branch");
  int NumOp = Inst->getNumExplicitOperands();
  
  // for both int and fp branches, the last explicit operand is the
  // MBB.
  BB = Inst->getOperand(NumOp-1).getMBB();
  Cond.push_back(MachineOperand::CreateImm(Opc));

  for (int i=0; i<NumOp-1; i++)
    Cond.push_back(Inst->getOperand(i));
}

bool MipsInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const
{
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
    ++I;

  if (I == REnd || !isUnpredicatedTerminator(&*I)) {
    // If this block ends with no branches (it just falls through to its succ)
    // just return false, leaving TBB/FBB null.
    TBB = FBB = NULL;
    return false;
  }

  MachineInstr *LastInst = &*I;
  unsigned LastOpc = LastInst->getOpcode();

  // Not an analyzable branch (must be an indirect jump).
  if (!GetAnalyzableBrOpc(LastOpc))
    return true;

  // Get the second to last instruction in the block.
  unsigned SecondLastOpc = 0;
  MachineInstr *SecondLastInst = NULL;

  if (++I != REnd) {
    SecondLastInst = &*I;
    SecondLastOpc = GetAnalyzableBrOpc(SecondLastInst->getOpcode());

    // Not an analyzable branch (must be an indirect jump).
    if (isUnpredicatedTerminator(SecondLastInst) && !SecondLastOpc)
      return true;
  }

  // If there is only one terminator instruction, process it.
  if (!SecondLastOpc) {
    // Unconditional branch
    if (LastOpc == Mips::J) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }

    // Conditional branch
    AnalyzeCondBr(LastInst, LastOpc, TBB, Cond);
    return false;
  }

  // If we reached here, there are two branches.
  // If there are three terminators, we don't know what sort of block this is.
  if (++I != REnd && isUnpredicatedTerminator(&*I))
    return true;

  // If second to last instruction is an unconditional branch,
  // analyze it and remove the last instruction.
  if (SecondLastOpc == Mips::J) {
    // Return if the last instruction cannot be removed.
    if (!AllowModify)
      return true;

    TBB = SecondLastInst->getOperand(0).getMBB();
    LastInst->eraseFromParent();
    return false;
  }

  // Conditional branch followed by an unconditional branch.
  // The last one must be unconditional.
  if (LastOpc != Mips::J)
    return true;

  AnalyzeCondBr(SecondLastInst, SecondLastOpc, TBB, Cond);
  FBB = LastInst->getOperand(0).getMBB();

  return false;
} 
  
void MipsInstrInfo::BuildCondBr(MachineBasicBlock &MBB,
                                MachineBasicBlock *TBB, DebugLoc DL,
                                const SmallVectorImpl<MachineOperand>& Cond)
  const {
  unsigned Opc = Cond[0].getImm();
  const MCInstrDesc &MCID = get(Opc);
  MachineInstrBuilder MIB = BuildMI(&MBB, DL, MCID);

  for (unsigned i = 1; i < Cond.size(); ++i)
    MIB.addReg(Cond[i].getReg());

  MIB.addMBB(TBB);
}

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  // # of condition operands:
  //  Unconditional branches: 0
  //  Floating point branches: 1 (opc)
  //  Int BranchZero: 2 (opc, reg)
  //  Int Branch: 3 (opc, reg0, reg1)
  assert((Cond.size() <= 3) &&
         "# of Mips branch conditions must be <= 3!");

  // Two-way Conditional branch.
  if (FBB) {
    BuildCondBr(MBB, TBB, DL, Cond);
    BuildMI(&MBB, DL, get(Mips::J)).addMBB(FBB);
    return 2;
  }

  // One way branch.
  // Unconditional branch.
  if (Cond.empty())
    BuildMI(&MBB, DL, get(Mips::J)).addMBB(TBB);
  else // Conditional branch.
    BuildCondBr(MBB, TBB, DL, Cond);
  return 1;
}

unsigned MipsInstrInfo::
RemoveBranch(MachineBasicBlock &MBB) const
{
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), REnd = MBB.rend();
  MachineBasicBlock::reverse_iterator FirstBr;
  unsigned removed;

  // Skip all the debug instructions.
  while (I != REnd && I->isDebugValue())
    ++I;

  FirstBr = I;

  // Up to 2 branches are removed.
  // Note that indirect branches are not removed.
  for(removed = 0; I != REnd && removed < 2; ++I, ++removed)
    if (!GetAnalyzableBrOpc(I->getOpcode()))
      break;

  MBB.erase(I.base(), FirstBr.base());

  return removed;
}

/// ReverseBranchCondition - Return the inverse opcode of the
/// specified Branch instruction.
bool MipsInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const
{
  assert( (Cond.size() && Cond.size() <= 3) &&
          "Invalid Mips branch condition!");
  Cond[0].setImm(Mips::GetOppositeBranchOpc(Cond[0].getImm()));
  return false;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned MipsInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  MipsFunctionInfo *MipsFI = MF->getInfo<MipsFunctionInfo>();
  unsigned GlobalBaseReg = MipsFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalBaseReg = RegInfo.createVirtualRegister(Mips::CPURegsRegisterClass);
  BuildMI(FirstMBB, MBBI, DebugLoc(), TII->get(TargetOpcode::COPY),
          GlobalBaseReg).addReg(Mips::GP);
  RegInfo.addLiveIn(Mips::GP);

  MipsFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}
