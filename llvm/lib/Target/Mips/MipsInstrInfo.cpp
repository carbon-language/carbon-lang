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
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "MipsGenInstrInfo.inc"

using namespace llvm;

MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm)
  : TargetInstrInfoImpl(MipsInsts, array_lengthof(MipsInsts)),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

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
unsigned MipsInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const 
{
  if ((MI->getOpcode() == Mips::SW) || (MI->getOpcode() == Mips::SWC1) ||
      (MI->getOpcode() == Mips::SDC1)) {
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
          .addImm(0).addFrameIndex(FI);
  else if (RC == Mips::FGR32RegisterClass)
    BuildMI(MBB, I, DL, get(Mips::SWC1)).addReg(SrcReg, getKillRegState(isKill))
          .addImm(0).addFrameIndex(FI);
  else if (RC == Mips::AFGR64RegisterClass) {
    if (!TM.getSubtarget<MipsSubtarget>().isMips1()) {
      BuildMI(MBB, I, DL, get(Mips::SDC1))
        .addReg(SrcReg, getKillRegState(isKill))
        .addImm(0).addFrameIndex(FI);
    } else {
      const TargetRegisterInfo *TRI = 
        MBB.getParent()->getTarget().getRegisterInfo();
      const unsigned *SubSet = TRI->getSubRegisters(SrcReg);
      BuildMI(MBB, I, DL, get(Mips::SWC1))
        .addReg(SubSet[0], getKillRegState(isKill))
        .addImm(0).addFrameIndex(FI);
      BuildMI(MBB, I, DL, get(Mips::SWC1))
        .addReg(SubSet[1], getKillRegState(isKill))
        .addImm(4).addFrameIndex(FI);
    }
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
    BuildMI(MBB, I, DL, get(Mips::LW), DestReg).addImm(0).addFrameIndex(FI);
  else if (RC == Mips::FGR32RegisterClass)
    BuildMI(MBB, I, DL, get(Mips::LWC1), DestReg).addImm(0).addFrameIndex(FI);
  else if (RC == Mips::AFGR64RegisterClass) {
    if (!TM.getSubtarget<MipsSubtarget>().isMips1()) {
      BuildMI(MBB, I, DL, get(Mips::LDC1), DestReg).addImm(0).addFrameIndex(FI);
    } else {
      const TargetRegisterInfo *TRI = 
        MBB.getParent()->getTarget().getRegisterInfo();
      const unsigned *SubSet = TRI->getSubRegisters(DestReg);
      BuildMI(MBB, I, DL, get(Mips::LWC1), SubSet[0])
        .addImm(0).addFrameIndex(FI);
      BuildMI(MBB, I, DL, get(Mips::LWC1), SubSet[1])
        .addImm(4).addFrameIndex(FI);
    }
  } else
    llvm_unreachable("Register class not handled!");
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

/// GetCondFromBranchOpc - Return the Mips CC that matches 
/// the correspondent Branch instruction opcode.
static Mips::CondCode GetCondFromBranchOpc(unsigned BrOpc) 
{
  switch (BrOpc) {
  default: return Mips::COND_INVALID;
  case Mips::BEQ  : return Mips::COND_E;
  case Mips::BNE  : return Mips::COND_NE;
  case Mips::BGTZ : return Mips::COND_GZ;
  case Mips::BGEZ : return Mips::COND_GEZ;
  case Mips::BLTZ : return Mips::COND_LZ;
  case Mips::BLEZ : return Mips::COND_LEZ;

  // We dont do fp branch analysis yet!  
  case Mips::BC1T : 
  case Mips::BC1F : return Mips::COND_INVALID;
  }
}

/// GetCondBranchFromCond - Return the Branch instruction
/// opcode that matches the cc.
unsigned Mips::GetCondBranchFromCond(Mips::CondCode CC) 
{
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case Mips::COND_E   : return Mips::BEQ;
  case Mips::COND_NE  : return Mips::BNE;
  case Mips::COND_GZ  : return Mips::BGTZ;
  case Mips::COND_GEZ : return Mips::BGEZ;
  case Mips::COND_LZ  : return Mips::BLTZ;
  case Mips::COND_LEZ : return Mips::BLEZ;

  case Mips::FCOND_F:
  case Mips::FCOND_UN:
  case Mips::FCOND_EQ:
  case Mips::FCOND_UEQ:
  case Mips::FCOND_OLT:
  case Mips::FCOND_ULT:
  case Mips::FCOND_OLE:
  case Mips::FCOND_ULE:
  case Mips::FCOND_SF:
  case Mips::FCOND_NGLE:
  case Mips::FCOND_SEQ:
  case Mips::FCOND_NGL:
  case Mips::FCOND_LT:
  case Mips::FCOND_NGE:
  case Mips::FCOND_LE:
  case Mips::FCOND_NGT: return Mips::BC1T;

  case Mips::FCOND_T:
  case Mips::FCOND_OR:
  case Mips::FCOND_NEQ:
  case Mips::FCOND_OGL:
  case Mips::FCOND_UGE:
  case Mips::FCOND_OGE:
  case Mips::FCOND_UGT:
  case Mips::FCOND_OGT:
  case Mips::FCOND_ST:
  case Mips::FCOND_GLE:
  case Mips::FCOND_SNE:
  case Mips::FCOND_GL:
  case Mips::FCOND_NLT:
  case Mips::FCOND_GE:
  case Mips::FCOND_NLE:
  case Mips::FCOND_GT: return Mips::BC1F;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified 
/// condition, e.g. turning COND_E to COND_NE.
Mips::CondCode Mips::GetOppositeBranchCondition(Mips::CondCode CC) 
{
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case Mips::COND_E   : return Mips::COND_NE;
  case Mips::COND_NE  : return Mips::COND_E;
  case Mips::COND_GZ  : return Mips::COND_LEZ;
  case Mips::COND_GEZ : return Mips::COND_LZ;
  case Mips::COND_LZ  : return Mips::COND_GEZ;
  case Mips::COND_LEZ : return Mips::COND_GZ;
  case Mips::FCOND_F  : return Mips::FCOND_T;
  case Mips::FCOND_UN : return Mips::FCOND_OR;
  case Mips::FCOND_EQ : return Mips::FCOND_NEQ;
  case Mips::FCOND_UEQ: return Mips::FCOND_OGL;
  case Mips::FCOND_OLT: return Mips::FCOND_UGE;
  case Mips::FCOND_ULT: return Mips::FCOND_OGE;
  case Mips::FCOND_OLE: return Mips::FCOND_UGT;
  case Mips::FCOND_ULE: return Mips::FCOND_OGT;
  case Mips::FCOND_SF:  return Mips::FCOND_ST;
  case Mips::FCOND_NGLE:return Mips::FCOND_GLE;
  case Mips::FCOND_SEQ: return Mips::FCOND_SNE;
  case Mips::FCOND_NGL: return Mips::FCOND_GL;
  case Mips::FCOND_LT:  return Mips::FCOND_NLT;
  case Mips::FCOND_NGE: return Mips::FCOND_GE;
  case Mips::FCOND_LE:  return Mips::FCOND_NLE;
  case Mips::FCOND_NGT: return Mips::FCOND_GT;
  }
}

bool MipsInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const 
{
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (!LastInst->getDesc().isBranch())
      return true;

    // Unconditional branch
    if (LastOpc == Mips::J) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }

    Mips::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == Mips::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Conditional branch
    // Block ends with fall-through condbranch.
    if (LastOpc != Mips::COND_INVALID) {
      int LastNumOp = LastInst->getNumOperands();

      TBB = LastInst->getOperand(LastNumOp-1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));

      for (int i=0; i<LastNumOp-1; i++) {
        Cond.push_back(LastInst->getOperand(i));
      }

      return false;
    }
  }
  
  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with Mips::J and a Mips::BNE/Mips::BEQ, handle it.
  unsigned SecondLastOpc    = SecondLastInst->getOpcode();
  Mips::CondCode BranchCode = GetCondFromBranchOpc(SecondLastOpc);

  if (BranchCode != Mips::COND_INVALID && LastOpc == Mips::J) {
    int SecondNumOp = SecondLastInst->getNumOperands();

    TBB = SecondLastInst->getOperand(SecondNumOp-1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));

    for (int i=0; i<SecondNumOp-1; i++) {
      Cond.push_back(SecondLastInst->getOperand(i));
    }

    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it. The last 
  // one is not executed, so remove it.
  if ((SecondLastOpc == Mips::J) && (LastOpc == Mips::J)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB, 
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond,
             DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 2 || Cond.size() == 0) &&
         "Mips branch conditions can have two|three components!");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, DL, get(Mips::J)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
      const TargetInstrDesc &TID = get(Opc);

      if (TID.getNumOperands() == 3)
        BuildMI(&MBB, DL, TID).addReg(Cond[1].getReg())
                          .addReg(Cond[2].getReg())
                          .addMBB(TBB);
      else
        BuildMI(&MBB, DL, TID).addReg(Cond[1].getReg())
                          .addMBB(TBB);

    }                             
    return 1;
  }
  
  // Two-way Conditional branch.
  unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
  const TargetInstrDesc &TID = get(Opc);

  if (TID.getNumOperands() == 3)
    BuildMI(&MBB, DL, TID).addReg(Cond[1].getReg()).addReg(Cond[2].getReg())
                      .addMBB(TBB);
  else
    BuildMI(&MBB, DL, TID).addReg(Cond[1].getReg()).addMBB(TBB);

  BuildMI(&MBB, DL, get(Mips::J)).addMBB(FBB);
  return 2;
}

unsigned MipsInstrInfo::
RemoveBranch(MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (I->getOpcode() != Mips::J && 
      GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

/// ReverseBranchCondition - Return the inverse opcode of the 
/// specified Branch instruction.
bool MipsInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const 
{
  assert( (Cond.size() == 3 || Cond.size() == 2) && 
          "Invalid Mips branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((Mips::CondCode)Cond[0].getImm()));
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
