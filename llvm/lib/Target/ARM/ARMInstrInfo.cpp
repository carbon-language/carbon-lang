//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool> EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
                                  cl::desc("Enable ARM 2-addr to 3-addr conv"));

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : TargetInstrInfo(ARMInsts, sizeof(ARMInsts)/sizeof(ARMInsts[0])),
    RI(*this, STI) {
}

const TargetRegisterClass *ARMInstrInfo::getPointerRegClass() const {
  return &ARM::GPRRegClass;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool ARMInstrInfo::isMoveInstr(const MachineInstr &MI,
                               unsigned &SrcReg, unsigned &DstReg) const {
  MachineOpCode oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  case ARM::FCPYS:
  case ARM::FCPYD:
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  case ARM::MOVrr:
  case ARM::tMOVrr:
    assert(MI.getNumOperands() == 2 && MI.getOperand(0).isRegister() &&
	   MI.getOperand(1).isRegister() &&
	   "Invalid ARM MOV instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}

unsigned ARMInstrInfo::isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const{
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDR:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImmediate() && 
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FLDD:
  case ARM::FLDS:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isImmediate() && 
        MI->getOperand(2).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::tLDRspi:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isImmediate() && 
        MI->getOperand(2).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned ARMInstrInfo::isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STR:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImmediate() && 
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FSTD:
  case ARM::FSTS:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isImmediate() && 
        MI->getOperand(2).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::tSTRspi:
    if (MI->getOperand(1).isFrameIndex() &&
        MI->getOperand(2).isImmediate() && 
        MI->getOperand(2).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

static unsigned getUnindexedOpcode(unsigned Opc) {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDR;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRB;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STR;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRB;
  }
  return 0;
}

MachineInstr *
ARMInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                    MachineBasicBlock::iterator &MBBI,
                                    LiveVariables &LV) const {
  if (!EnableARM3Addr)
    return NULL;

  MachineInstr *MI = MBBI;
  unsigned TSFlags = MI->getInstrDescriptor()->TSFlags;
  bool isPre = false;
  switch ((TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift) {
  default: return NULL;
  case ARMII::IndexModePre:
    isPre = true;
    break;
  case ARMII::IndexModePost:
    break;
  }

  // Try spliting an indexed load / store to a un-indexed one plus an add/sub
  // operation.
  unsigned MemOpc = getUnindexedOpcode(MI->getOpcode());
  if (MemOpc == 0)
    return NULL;

  MachineInstr *UpdateMI = NULL;
  MachineInstr *MemMI = NULL;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  unsigned NumOps = MI->getNumOperands();
  bool isLoad = (MI->getInstrDescriptor()->Flags & M_LOAD_FLAG) != 0;
  const MachineOperand &WB = isLoad ? MI->getOperand(1) : MI->getOperand(0);
  const MachineOperand &Base = MI->getOperand(2);
  const MachineOperand &Offset = MI->getOperand(NumOps-2);
  unsigned WBReg = WB.getReg();
  unsigned BaseReg = Base.getReg();
  unsigned OffReg = Offset.getReg();
  unsigned OffImm = MI->getOperand(NumOps-1).getImm();
  switch (AddrMode) {
  default:
    assert(false && "Unknown indexed op!");
    return NULL;
  case ARMII::AddrMode2: {
    bool isSub = ARM_AM::getAM2Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM2Offset(OffImm);
    if (OffReg == 0) {
      int SOImmVal = ARM_AM::getSOImmVal(Amt);
      if (SOImmVal == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(SOImmVal);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc);
    } else 
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt);
    else
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0);
    else
      MemMI = BuildMI(get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0);
    else
      MemMI = BuildMI(get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0);
    if (WB.isDead())
      UpdateMI->getOperand(0).setIsDead();
    NewMIs.push_back(UpdateMI);
    NewMIs.push_back(MemMI);
  }
  
  // Transfer LiveVariables states, kill / dead info.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.getReg() &&
        MRegisterInfo::isVirtualRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      LiveVariables::VarInfo &VI = LV.getVarInfo(Reg);
      if (MO.isDef()) {
        MachineInstr *NewMI = (Reg == WBReg) ? UpdateMI : MemMI;
        if (MO.isDead())
          LV.addVirtualRegisterDead(Reg, NewMI);
        // Update the defining instruction.
        if (VI.DefInst == MI)
          VI.DefInst = NewMI;
      }
      if (MO.isUse() && MO.isKill()) {
        for (unsigned j = 0; j < 2; ++j) {
          // Look at the two new MI's in reverse order.
          MachineInstr *NewMI = NewMIs[j];
          MachineOperand *NMO = NewMI->findRegisterUseOperand(Reg);
          if (!NMO)
            continue;
          LV.addVirtualRegisterKilled(Reg, NewMI);
          if (VI.removeKill(MI))
            VI.Kills.push_back(NewMI);
          break;
        }
      }
    }
  }

  MFI->insert(MBBI, NewMIs[1]);
  MFI->insert(MBBI, NewMIs[0]);
  return NewMIs[0];
}

// Branch analysis.
bool ARMInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 std::vector<MachineOperand> &Cond) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode()))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode())) {
    if (LastOpc == ARM::B || LastOpc == ARM::tB) {
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      return false;
    }
    if (LastOpc == ARM::Bcc || LastOpc == ARM::tBcc) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      Cond.push_back(LastInst->getOperand(1));
      return false;
    }
    return true;  // Can't handle indirect branch.
  }
  
  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isTerminatorInstr((--I)->getOpcode()))
    return true;
  
  // If the block ends with ARM::B/ARM::tB and a ARM::Bcc/ARM::tBcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if ((SecondLastOpc == ARM::Bcc && LastOpc == ARM::B) ||
      (SecondLastOpc == ARM::tBcc && LastOpc == ARM::tB)) {
    TBB =  SecondLastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMachineBasicBlock();
    return false;
  }
  
  // Otherwise, can't handle this.
  return true;
}


void ARMInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return;
  --I;
  if (I->getOpcode() != BccOpc)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
}

void ARMInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");
  
  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, get(BccOpc)).addMBB(TBB).addImm(Cond[0].getImm());
    return;
  }
  
  // Two-way conditional branch.
  BuildMI(&MBB, get(BccOpc)).addMBB(TBB).addImm(Cond[0].getImm());
  BuildMI(&MBB, get(BOpc)).addMBB(FBB);
}

bool ARMInstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case ARM::B:
  case ARM::tB:       // Uncond branch.
  case ARM::BR_JTr:   // Jumptable branch.
  case ARM::BR_JTm:   // Jumptable branch through mem.
  case ARM::BR_JTadd: // Jumptable branch add to pc.
    return true;
  default: return false;
  }
}

bool ARMInstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
  Cond[0].setImm(ARMCC::getOppositeCondition(CC));
  return false;
}
