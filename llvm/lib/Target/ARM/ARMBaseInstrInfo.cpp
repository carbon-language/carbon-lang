//===- ARMBaseInstrInfo.cpp - ARM Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Base ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMBaseInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

static cl::opt<bool>
EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
               cl::desc("Enable ARM 2-addr to 3-addr conv"));

ARMBaseInstrInfo::ARMBaseInstrInfo(const ARMSubtarget& STI)
  : TargetInstrInfoImpl(ARMInsts, array_lengthof(ARMInsts)),
    Subtarget(STI) {
}

MachineInstr *
ARMBaseInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                        MachineBasicBlock::iterator &MBBI,
                                        LiveVariables *LV) const {
  // FIXME: Thumb2 support.

  if (!EnableARM3Addr)
    return NULL;

  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  unsigned TSFlags = MI->getDesc().TSFlags;
  bool isPre = false;
  switch ((TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift) {
  default: return NULL;
  case ARMII::IndexModePre:
    isPre = true;
    break;
  case ARMII::IndexModePost:
    break;
  }

  // Try splitting an indexed load/store to an un-indexed one plus an add/sub
  // operation.
  unsigned MemOpc = getUnindexedOpcode(MI->getOpcode());
  if (MemOpc == 0)
    return NULL;

  MachineInstr *UpdateMI = NULL;
  MachineInstr *MemMI = NULL;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned NumOps = TID.getNumOperands();
  bool isLoad = !TID.mayStore();
  const MachineOperand &WB = isLoad ? MI->getOperand(1) : MI->getOperand(0);
  const MachineOperand &Base = MI->getOperand(2);
  const MachineOperand &Offset = MI->getOperand(NumOps-3);
  unsigned WBReg = WB.getReg();
  unsigned BaseReg = Base.getReg();
  unsigned OffReg = Offset.getReg();
  unsigned OffImm = MI->getOperand(NumOps-2).getImm();
  ARMCC::CondCodes Pred = (ARMCC::CondCodes)MI->getOperand(NumOps-1).getImm();
  switch (AddrMode) {
  default:
    assert(false && "Unknown indexed op!");
    return NULL;
  case ARMII::AddrMode2: {
    bool isSub = ARM_AM::getAM2Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM2Offset(OffImm);
    if (OffReg == 0) {
      if (ARM_AM::getSOImmVal(Amt) == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc)
        .addImm(Pred).addReg(0).addReg(0);
    } else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    if (WB.isDead())
      UpdateMI->getOperand(0).setIsDead();
    NewMIs.push_back(UpdateMI);
    NewMIs.push_back(MemMI);
  }

  // Transfer LiveVariables states, kill / dead info.
  if (LV) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.getReg() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();

        LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);
        if (MO.isDef()) {
          MachineInstr *NewMI = (Reg == WBReg) ? UpdateMI : MemMI;
          if (MO.isDead())
            LV->addVirtualRegisterDead(Reg, NewMI);
        }
        if (MO.isUse() && MO.isKill()) {
          for (unsigned j = 0; j < 2; ++j) {
            // Look at the two new MI's in reverse order.
            MachineInstr *NewMI = NewMIs[j];
            if (!NewMI->readsRegister(Reg))
              continue;
            LV->addVirtualRegisterKilled(Reg, NewMI);
            if (VI.removeKill(MI))
              VI.Kills.push_back(NewMI);
            break;
          }
        }
      }
    }
  }

  MFI->insert(MBBI, NewMIs[1]);
  MFI->insert(MBBI, NewMIs[0]);
  return NewMIs[0];
}

// Branch analysis.
bool
ARMBaseInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(LastInst->getOperand(1));
      Cond.push_back(LastInst->getOperand(2));
      return false;
    }
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB =  SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(SecondLastInst->getOperand(1));
    Cond.push_back(SecondLastInst->getOperand(2));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (isUncondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // ...likewise if it ends with a branch table followed by an unconditional
  // branch. The branch folder can create these, and we must get rid of them for
  // correctness of Thumb constant islands.
  if ((isJumpTableBranchOpcode(SecondLastOpc) ||
       isIndirectBranchOpcode(SecondLastOpc)) &&
      isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}


unsigned ARMBaseInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (!isUncondBranchOpcode(I->getOpcode()) &&
      !isCondBranchOpcode(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!isCondBranchOpcode(I->getOpcode()))
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
ARMBaseInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                               MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc argument
  DebugLoc dl = DebugLoc::getUnknownLoc();

  ARMFunctionInfo *AFI = MBB.getParent()->getInfo<ARMFunctionInfo>();
  int BOpc   = !AFI->isThumbFunction()
    ? ARM::B : (AFI->isThumb2Function() ? ARM::t2B : ARM::tB);
  int BccOpc = !AFI->isThumbFunction()
    ? ARM::Bcc : (AFI->isThumb2Function() ? ARM::t2Bcc : ARM::tBcc);

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");

  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, dl, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, dl, get(BccOpc)).addMBB(TBB)
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
    return 1;
  }

  // Two-way conditional branch.
  BuildMI(&MBB, dl, get(BccOpc)).addMBB(TBB)
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
  BuildMI(&MBB, dl, get(BOpc)).addMBB(FBB);
  return 2;
}

bool ARMBaseInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
  Cond[0].setImm(ARMCC::getOppositeCondition(CC));
  return false;
}

bool ARMBaseInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Pred) const {
  unsigned Opc = MI->getOpcode();
  if (isUncondBranchOpcode(Opc)) {
    MI->setDesc(get(getMatchingCondBranchOpcode(Opc)));
    MI->addOperand(MachineOperand::CreateImm(Pred[0].getImm()));
    MI->addOperand(MachineOperand::CreateReg(Pred[1].getReg(), false));
    return true;
  }

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setImm(Pred[0].getImm());
    MI->getOperand(PIdx+1).setReg(Pred[1].getReg());
    return true;
  }
  return false;
}

bool ARMBaseInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  if (Pred1.size() > 2 || Pred2.size() > 2)
    return false;

  ARMCC::CondCodes CC1 = (ARMCC::CondCodes)Pred1[0].getImm();
  ARMCC::CondCodes CC2 = (ARMCC::CondCodes)Pred2[0].getImm();
  if (CC1 == CC2)
    return true;

  switch (CC1) {
  default:
    return false;
  case ARMCC::AL:
    return true;
  case ARMCC::HS:
    return CC2 == ARMCC::HI;
  case ARMCC::LS:
    return CC2 == ARMCC::LO || CC2 == ARMCC::EQ;
  case ARMCC::GE:
    return CC2 == ARMCC::GT;
  case ARMCC::LE:
    return CC2 == ARMCC::LT;
  }
}

bool ARMBaseInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  // FIXME: This confuses implicit_def with optional CPSR def.
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.getImplicitDefs() && !TID.hasOptionalDef())
    return false;

  bool Found = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == ARM::CPSR) {
      Pred.push_back(MO);
      Found = true;
    }
  }

  return Found;
}


/// FIXME: Works around a gcc miscompilation with -fstrict-aliasing
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) DISABLE_INLINE;
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) {
  return JT[JTI].MBBs.size();
}

/// GetInstSize - Return the size of the specified MachineInstr.
///
unsigned ARMBaseInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();

  // Basic size info comes from the TSFlags field.
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned TSFlags = TID.TSFlags;

  unsigned Opc = MI->getOpcode();
  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default: {
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return getInlineAsmLength(MI->getOperand(0).getSymbolName(), *MAI);
    if (MI->isLabel())
      return 0;
    switch (Opc) {
    default:
      llvm_unreachable("Unknown or unset size field for instr!");
    case TargetInstrInfo::IMPLICIT_DEF:
    case TargetInstrInfo::KILL:
    case TargetInstrInfo::DBG_LABEL:
    case TargetInstrInfo::EH_LABEL:
      return 0;
    }
    break;
  }
  case ARMII::Size8Bytes: return 8;          // ARM instruction x 2.
  case ARMII::Size4Bytes: return 4;          // ARM / Thumb2 instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb1 instruction.
  case ARMII::SizeSpecial: {
    switch (Opc) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::Int_eh_sjlj_setjmp:
      return 24;
    case ARM::t2Int_eh_sjlj_setjmp:
      return 22;
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd:
    case ARM::tBR_JTr:
    case ARM::t2BR_JT:
    case ARM::t2TBB:
    case ARM::t2TBH: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries. For TBB, each
      // entry is one byte; TBH two byte each.
      unsigned EntrySize = (Opc == ARM::t2TBB)
        ? 1 : ((Opc == ARM::t2TBH) ? 2 : 4);
      unsigned NumOps = TID.getNumOperands();
      MachineOperand JTOP =
        MI->getOperand(NumOps - (TID.isPredicable() ? 3 : 2));
      unsigned JTI = JTOP.getIndex();
      const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      // Thumb instructions are 2 byte aligned, but JT entries are 4 byte
      // 4 aligned. The assembler / linker may add 2 byte padding just before
      // the JT entries.  The size does not include this padding; the
      // constant islands pass does separate bookkeeping for it.
      // FIXME: If we know the size of the function is less than (1 << 16) *2
      // bytes, we can use 16-bit entries instead. Then there won't be an
      // alignment issue.
      unsigned InstSize = (Opc == ARM::tBR_JTr || Opc == ARM::t2BR_JT) ? 2 : 4;
      unsigned NumEntries = getNumJTEntries(JT, JTI);
      if (Opc == ARM::t2TBB && (NumEntries & 1))
        // Make sure the instruction that follows TBB is 2-byte aligned.
        // FIXME: Constant island pass should insert an "ALIGN" instruction
        // instead.
        ++NumEntries;
      return NumEntries * EntrySize + InstSize;
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
  }
  return 0; // Not reached
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool
ARMBaseInstrInfo::isMoveInstr(const MachineInstr &MI,
                              unsigned &SrcReg, unsigned &DstReg,
                              unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  switch (MI.getOpcode()) {
  default: break;
  case ARM::FCPYS:
  case ARM::FCPYD:
  case ARM::VMOVD:
  case ARM::VMOVQ: {
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
  case ARM::MOVr:
  case ARM::tMOVr:
  case ARM::tMOVgpr2tgpr:
  case ARM::tMOVtgpr2gpr:
  case ARM::tMOVgpr2gpr:
  case ARM::t2MOVr: {
    assert(MI.getDesc().getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "Invalid ARM MOV instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
  }

  return false;
}

unsigned
ARMBaseInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDR:
  case ARM::t2LDRs:  // FIXME: don't use t2LDRs to access frame.
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::t2LDRi12:
  case ARM::tRestore:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FLDD:
  case ARM::FLDS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

unsigned
ARMBaseInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                     int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STR:
  case ARM::t2STRs: // FIXME: don't use t2STRs to access frame.
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::t2STRi12:
  case ARM::tSpill:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FSTD:
  case ARM::FSTS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

bool
ARMBaseInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               unsigned DestReg, unsigned SrcReg,
                               const TargetRegisterClass *DestRC,
                               const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (DestRC != SrcRC) {
    if (DestRC->getSize() != SrcRC->getSize())
      return false;

    // Allow DPR / DPR_VFP2 / DPR_8 cross-class copies.
    // Allow QPR / QPR_VFP2 / QPR_8 cross-class copies.
    if (DestRC->getSize() != 8 && DestRC->getSize() != 16)
      return false;
  }

  if (DestRC == ARM::GPRRegisterClass) {
    AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::MOVr),
                                        DestReg).addReg(SrcReg)));
  } else if (DestRC == ARM::SPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FCPYS), DestReg)
                   .addReg(SrcReg));
  } else if (DestRC == ARM::DPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FCPYD), DestReg)
                   .addReg(SrcReg));
  } else if (DestRC == ARM::DPR_VFP2RegisterClass ||
             DestRC == ARM::DPR_8RegisterClass ||
             SrcRC == ARM::DPR_VFP2RegisterClass ||
             SrcRC == ARM::DPR_8RegisterClass) {
    // Always use neon reg-reg move if source or dest is NEON-only regclass.
    BuildMI(MBB, I, DL, get(ARM::VMOVD), DestReg).addReg(SrcReg);
  } else if (DestRC == ARM::QPRRegisterClass ||
             DestRC == ARM::QPR_VFP2RegisterClass ||
             DestRC == ARM::QPR_8RegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::VMOVQ), DestReg).addReg(SrcReg);
  } else {
    return false;
  }

  return true;
}

void ARMBaseInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                            MachineMemOperand::MOStore, 0,
                            MFI.getObjectSize(FI),
                            Align);

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::STR))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addReg(0).addImm(0).addMemOperand(MMO));
  } else if (RC == ARM::DPRRegisterClass ||
             RC == ARM::DPR_VFP2RegisterClass ||
             RC == ARM::DPR_8RegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FSTD))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  } else if (RC == ARM::SPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FSTS))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  } else {
    assert((RC == ARM::QPRRegisterClass ||
            RC == ARM::QPR_VFP2RegisterClass) && "Unknown regclass!");
    // FIXME: Neon instructions should support predicates
    if (Align >= 16
        && (getRegisterInfo().needsStackRealignment(MF))) {
      BuildMI(MBB, I, DL, get(ARM::VST1q64))
        .addFrameIndex(FI).addImm(0).addImm(0).addImm(128).addMemOperand(MMO)
        .addReg(SrcReg, getKillRegState(isKill));
    } else {
      BuildMI(MBB, I, DL, get(ARM::VSTRQ)).
        addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
    }
  }
}

void ARMBaseInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                            MachineMemOperand::MOLoad, 0,
                            MFI.getObjectSize(FI),
                            Align);

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::LDR), DestReg)
                   .addFrameIndex(FI).addReg(0).addImm(0).addMemOperand(MMO));
  } else if (RC == ARM::DPRRegisterClass ||
             RC == ARM::DPR_VFP2RegisterClass ||
             RC == ARM::DPR_8RegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FLDD), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  } else if (RC == ARM::SPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FLDS), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  } else {
    assert((RC == ARM::QPRRegisterClass ||
            RC == ARM::QPR_VFP2RegisterClass ||
            RC == ARM::QPR_8RegisterClass) && "Unknown regclass!");
    // FIXME: Neon instructions should support predicates
    if (Align >= 16
        && (getRegisterInfo().needsStackRealignment(MF))) {
      BuildMI(MBB, I, DL, get(ARM::VLD1q64))
        .addReg(DestReg)
        .addFrameIndex(FI).addImm(0).addImm(0).addImm(128).addMemOperand(MMO);
    } else {
      BuildMI(MBB, I, DL, get(ARM::VLDRQ), DestReg).addFrameIndex(FI).addImm(0).
        addMemOperand(MMO);
    }
  }
}

MachineInstr *ARMBaseInstrInfo::
foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const {
  if (Ops.size() != 1) return NULL;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  if (Opc == ARM::MOVr || Opc == ARM::t2MOVr) {
    // If it is updating CPSR, then it cannot be folded.
    if (MI->getOperand(4).getReg() == ARM::CPSR && !MI->getOperand(4).isDead())
      return NULL;
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      unsigned SrcSubReg = MI->getOperand(1).getSubReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      if (Opc == ARM::MOVr)
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::STR))
          .addReg(SrcReg,
                  getKillRegState(isKill) | getUndefRegState(isUndef),
                  SrcSubReg)
          .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
      else // ARM::t2MOVr
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::t2STRi12))
          .addReg(SrcReg,
                  getKillRegState(isKill) | getUndefRegState(isUndef),
                  SrcSubReg)
          .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned DstSubReg = MI->getOperand(0).getSubReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      if (Opc == ARM::MOVr)
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::LDR))
          .addReg(DstReg,
                  RegState::Define |
                  getDeadRegState(isDead) |
                  getUndefRegState(isUndef), DstSubReg)
          .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
      else // ARM::t2MOVr
        NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::t2LDRi12))
          .addReg(DstReg,
                  RegState::Define |
                  getDeadRegState(isDead) |
                  getUndefRegState(isUndef), DstSubReg)
          .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
  } else if (Opc == ARM::tMOVgpr2gpr ||
             Opc == ARM::tMOVtgpr2gpr ||
             Opc == ARM::tMOVgpr2tgpr) {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      unsigned SrcSubReg = MI->getOperand(1).getSubReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::t2STRi12))
        .addReg(SrcReg,
                getKillRegState(isKill) | getUndefRegState(isUndef),
                SrcSubReg)
        .addFrameIndex(FI).addImm(0).addImm(ARMCC::AL).addReg(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned DstSubReg = MI->getOperand(0).getSubReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::t2LDRi12))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef),
                DstSubReg)
        .addFrameIndex(FI).addImm(0).addImm(ARMCC::AL).addReg(0);
    }
  } else if (Opc == ARM::FCPYS) {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      unsigned SrcSubReg = MI->getOperand(1).getSubReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FSTS))
        .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef),
                SrcSubReg)
        .addFrameIndex(FI)
        .addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned DstSubReg = MI->getOperand(0).getSubReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FLDS))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef),
                DstSubReg)
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
  }
  else if (Opc == ARM::FCPYD) {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      unsigned SrcSubReg = MI->getOperand(1).getSubReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FSTD))
        .addReg(SrcReg,
                getKillRegState(isKill) | getUndefRegState(isUndef),
                SrcSubReg)
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned DstSubReg = MI->getOperand(0).getSubReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FLDD))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef),
                DstSubReg)
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
  }

  return NewMI;
}

MachineInstr*
ARMBaseInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                        MachineInstr* MI,
                                        const SmallVectorImpl<unsigned> &Ops,
                                        MachineInstr* LoadMI) const {
  // FIXME
  return 0;
}

bool
ARMBaseInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                   const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::MOVr || Opc == ARM::t2MOVr) {
    // If it is updating CPSR, then it cannot be folded.
    return MI->getOperand(4).getReg() != ARM::CPSR ||
      MI->getOperand(4).isDead();
  } else if (Opc == ARM::tMOVgpr2gpr ||
             Opc == ARM::tMOVtgpr2gpr ||
             Opc == ARM::tMOVgpr2tgpr) {
    return true;
  } else if (Opc == ARM::FCPYS || Opc == ARM::FCPYD) {
    return true;
  } else if (Opc == ARM::VMOVD || Opc == ARM::VMOVQ) {
    return false; // FIXME
  }

  return false;
}

void ARMBaseInstrInfo::
reMaterialize(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator I,
              unsigned DestReg, unsigned SubIdx,
              const MachineInstr *Orig) const {
  DebugLoc dl = Orig->getDebugLoc();
  unsigned Opcode = Orig->getOpcode();
  switch (Opcode) {
  default: {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
    break;
  }
  case ARM::tLDRpci_pic:
  case ARM::t2LDRpci_pic: {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    MachineConstantPool *MCP = MF.getConstantPool();
    unsigned CPI = Orig->getOperand(1).getIndex();
    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPI];
    assert(MCPE.isMachineConstantPoolEntry() &&
           "Expecting a machine constantpool entry!");
    ARMConstantPoolValue *ACPV =
      static_cast<ARMConstantPoolValue*>(MCPE.Val.MachineCPVal);
    unsigned PCLabelId = AFI->createConstPoolEntryUId();
    ARMConstantPoolValue *NewCPV = 0;
    if (ACPV->isGlobalValue())
      NewCPV = new ARMConstantPoolValue(ACPV->getGV(), PCLabelId,
                                        ARMCP::CPValue, 4);
    else if (ACPV->isExtSymbol())
      NewCPV = new ARMConstantPoolValue(MF.getFunction()->getContext(),
                                        ACPV->getSymbol(), PCLabelId, 4);
    else if (ACPV->isBlockAddress())
      NewCPV = new ARMConstantPoolValue(ACPV->getBlockAddress(), PCLabelId,
                                        ARMCP::CPBlockAddress, 4);
    else
      llvm_unreachable("Unexpected ARM constantpool value type!!");
    CPI = MCP->getConstantPoolIndex(NewCPV, MCPE.getAlignment());
    MachineInstrBuilder MIB = BuildMI(MBB, I, Orig->getDebugLoc(), get(Opcode),
                                      DestReg)
      .addConstantPoolIndex(CPI).addImm(PCLabelId);
    (*MIB).setMemRefs(Orig->memoperands_begin(), Orig->memoperands_end());
    break;
  }
  }

  MachineInstr *NewMI = prior(I);
  NewMI->getOperand(0).setSubReg(SubIdx);
}

bool ARMBaseInstrInfo::isIdentical(const MachineInstr *MI0,
                                  const MachineInstr *MI1,
                                  const MachineRegisterInfo *MRI) const {
  int Opcode = MI0->getOpcode();
  if (Opcode == ARM::t2LDRpci_pic || Opcode == ARM::tLDRpci_pic) {
    if (MI1->getOpcode() != Opcode)
      return false;
    if (MI0->getNumOperands() != MI1->getNumOperands())
      return false;

    const MachineOperand &MO0 = MI0->getOperand(1);
    const MachineOperand &MO1 = MI1->getOperand(1);
    if (MO0.getOffset() != MO1.getOffset())
      return false;

    const MachineFunction *MF = MI0->getParent()->getParent();
    const MachineConstantPool *MCP = MF->getConstantPool();
    int CPI0 = MO0.getIndex();
    int CPI1 = MO1.getIndex();
    const MachineConstantPoolEntry &MCPE0 = MCP->getConstants()[CPI0];
    const MachineConstantPoolEntry &MCPE1 = MCP->getConstants()[CPI1];
    ARMConstantPoolValue *ACPV0 =
      static_cast<ARMConstantPoolValue*>(MCPE0.Val.MachineCPVal);
    ARMConstantPoolValue *ACPV1 =
      static_cast<ARMConstantPoolValue*>(MCPE1.Val.MachineCPVal);
    return ACPV0->hasSameValue(ACPV1);
  }

  return TargetInstrInfoImpl::isIdentical(MI0, MI1, MRI);
}

/// getInstrPredicate - If instruction is predicated, returns its predicate
/// condition, otherwise returns AL. It also returns the condition code
/// register by reference.
ARMCC::CondCodes
llvm::getInstrPredicate(const MachineInstr *MI, unsigned &PredReg) {
  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx == -1) {
    PredReg = 0;
    return ARMCC::AL;
  }

  PredReg = MI->getOperand(PIdx+1).getReg();
  return (ARMCC::CondCodes)MI->getOperand(PIdx).getImm();
}


int llvm::getMatchingCondBranchOpcode(int Opc) {
  if (Opc == ARM::B)
    return ARM::Bcc;
  else if (Opc == ARM::tB)
    return ARM::tBcc;
  else if (Opc == ARM::t2B)
      return ARM::t2Bcc;

  llvm_unreachable("Unknown unconditional branch opcode!");
  return 0;
}


void llvm::emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                               unsigned DestReg, unsigned BaseReg, int NumBytes,
                               ARMCC::CondCodes Pred, unsigned PredReg,
                               const ARMBaseInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  if (isSub) NumBytes = -NumBytes;

  while (NumBytes) {
    unsigned RotAmt = ARM_AM::getSOImmValRotate(NumBytes);
    unsigned ThisVal = NumBytes & ARM_AM::rotr32(0xFF, RotAmt);
    assert(ThisVal && "Didn't extract field correctly");

    // We will handle these bits from offset, clear them.
    NumBytes &= ~ThisVal;

    assert(ARM_AM::getSOImmVal(ThisVal) != -1 && "Bit extraction didn't work?");

    // Build the new ADD / SUB.
    unsigned Opc = isSub ? ARM::SUBri : ARM::ADDri;
    BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg)
      .addReg(BaseReg, RegState::Kill).addImm(ThisVal)
      .addImm((unsigned)Pred).addReg(PredReg).addReg(0);
    BaseReg = DestReg;
  }
}

bool llvm::rewriteARMFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                unsigned FrameReg, int &Offset,
                                const ARMBaseInstrInfo &TII) {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  bool isSub = false;

  // Memory operands in inline assembly always use AddrMode2.
  if (Opcode == ARM::INLINEASM)
    AddrMode = ARMII::AddrMode2;

  if (Opcode == ARM::ADDri) {
    Offset += MI.getOperand(FrameRegIdx+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::MOVr));
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(FrameRegIdx+1);
      Offset = 0;
      return true;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setDesc(TII.get(ARM::SUBri));
    }

    // Common case: small offset, fits into instruction.
    if (ARM_AM::getSOImmVal(Offset) != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.getOperand(FrameRegIdx+1).ChangeToImmediate(Offset);
      Offset = 0;
      return true;
    }

    // Otherwise, pull as much of the immedidate into this ADDri/SUBri
    // as possible.
    unsigned RotAmt = ARM_AM::getSOImmValRotate(Offset);
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xFF, RotAmt);

    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;

    // Get the properly encoded SOImmVal field.
    assert(ARM_AM::getSOImmVal(ThisImmVal) != -1 &&
           "Bit extraction didn't work?");
    MI.getOperand(FrameRegIdx+1).ChangeToImmediate(ThisImmVal);
 } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrMode2: {
      ImmIdx = FrameRegIdx+2;
      InstrOffs = ARM_AM::getAM2Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM2Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 12;
      break;
    }
    case ARMII::AddrMode3: {
      ImmIdx = FrameRegIdx+2;
      InstrOffs = ARM_AM::getAM3Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM3Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      break;
    }
    case ARMII::AddrMode4:
      // Can't fold any offset even if it's zero.
      return false;
    case ARMII::AddrMode5: {
      ImmIdx = FrameRegIdx+1;
      InstrOffs = ARM_AM::getAM5Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM5Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      Scale = 4;
      break;
    }
    default:
      llvm_unreachable("Unsupported addressing mode!");
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale-1)) == 0 && "Can't encode this offset!");
    if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
    }

    // Attempt to fold address comp. if opcode has offset bits
    if (NumBits > 0) {
      // Common case: small offset, fits into instruction.
      MachineOperand &ImmOp = MI.getOperand(ImmIdx);
      int ImmedOffset = Offset / Scale;
      unsigned Mask = (1 << NumBits) - 1;
      if ((unsigned)Offset <= Mask * Scale) {
        // Replace the FrameIndex with sp
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        if (isSub)
          ImmedOffset |= 1 << NumBits;
        ImmOp.ChangeToImmediate(ImmedOffset);
        Offset = 0;
        return true;
      }

      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask*Scale);
    }
  }

  Offset = (isSub) ? -Offset : Offset;
  return Offset == 0;
}
