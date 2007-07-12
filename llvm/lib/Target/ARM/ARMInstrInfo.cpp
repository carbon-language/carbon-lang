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
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
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
  case ARM::MOVr:
  case ARM::tMOVr:
    assert(MI.getInstrDescriptor()->numOperands >= 2 &&
           MI.getOperand(0).isRegister() &&
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
  case ARM::tRestore:
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
  case ARM::tSpill:
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
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  unsigned NumOps = TID->numOperands;
  bool isLoad = (TID->Flags & M_LOAD_FLAG) != 0;
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
      int SOImmVal = ARM_AM::getSOImmVal(Amt);
      if (SOImmVal == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(SOImmVal)
        .addImm(Pred).addReg(0).addReg(0);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc)
        .addImm(Pred).addReg(0).addReg(0);
    } else 
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    else
      UpdateMI = BuildMI(get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
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
          int NIdx = NewMI->findRegisterUseOperandIdx(Reg);
          if (NIdx == -1)
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
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastOpc == ARM::B || LastOpc == ARM::tB) {
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      return false;
    }
    if (LastOpc == ARM::Bcc || LastOpc == ARM::tBcc) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
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
  
  // If the block ends with ARM::B/ARM::tB and a ARM::Bcc/ARM::tBcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if ((SecondLastOpc == ARM::Bcc && LastOpc == ARM::B) ||
      (SecondLastOpc == ARM::tBcc && LastOpc == ARM::tB)) {
    TBB =  SecondLastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(SecondLastInst->getOperand(1));
    Cond.push_back(SecondLastInst->getOperand(2));
    FBB = LastInst->getOperand(0).getMachineBasicBlock();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it.  The second 
  // one is not executed, so remove it.
  if ((SecondLastOpc == ARM::B || SecondLastOpc==ARM::tB) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB)) {
    TBB = SecondLastInst->getOperand(0).getMachineBasicBlock();
    I = LastInst;
    I->eraseFromParent();
    return false;
  }

  // Likewise if it ends with a branch table followed by an unconditional branch.
  // The branch folder can create these, and we must get rid of them for
  // correctness of Thumb constant islands.
  if ((SecondLastOpc == ARM::BR_JTr || SecondLastOpc==ARM::BR_JTm ||
       SecondLastOpc == ARM::BR_JTadd || SecondLastOpc==ARM::tBR_JTr) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB)) {
    I = LastInst;
    I->eraseFromParent();
    return true;
  } 

  // Otherwise, can't handle this.
  return true;
}


unsigned ARMInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != BccOpc)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned ARMInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? ARM::tB : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? ARM::tBcc : ARM::Bcc;

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");
  
  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, get(BccOpc)).addMBB(TBB)
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
    return 1;
  }
  
  // Two-way conditional branch.
  BuildMI(&MBB, get(BccOpc)).addMBB(TBB)
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
  BuildMI(&MBB, get(BOpc)).addMBB(FBB);
  return 2;
}

bool ARMInstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case ARM::BX_RET:   // Return.
  case ARM::LDM_RET:
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::B:
  case ARM::tB:       // Uncond branch.
  case ARM::tBR_JTr:
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

bool ARMInstrInfo::isPredicated(const MachineInstr *MI) const {
  int PIdx = MI->findFirstPredOperandIdx();
  return PIdx != -1 && MI->getOperand(PIdx).getImmedValue() != ARMCC::AL;
}

bool ARMInstrInfo::PredicateInstruction(MachineInstr *MI,
                                const std::vector<MachineOperand> &Pred) const {
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::B || Opc == ARM::tB) {
    MI->setInstrDescriptor(get(Opc == ARM::B ? ARM::Bcc : ARM::tBcc));
    MI->addImmOperand(Pred[0].getImmedValue());
    MI->addRegOperand(Pred[1].getReg(), false);
    return true;
  }

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setImm(Pred[0].getImmedValue());
    MI->getOperand(PIdx+1).setReg(Pred[1].getReg());
    return true;
  }
  return false;
}

bool
ARMInstrInfo::SubsumesPredicate(const std::vector<MachineOperand> &Pred1,
                                const std::vector<MachineOperand> &Pred2) const{
  if (Pred1.size() > 2 || Pred2.size() > 2)
    return false;

  ARMCC::CondCodes CC1 = (ARMCC::CondCodes)Pred1[0].getImmedValue();
  ARMCC::CondCodes CC2 = (ARMCC::CondCodes)Pred2[0].getImmedValue();
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

bool ARMInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  if (!TID->ImplicitDefs && (TID->Flags & M_HAS_OPTIONAL_DEF) == 0)
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
unsigned ARM::GetInstSize(MachineInstr *MI) {
  MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const TargetAsmInfo *TAI = MF->getTarget().getTargetAsmInfo();

  // Basic size info comes from the TSFlags field.
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  unsigned TSFlags = TID->TSFlags;
  
  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default:
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return TAI->getInlineAsmLength(MI->getOperand(0).getSymbolName());
    if (MI->getOpcode() == ARM::LABEL)
      return 0;
    assert(0 && "Unknown or unset size field for instr!");
    break;
  case ARMII::Size8Bytes: return 8;          // Arm instruction x 2.
  case ARMII::Size4Bytes: return 4;          // Arm instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb instruction.
  case ARMII::SizeSpecial: {
    switch (MI->getOpcode()) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd:
    case ARM::tBR_JTr: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries.
      unsigned NumOps = TID->numOperands;
      MachineOperand JTOP =
        MI->getOperand(NumOps - ((TID->Flags & M_PREDICABLE) ? 3 : 2));
      unsigned JTI = JTOP.getJumpTableIndex();
      MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      // Thumb instructions are 2 byte aligned, but JT entries are 4 byte
      // 4 aligned. The assembler / linker may add 2 byte padding just before
      // the JT entries.  The size does not include this padding; the
      // constant islands pass does separate bookkeeping for it.
      // FIXME: If we know the size of the function is less than (1 << 16) *2
      // bytes, we can use 16-bit entries instead. Then there won't be an
      // alignment issue.
      return getNumJTEntries(JT, JTI) * 4 + 
             (MI->getOpcode()==ARM::tBR_JTr ? 2 : 4);
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
  }
}

/// GetFunctionSize - Returns the size of the specified MachineFunction.
///
unsigned ARM::GetFunctionSize(MachineFunction &MF) {
  unsigned FnSize = 0;
  for (MachineFunction::iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::iterator I = MBB.begin(),E = MBB.end(); I != E; ++I)
      FnSize += ARM::GetInstSize(I);
  }
  return FnSize;
}
