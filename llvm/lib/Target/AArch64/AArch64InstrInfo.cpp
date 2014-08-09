//===- AArch64InstrInfo.cpp - AArch64 Instruction Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "AArch64MachineCombinerPattern.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "AArch64GenInstrInfo.inc"

AArch64InstrInfo::AArch64InstrInfo(const AArch64Subtarget &STI)
    : AArch64GenInstrInfo(AArch64::ADJCALLSTACKDOWN, AArch64::ADJCALLSTACKUP),
      RI(this, &STI), Subtarget(STI) {}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
unsigned AArch64InstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();

  if (MI->getOpcode() == AArch64::INLINEASM)
    return getInlineAsmLength(MI->getOperand(0).getSymbolName(), *MAI);

  const MCInstrDesc &Desc = MI->getDesc();
  switch (Desc.getOpcode()) {
  default:
    // Anything not explicitly designated otherwise is a nomal 4-byte insn.
    return 4;
  case TargetOpcode::DBG_VALUE:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
    return 0;
  }

  llvm_unreachable("GetInstSizeInBytes()- Unable to determin insn size");
}

static void parseCondBranch(MachineInstr *LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  switch (LastInst->getOpcode()) {
  default:
    llvm_unreachable("Unknown branch instruction?");
  case AArch64::Bcc:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(LastInst->getOperand(0));
    break;
  case AArch64::CBZW:
  case AArch64::CBZX:
  case AArch64::CBNZW:
  case AArch64::CBNZX:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    break;
  case AArch64::TBZW:
  case AArch64::TBZX:
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    Target = LastInst->getOperand(2).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    Cond.push_back(LastInst->getOperand(1));
  }
}

// Branch analysis.
bool AArch64InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
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
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      parseCondBranch(LastInst, TBB, Cond);
      return false;
    }
    return true; // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && isUncondBranchOpcode(LastOpc)) {
    while (isUncondBranchOpcode(SecondLastOpc)) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      } else {
        SecondLastInst = I;
        SecondLastOpc = SecondLastInst->getOpcode();
      }
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    parseCondBranch(SecondLastInst, TBB, Cond);
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

  // ...likewise if it ends with an indirect branch followed by an unconditional
  // branch.
  if (isIndirectBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}

bool AArch64InstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    AArch64CC::CondCode CC = (AArch64CC::CondCode)(int)Cond[0].getImm();
    Cond[0].setImm(AArch64CC::getInvertedCondCode(CC));
  } else {
    // Folded compare-and-branch
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown conditional branch!");
    case AArch64::CBZW:
      Cond[1].setImm(AArch64::CBNZW);
      break;
    case AArch64::CBNZW:
      Cond[1].setImm(AArch64::CBZW);
      break;
    case AArch64::CBZX:
      Cond[1].setImm(AArch64::CBNZX);
      break;
    case AArch64::CBNZX:
      Cond[1].setImm(AArch64::CBZX);
      break;
    case AArch64::TBZW:
      Cond[1].setImm(AArch64::TBNZW);
      break;
    case AArch64::TBNZW:
      Cond[1].setImm(AArch64::TBZW);
      break;
    case AArch64::TBZX:
      Cond[1].setImm(AArch64::TBNZX);
      break;
    case AArch64::TBNZX:
      Cond[1].setImm(AArch64::TBZX);
      break;
    }
  }

  return false;
}

unsigned AArch64InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (!isUncondBranchOpcode(I->getOpcode()) &&
      !isCondBranchOpcode(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!isCondBranchOpcode(I->getOpcode()))
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

void AArch64InstrInfo::instantiateCondBranch(
    MachineBasicBlock &MBB, DebugLoc DL, MachineBasicBlock *TBB,
    const SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    BuildMI(&MBB, DL, get(AArch64::Bcc)).addImm(Cond[0].getImm()).addMBB(TBB);
  } else {
    // Folded compare-and-branch
    const MachineInstrBuilder MIB =
        BuildMI(&MBB, DL, get(Cond[1].getImm())).addReg(Cond[2].getReg());
    if (Cond.size() > 3)
      MIB.addImm(Cond[3].getImm());
    MIB.addMBB(TBB);
  }
}

unsigned AArch64InstrInfo::InsertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    const SmallVectorImpl<MachineOperand> &Cond, DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  if (!FBB) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, DL, get(AArch64::B)).addMBB(TBB);
    else
      instantiateCondBranch(MBB, DL, TBB, Cond);
    return 1;
  }

  // Two-way conditional branch.
  instantiateCondBranch(MBB, DL, TBB, Cond);
  BuildMI(&MBB, DL, get(AArch64::B)).addMBB(FBB);
  return 2;
}

// Find the original register that VReg is copied from.
static unsigned removeCopies(const MachineRegisterInfo &MRI, unsigned VReg) {
  while (TargetRegisterInfo::isVirtualRegister(VReg)) {
    const MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (!DefMI->isFullCopy())
      return VReg;
    VReg = DefMI->getOperand(1).getReg();
  }
  return VReg;
}

// Determine if VReg is defined by an instruction that can be folded into a
// csel instruction. If so, return the folded opcode, and the replacement
// register.
static unsigned canFoldIntoCSel(const MachineRegisterInfo &MRI, unsigned VReg,
                                unsigned *NewVReg = nullptr) {
  VReg = removeCopies(MRI, VReg);
  if (!TargetRegisterInfo::isVirtualRegister(VReg))
    return 0;

  bool Is64Bit = AArch64::GPR64allRegClass.hasSubClassEq(MRI.getRegClass(VReg));
  const MachineInstr *DefMI = MRI.getVRegDef(VReg);
  unsigned Opc = 0;
  unsigned SrcOpNum = 0;
  switch (DefMI->getOpcode()) {
  case AArch64::ADDSXri:
  case AArch64::ADDSWri:
    // if NZCV is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(AArch64::NZCV, true) == -1)
      return 0;
  // fall-through to ADDXri and ADDWri.
  case AArch64::ADDXri:
  case AArch64::ADDWri:
    // add x, 1 -> csinc.
    if (!DefMI->getOperand(2).isImm() || DefMI->getOperand(2).getImm() != 1 ||
        DefMI->getOperand(3).getImm() != 0)
      return 0;
    SrcOpNum = 1;
    Opc = Is64Bit ? AArch64::CSINCXr : AArch64::CSINCWr;
    break;

  case AArch64::ORNXrr:
  case AArch64::ORNWrr: {
    // not x -> csinv, represented as orn dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != AArch64::XZR && ZReg != AArch64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? AArch64::CSINVXr : AArch64::CSINVWr;
    break;
  }

  case AArch64::SUBSXrr:
  case AArch64::SUBSWrr:
    // if NZCV is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(AArch64::NZCV, true) == -1)
      return 0;
  // fall-through to SUBXrr and SUBWrr.
  case AArch64::SUBXrr:
  case AArch64::SUBWrr: {
    // neg x -> csneg, represented as sub dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != AArch64::XZR && ZReg != AArch64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? AArch64::CSNEGXr : AArch64::CSNEGWr;
    break;
  }
  default:
    return 0;
  }
  assert(Opc && SrcOpNum && "Missing parameters");

  if (NewVReg)
    *NewVReg = DefMI->getOperand(SrcOpNum).getReg();
  return Opc;
}

bool AArch64InstrInfo::canInsertSelect(
    const MachineBasicBlock &MBB, const SmallVectorImpl<MachineOperand> &Cond,
    unsigned TrueReg, unsigned FalseReg, int &CondCycles, int &TrueCycles,
    int &FalseCycles) const {
  // Check register classes.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
      RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  if (!RC)
    return false;

  // Expanding cbz/tbz requires an extra cycle of latency on the condition.
  unsigned ExtraCondLat = Cond.size() != 1;

  // GPRs are handled by csel.
  // FIXME: Fold in x+1, -x, and ~x when applicable.
  if (AArch64::GPR64allRegClass.hasSubClassEq(RC) ||
      AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
    // Single-cycle csel, csinc, csinv, and csneg.
    CondCycles = 1 + ExtraCondLat;
    TrueCycles = FalseCycles = 1;
    if (canFoldIntoCSel(MRI, TrueReg))
      TrueCycles = 0;
    else if (canFoldIntoCSel(MRI, FalseReg))
      FalseCycles = 0;
    return true;
  }

  // Scalar floating point is handled by fcsel.
  // FIXME: Form fabs, fmin, and fmax when applicable.
  if (AArch64::FPR64RegClass.hasSubClassEq(RC) ||
      AArch64::FPR32RegClass.hasSubClassEq(RC)) {
    CondCycles = 5 + ExtraCondLat;
    TrueCycles = FalseCycles = 2;
    return true;
  }

  // Can't do vectors.
  return false;
}

void AArch64InstrInfo::insertSelect(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I, DebugLoc DL,
                                    unsigned DstReg,
                                    const SmallVectorImpl<MachineOperand> &Cond,
                                    unsigned TrueReg, unsigned FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Parse the condition code, see parseCondBranch() above.
  AArch64CC::CondCode CC;
  switch (Cond.size()) {
  default:
    llvm_unreachable("Unknown condition opcode in Cond");
  case 1: // b.cc
    CC = AArch64CC::CondCode(Cond[0].getImm());
    break;
  case 3: { // cbz/cbnz
    // We must insert a compare against 0.
    bool Is64Bit;
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown branch opcode in Cond");
    case AArch64::CBZW:
      Is64Bit = 0;
      CC = AArch64CC::EQ;
      break;
    case AArch64::CBZX:
      Is64Bit = 1;
      CC = AArch64CC::EQ;
      break;
    case AArch64::CBNZW:
      Is64Bit = 0;
      CC = AArch64CC::NE;
      break;
    case AArch64::CBNZX:
      Is64Bit = 1;
      CC = AArch64CC::NE;
      break;
    }
    unsigned SrcReg = Cond[2].getReg();
    if (Is64Bit) {
      // cmp reg, #0 is actually subs xzr, reg, #0.
      MRI.constrainRegClass(SrcReg, &AArch64::GPR64spRegClass);
      BuildMI(MBB, I, DL, get(AArch64::SUBSXri), AArch64::XZR)
          .addReg(SrcReg)
          .addImm(0)
          .addImm(0);
    } else {
      MRI.constrainRegClass(SrcReg, &AArch64::GPR32spRegClass);
      BuildMI(MBB, I, DL, get(AArch64::SUBSWri), AArch64::WZR)
          .addReg(SrcReg)
          .addImm(0)
          .addImm(0);
    }
    break;
  }
  case 4: { // tbz/tbnz
    // We must insert a tst instruction.
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown branch opcode in Cond");
    case AArch64::TBZW:
    case AArch64::TBZX:
      CC = AArch64CC::EQ;
      break;
    case AArch64::TBNZW:
    case AArch64::TBNZX:
      CC = AArch64CC::NE;
      break;
    }
    // cmp reg, #foo is actually ands xzr, reg, #1<<foo.
    if (Cond[1].getImm() == AArch64::TBZW || Cond[1].getImm() == AArch64::TBNZW)
      BuildMI(MBB, I, DL, get(AArch64::ANDSWri), AArch64::WZR)
          .addReg(Cond[2].getReg())
          .addImm(
              AArch64_AM::encodeLogicalImmediate(1ull << Cond[3].getImm(), 32));
    else
      BuildMI(MBB, I, DL, get(AArch64::ANDSXri), AArch64::XZR)
          .addReg(Cond[2].getReg())
          .addImm(
              AArch64_AM::encodeLogicalImmediate(1ull << Cond[3].getImm(), 64));
    break;
  }
  }

  unsigned Opc = 0;
  const TargetRegisterClass *RC = nullptr;
  bool TryFold = false;
  if (MRI.constrainRegClass(DstReg, &AArch64::GPR64RegClass)) {
    RC = &AArch64::GPR64RegClass;
    Opc = AArch64::CSELXr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::GPR32RegClass)) {
    RC = &AArch64::GPR32RegClass;
    Opc = AArch64::CSELWr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::FPR64RegClass)) {
    RC = &AArch64::FPR64RegClass;
    Opc = AArch64::FCSELDrrr;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::FPR32RegClass)) {
    RC = &AArch64::FPR32RegClass;
    Opc = AArch64::FCSELSrrr;
  }
  assert(RC && "Unsupported regclass");

  // Try folding simple instructions into the csel.
  if (TryFold) {
    unsigned NewVReg = 0;
    unsigned FoldedOpc = canFoldIntoCSel(MRI, TrueReg, &NewVReg);
    if (FoldedOpc) {
      // The folded opcodes csinc, csinc and csneg apply the operation to
      // FalseReg, so we need to invert the condition.
      CC = AArch64CC::getInvertedCondCode(CC);
      TrueReg = FalseReg;
    } else
      FoldedOpc = canFoldIntoCSel(MRI, FalseReg, &NewVReg);

    // Fold the operation. Leave any dead instructions for DCE to clean up.
    if (FoldedOpc) {
      FalseReg = NewVReg;
      Opc = FoldedOpc;
      // The extends the live range of NewVReg.
      MRI.clearKillFlags(NewVReg);
    }
  }

  // Pull all virtual register into the appropriate class.
  MRI.constrainRegClass(TrueReg, RC);
  MRI.constrainRegClass(FalseReg, RC);

  // Insert the csel.
  BuildMI(MBB, I, DL, get(Opc), DstReg).addReg(TrueReg).addReg(FalseReg).addImm(
      CC);
}

// FIXME: this implementation should be micro-architecture dependent, so a
// micro-architecture target hook should be introduced here in future.
bool AArch64InstrInfo::isAsCheapAsAMove(const MachineInstr *MI) const {
  if (!Subtarget.isCortexA57() && !Subtarget.isCortexA53())
    return MI->isAsCheapAsAMove();

  switch (MI->getOpcode()) {
  default:
    return false;

  // add/sub on register without shift
  case AArch64::ADDWri:
  case AArch64::ADDXri:
  case AArch64::SUBWri:
  case AArch64::SUBXri:
    return (MI->getOperand(3).getImm() == 0);

  // logical ops on immediate
  case AArch64::ANDWri:
  case AArch64::ANDXri:
  case AArch64::EORWri:
  case AArch64::EORXri:
  case AArch64::ORRWri:
  case AArch64::ORRXri:
    return true;

  // logical ops on register without shift
  case AArch64::ANDWrr:
  case AArch64::ANDXrr:
  case AArch64::BICWrr:
  case AArch64::BICXrr:
  case AArch64::EONWrr:
  case AArch64::EONXrr:
  case AArch64::EORWrr:
  case AArch64::EORXrr:
  case AArch64::ORNWrr:
  case AArch64::ORNXrr:
  case AArch64::ORRWrr:
  case AArch64::ORRXrr:
    return true;
  }

  llvm_unreachable("Unknown opcode to check as cheap as a move!");
}

bool AArch64InstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                             unsigned &SrcReg, unsigned &DstReg,
                                             unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::SBFMXri: // aka sxtw
  case AArch64::UBFMXri: // aka uxtw
    // Check for the 32 -> 64 bit extension case, these instructions can do
    // much more.
    if (MI.getOperand(2).getImm() != 0 || MI.getOperand(3).getImm() != 31)
      return false;
    // This is a signed or unsigned 32 -> 64 bit extension.
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SubIdx = AArch64::sub_32;
    return true;
  }
}

/// analyzeCompare - For a comparison instruction, return the source registers
/// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
/// Return true if the comparison instruction can be analyzed.
bool AArch64InstrInfo::analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                                      unsigned &SrcReg2, int &CmpMask,
                                      int &CmpValue) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::SUBSWrr:
  case AArch64::SUBSWrs:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrr:
  case AArch64::SUBSXrs:
  case AArch64::SUBSXrx:
  case AArch64::ADDSWrr:
  case AArch64::ADDSWrs:
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXrs:
  case AArch64::ADDSXrx:
    // Replace SUBSWrr with SUBWrr if NZCV is not used.
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = MI->getOperand(2).getReg();
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case AArch64::SUBSWri:
  case AArch64::ADDSWri:
  case AArch64::SUBSXri:
  case AArch64::ADDSXri:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    // FIXME: In order to convert CmpValue to 0 or 1
    CmpValue = (MI->getOperand(2).getImm() != 0);
    return true;
  case AArch64::ANDSWri:
  case AArch64::ANDSXri:
    // ANDS does not use the same encoding scheme as the others xxxS
    // instructions.
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    // FIXME:The return val type of decodeLogicalImmediate is uint64_t,
    // while the type of CmpValue is int. When converting uint64_t to int,
    // the high 32 bits of uint64_t will be lost.
    // In fact it causes a bug in spec2006-483.xalancbmk
    // CmpValue is only used to compare with zero in OptimizeCompareInstr
    CmpValue = (AArch64_AM::decodeLogicalImmediate(
                    MI->getOperand(2).getImm(),
                    MI->getOpcode() == AArch64::ANDSWri ? 32 : 64) != 0);
    return true;
  }

  return false;
}

static bool UpdateOperandRegClass(MachineInstr *Instr) {
  MachineBasicBlock *MBB = Instr->getParent();
  assert(MBB && "Can't get MachineBasicBlock here");
  MachineFunction *MF = MBB->getParent();
  assert(MF && "Can't get MachineFunction here");
  const TargetMachine *TM = &MF->getTarget();
  const TargetInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
  const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
  MachineRegisterInfo *MRI = &MF->getRegInfo();

  for (unsigned OpIdx = 0, EndIdx = Instr->getNumOperands(); OpIdx < EndIdx;
       ++OpIdx) {
    MachineOperand &MO = Instr->getOperand(OpIdx);
    const TargetRegisterClass *OpRegCstraints =
        Instr->getRegClassConstraint(OpIdx, TII, TRI);

    // If there's no constraint, there's nothing to do.
    if (!OpRegCstraints)
      continue;
    // If the operand is a frame index, there's nothing to do here.
    // A frame index operand will resolve correctly during PEI.
    if (MO.isFI())
      continue;

    assert(MO.isReg() &&
           "Operand has register constraints without being a register!");

    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (!OpRegCstraints->contains(Reg))
        return false;
    } else if (!OpRegCstraints->hasSubClassEq(MRI->getRegClass(Reg)) &&
               !MRI->constrainRegClass(Reg, OpRegCstraints))
      return false;
  }

  return true;
}

/// convertFlagSettingOpcode - return opcode that does not
/// set flags when possible. The caller is responsible to do
/// the actual substitution and legality checking.
static unsigned convertFlagSettingOpcode(MachineInstr *MI) {
    unsigned NewOpc;
    switch (MI->getOpcode()) {
    default:
      return false;
    case AArch64::ADDSWrr:      NewOpc = AArch64::ADDWrr; break;
    case AArch64::ADDSWri:      NewOpc = AArch64::ADDWri; break;
    case AArch64::ADDSWrs:      NewOpc = AArch64::ADDWrs; break;
    case AArch64::ADDSWrx:      NewOpc = AArch64::ADDWrx; break;
    case AArch64::ADDSXrr:      NewOpc = AArch64::ADDXrr; break;
    case AArch64::ADDSXri:      NewOpc = AArch64::ADDXri; break;
    case AArch64::ADDSXrs:      NewOpc = AArch64::ADDXrs; break;
    case AArch64::ADDSXrx:      NewOpc = AArch64::ADDXrx; break;
    case AArch64::SUBSWrr:      NewOpc = AArch64::SUBWrr; break;
    case AArch64::SUBSWri:      NewOpc = AArch64::SUBWri; break;
    case AArch64::SUBSWrs:      NewOpc = AArch64::SUBWrs; break;
    case AArch64::SUBSWrx:      NewOpc = AArch64::SUBWrx; break;
    case AArch64::SUBSXrr:      NewOpc = AArch64::SUBXrr; break;
    case AArch64::SUBSXri:      NewOpc = AArch64::SUBXri; break;
    case AArch64::SUBSXrs:      NewOpc = AArch64::SUBXrs; break;
    case AArch64::SUBSXrx:      NewOpc = AArch64::SUBXrx; break;
    }
    return NewOpc;
}

/// optimizeCompareInstr - Convert the instruction supplying the argument to the
/// comparison into one that sets the zero bit in the flags register.
bool AArch64InstrInfo::optimizeCompareInstr(
    MachineInstr *CmpInstr, unsigned SrcReg, unsigned SrcReg2, int CmpMask,
    int CmpValue, const MachineRegisterInfo *MRI) const {

  // Replace SUBSWrr with SUBWrr if NZCV is not used.
  int Cmp_NZCV = CmpInstr->findRegisterDefOperandIdx(AArch64::NZCV, true);
  if (Cmp_NZCV != -1) {
    unsigned Opc = CmpInstr->getOpcode();
    unsigned NewOpc = convertFlagSettingOpcode(CmpInstr);
    if (NewOpc == Opc)
      return false;
    const MCInstrDesc &MCID = get(NewOpc);
    CmpInstr->setDesc(MCID);
    CmpInstr->RemoveOperand(Cmp_NZCV);
    bool succeeded = UpdateOperandRegClass(CmpInstr);
    (void)succeeded;
    assert(succeeded && "Some operands reg class are incompatible!");
    return true;
  }

  // Continue only if we have a "ri" where immediate is zero.
  // FIXME:CmpValue has already been converted to 0 or 1 in analyzeCompare
  // function.
  assert((CmpValue == 0 || CmpValue == 1) && "CmpValue must be 0 or 1!");
  if (CmpValue != 0 || SrcReg2 != 0)
    return false;

  // CmpInstr is a Compare instruction if destination register is not used.
  if (!MRI->use_nodbg_empty(CmpInstr->getOperand(0).getReg()))
    return false;

  // Get the unique definition of SrcReg.
  MachineInstr *MI = MRI->getUniqueVRegDef(SrcReg);
  if (!MI)
    return false;

  // We iterate backward, starting from the instruction before CmpInstr and
  // stop when reaching the definition of the source register or done with the
  // basic block, to check whether NZCV is used or modified in between.
  MachineBasicBlock::iterator I = CmpInstr, E = MI,
                              B = CmpInstr->getParent()->begin();

  // Early exit if CmpInstr is at the beginning of the BB.
  if (I == B)
    return false;

  // Check whether the definition of SrcReg is in the same basic block as
  // Compare. If not, we can't optimize away the Compare.
  if (MI->getParent() != CmpInstr->getParent())
    return false;

  // Check that NZCV isn't set between the comparison instruction and the one we
  // want to change.
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  for (--I; I != E; --I) {
    const MachineInstr &Instr = *I;

    if (Instr.modifiesRegister(AArch64::NZCV, TRI) ||
        Instr.readsRegister(AArch64::NZCV, TRI))
      // This instruction modifies or uses NZCV after the one we want to
      // change. We can't do this transformation.
      return false;
    if (I == B)
      // The 'and' is below the comparison instruction.
      return false;
  }

  unsigned NewOpc = MI->getOpcode();
  switch (MI->getOpcode()) {
  default:
    return false;
  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSWri:
  case AArch64::SUBSXrr:
  case AArch64::SUBSXri:
    break;
  case AArch64::ADDWrr:    NewOpc = AArch64::ADDSWrr; break;
  case AArch64::ADDWri:    NewOpc = AArch64::ADDSWri; break;
  case AArch64::ADDXrr:    NewOpc = AArch64::ADDSXrr; break;
  case AArch64::ADDXri:    NewOpc = AArch64::ADDSXri; break;
  case AArch64::ADCWr:     NewOpc = AArch64::ADCSWr; break;
  case AArch64::ADCXr:     NewOpc = AArch64::ADCSXr; break;
  case AArch64::SUBWrr:    NewOpc = AArch64::SUBSWrr; break;
  case AArch64::SUBWri:    NewOpc = AArch64::SUBSWri; break;
  case AArch64::SUBXrr:    NewOpc = AArch64::SUBSXrr; break;
  case AArch64::SUBXri:    NewOpc = AArch64::SUBSXri; break;
  case AArch64::SBCWr:     NewOpc = AArch64::SBCSWr; break;
  case AArch64::SBCXr:     NewOpc = AArch64::SBCSXr; break;
  case AArch64::ANDWri:    NewOpc = AArch64::ANDSWri; break;
  case AArch64::ANDXri:    NewOpc = AArch64::ANDSXri; break;
  }

  // Scan forward for the use of NZCV.
  // When checking against MI: if it's a conditional code requires
  // checking of V bit, then this is not safe to do.
  // It is safe to remove CmpInstr if NZCV is redefined or killed.
  // If we are done with the basic block, we need to check whether NZCV is
  // live-out.
  bool IsSafe = false;
  for (MachineBasicBlock::iterator I = CmpInstr,
                                   E = CmpInstr->getParent()->end();
       !IsSafe && ++I != E;) {
    const MachineInstr &Instr = *I;
    for (unsigned IO = 0, EO = Instr.getNumOperands(); !IsSafe && IO != EO;
         ++IO) {
      const MachineOperand &MO = Instr.getOperand(IO);
      if (MO.isRegMask() && MO.clobbersPhysReg(AArch64::NZCV)) {
        IsSafe = true;
        break;
      }
      if (!MO.isReg() || MO.getReg() != AArch64::NZCV)
        continue;
      if (MO.isDef()) {
        IsSafe = true;
        break;
      }

      // Decode the condition code.
      unsigned Opc = Instr.getOpcode();
      AArch64CC::CondCode CC;
      switch (Opc) {
      default:
        return false;
      case AArch64::Bcc:
        CC = (AArch64CC::CondCode)Instr.getOperand(IO - 2).getImm();
        break;
      case AArch64::CSINVWr:
      case AArch64::CSINVXr:
      case AArch64::CSINCWr:
      case AArch64::CSINCXr:
      case AArch64::CSELWr:
      case AArch64::CSELXr:
      case AArch64::CSNEGWr:
      case AArch64::CSNEGXr:
      case AArch64::FCSELSrrr:
      case AArch64::FCSELDrrr:
        CC = (AArch64CC::CondCode)Instr.getOperand(IO - 1).getImm();
        break;
      }

      // It is not safe to remove Compare instruction if Overflow(V) is used.
      switch (CC) {
      default:
        // NZCV can be used multiple times, we should continue.
        break;
      case AArch64CC::VS:
      case AArch64CC::VC:
      case AArch64CC::GE:
      case AArch64CC::LT:
      case AArch64CC::GT:
      case AArch64CC::LE:
        return false;
      }
    }
  }

  // If NZCV is not killed nor re-defined, we should check whether it is
  // live-out. If it is live-out, do not optimize.
  if (!IsSafe) {
    MachineBasicBlock *ParentBlock = CmpInstr->getParent();
    for (auto *MBB : ParentBlock->successors())
      if (MBB->isLiveIn(AArch64::NZCV))
        return false;
  }

  // Update the instruction to set NZCV.
  MI->setDesc(get(NewOpc));
  CmpInstr->eraseFromParent();
  bool succeeded = UpdateOperandRegClass(MI);
  (void)succeeded;
  assert(succeeded && "Some operands reg class are incompatible!");
  MI->addRegisterDefined(AArch64::NZCV, TRI);
  return true;
}

bool
AArch64InstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  if (MI->getOpcode() != TargetOpcode::LOAD_STACK_GUARD)
    return false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned Reg = MI->getOperand(0).getReg();
  const GlobalValue *GV =
      cast<GlobalValue>((*MI->memoperands_begin())->getValue());
  const TargetMachine &TM = MBB.getParent()->getTarget();
  unsigned char OpFlags = Subtarget.ClassifyGlobalReference(GV, TM);
  const unsigned char MO_NC = AArch64II::MO_NC;

  if ((OpFlags & AArch64II::MO_GOT) != 0) {
    BuildMI(MBB, MI, DL, get(AArch64::LOADgot), Reg)
        .addGlobalAddress(GV, 0, AArch64II::MO_GOT);
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill).addImm(0)
        .addMemOperand(*MI->memoperands_begin());
  } else if (TM.getCodeModel() == CodeModel::Large) {
    BuildMI(MBB, MI, DL, get(AArch64::MOVZXi), Reg)
        .addGlobalAddress(GV, 0, AArch64II::MO_G3).addImm(48);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G2 | MO_NC).addImm(32);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G1 | MO_NC).addImm(16);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G0 | MO_NC).addImm(0);
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill).addImm(0)
        .addMemOperand(*MI->memoperands_begin());
  } else {
    BuildMI(MBB, MI, DL, get(AArch64::ADRP), Reg)
        .addGlobalAddress(GV, 0, OpFlags | AArch64II::MO_PAGE);
    unsigned char LoFlags = OpFlags | AArch64II::MO_PAGEOFF | MO_NC;
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, LoFlags)
        .addMemOperand(*MI->memoperands_begin());
  }

  MBB.erase(MI);

  return true;
}

/// Return true if this is this instruction has a non-zero immediate
bool AArch64InstrInfo::hasShiftedReg(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::ANDWrs:
  case AArch64::ANDXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
  case AArch64::BICWrs:
  case AArch64::BICXrs:
  case AArch64::CRC32Brr:
  case AArch64::CRC32CBrr:
  case AArch64::CRC32CHrr:
  case AArch64::CRC32CWrr:
  case AArch64::CRC32CXrr:
  case AArch64::CRC32Hrr:
  case AArch64::CRC32Wrr:
  case AArch64::CRC32Xrr:
  case AArch64::EONWrs:
  case AArch64::EONXrs:
  case AArch64::EORWrs:
  case AArch64::EORXrs:
  case AArch64::ORNWrs:
  case AArch64::ORNXrs:
  case AArch64::ORRWrs:
  case AArch64::ORRXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::SUBWrs:
  case AArch64::SUBXrs:
    if (MI->getOperand(3).isImm()) {
      unsigned val = MI->getOperand(3).getImm();
      return (val != 0);
    }
    break;
  }
  return false;
}

/// Return true if this is this instruction has a non-zero immediate
bool AArch64InstrInfo::hasExtendedReg(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrx:
  case AArch64::ADDSXrx64:
  case AArch64::ADDWrx:
  case AArch64::ADDXrx:
  case AArch64::ADDXrx64:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrx:
  case AArch64::SUBSXrx64:
  case AArch64::SUBWrx:
  case AArch64::SUBXrx:
  case AArch64::SUBXrx64:
    if (MI->getOperand(3).isImm()) {
      unsigned val = MI->getOperand(3).getImm();
      return (val != 0);
    }
    break;
  }

  return false;
}

// Return true if this instruction simply sets its single destination register
// to zero. This is equivalent to a register rename of the zero-register.
bool AArch64InstrInfo::isGPRZero(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::MOVZWi:
  case AArch64::MOVZXi: // movz Rd, #0 (LSL #0)
    if (MI->getOperand(1).isImm() && MI->getOperand(1).getImm() == 0) {
      assert(MI->getDesc().getNumOperands() == 3 &&
             MI->getOperand(2).getImm() == 0 && "invalid MOVZi operands");
      return true;
    }
    break;
  case AArch64::ANDWri: // and Rd, Rzr, #imm
    return MI->getOperand(1).getReg() == AArch64::WZR;
  case AArch64::ANDXri:
    return MI->getOperand(1).getReg() == AArch64::XZR;
  case TargetOpcode::COPY:
    return MI->getOperand(1).getReg() == AArch64::WZR;
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool AArch64InstrInfo::isGPRCopy(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // GPR32 copies will by lowered to ORRXrs
    unsigned DstReg = MI->getOperand(0).getReg();
    return (AArch64::GPR32RegClass.contains(DstReg) ||
            AArch64::GPR64RegClass.contains(DstReg));
  }
  case AArch64::ORRXrs: // orr Xd, Xzr, Xm (LSL #0)
    if (MI->getOperand(1).getReg() == AArch64::XZR) {
      assert(MI->getDesc().getNumOperands() == 4 &&
             MI->getOperand(3).getImm() == 0 && "invalid ORRrs operands");
      return true;
    }
    break;
  case AArch64::ADDXri: // add Xd, Xn, #0 (LSL #0)
    if (MI->getOperand(2).getImm() == 0) {
      assert(MI->getDesc().getNumOperands() == 4 &&
             MI->getOperand(3).getImm() == 0 && "invalid ADDXri operands");
      return true;
    }
    break;
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool AArch64InstrInfo::isFPRCopy(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // FPR64 copies will by lowered to ORR.16b
    unsigned DstReg = MI->getOperand(0).getReg();
    return (AArch64::FPR64RegClass.contains(DstReg) ||
            AArch64::FPR128RegClass.contains(DstReg));
  }
  case AArch64::ORRv16i8:
    if (MI->getOperand(1).getReg() == MI->getOperand(2).getReg()) {
      assert(MI->getDesc().getNumOperands() == 3 && MI->getOperand(0).isReg() &&
             "invalid ORRv16i8 operands");
      return true;
    }
    break;
  }
  return false;
}

unsigned AArch64InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                               int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::LDRWui:
  case AArch64::LDRXui:
  case AArch64::LDRBui:
  case AArch64::LDRHui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
    if (MI->getOperand(0).getSubReg() == 0 && MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() && MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

unsigned AArch64InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                              int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::STRWui:
  case AArch64::STRXui:
  case AArch64::STRBui:
  case AArch64::STRHui:
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:
    if (MI->getOperand(0).getSubReg() == 0 && MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() && MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

/// Return true if this is load/store scales or extends its register offset.
/// This refers to scaling a dynamic index as opposed to scaled immediates.
/// MI should be a memory op that allows scaled addressing.
bool AArch64InstrInfo::isScaledAddr(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::LDRBBroW:
  case AArch64::LDRBroW:
  case AArch64::LDRDroW:
  case AArch64::LDRHHroW:
  case AArch64::LDRHroW:
  case AArch64::LDRQroW:
  case AArch64::LDRSBWroW:
  case AArch64::LDRSBXroW:
  case AArch64::LDRSHWroW:
  case AArch64::LDRSHXroW:
  case AArch64::LDRSWroW:
  case AArch64::LDRSroW:
  case AArch64::LDRWroW:
  case AArch64::LDRXroW:
  case AArch64::STRBBroW:
  case AArch64::STRBroW:
  case AArch64::STRDroW:
  case AArch64::STRHHroW:
  case AArch64::STRHroW:
  case AArch64::STRQroW:
  case AArch64::STRSroW:
  case AArch64::STRWroW:
  case AArch64::STRXroW:
  case AArch64::LDRBBroX:
  case AArch64::LDRBroX:
  case AArch64::LDRDroX:
  case AArch64::LDRHHroX:
  case AArch64::LDRHroX:
  case AArch64::LDRQroX:
  case AArch64::LDRSBWroX:
  case AArch64::LDRSBXroX:
  case AArch64::LDRSHWroX:
  case AArch64::LDRSHXroX:
  case AArch64::LDRSWroX:
  case AArch64::LDRSroX:
  case AArch64::LDRWroX:
  case AArch64::LDRXroX:
  case AArch64::STRBBroX:
  case AArch64::STRBroX:
  case AArch64::STRDroX:
  case AArch64::STRHHroX:
  case AArch64::STRHroX:
  case AArch64::STRQroX:
  case AArch64::STRSroX:
  case AArch64::STRWroX:
  case AArch64::STRXroX:

    unsigned Val = MI->getOperand(3).getImm();
    AArch64_AM::ShiftExtendType ExtType = AArch64_AM::getMemExtendType(Val);
    return (ExtType != AArch64_AM::UXTX) || AArch64_AM::getMemDoShift(Val);
  }
  return false;
}

/// Check all MachineMemOperands for a hint to suppress pairing.
bool AArch64InstrInfo::isLdStPairSuppressed(const MachineInstr *MI) const {
  assert(MOSuppressPair < (1 << MachineMemOperand::MOTargetNumBits) &&
         "Too many target MO flags");
  for (auto *MM : MI->memoperands()) {
    if (MM->getFlags() &
        (MOSuppressPair << MachineMemOperand::MOTargetStartBit)) {
      return true;
    }
  }
  return false;
}

/// Set a flag on the first MachineMemOperand to suppress pairing.
void AArch64InstrInfo::suppressLdStPair(MachineInstr *MI) const {
  if (MI->memoperands_empty())
    return;

  assert(MOSuppressPair < (1 << MachineMemOperand::MOTargetNumBits) &&
         "Too many target MO flags");
  (*MI->memoperands_begin())
      ->setFlags(MOSuppressPair << MachineMemOperand::MOTargetStartBit);
}

bool
AArch64InstrInfo::getLdStBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                                       unsigned &Offset,
                                       const TargetRegisterInfo *TRI) const {
  switch (LdSt->getOpcode()) {
  default:
    return false;
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:
  case AArch64::STRXui:
  case AArch64::STRWui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
  case AArch64::LDRXui:
  case AArch64::LDRWui:
    if (!LdSt->getOperand(1).isReg() || !LdSt->getOperand(2).isImm())
      return false;
    BaseReg = LdSt->getOperand(1).getReg();
    MachineFunction &MF = *LdSt->getParent()->getParent();
    unsigned Width = getRegClass(LdSt->getDesc(), 0, TRI, MF)->getSize();
    Offset = LdSt->getOperand(2).getImm() * Width;
    return true;
  };
}

/// Detect opportunities for ldp/stp formation.
///
/// Only called for LdSt for which getLdStBaseRegImmOfs returns true.
bool AArch64InstrInfo::shouldClusterLoads(MachineInstr *FirstLdSt,
                                          MachineInstr *SecondLdSt,
                                          unsigned NumLoads) const {
  // Only cluster up to a single pair.
  if (NumLoads > 1)
    return false;
  if (FirstLdSt->getOpcode() != SecondLdSt->getOpcode())
    return false;
  // getLdStBaseRegImmOfs guarantees that oper 2 isImm.
  unsigned Ofs1 = FirstLdSt->getOperand(2).getImm();
  // Allow 6 bits of positive range.
  if (Ofs1 > 64)
    return false;
  // The caller should already have ordered First/SecondLdSt by offset.
  unsigned Ofs2 = SecondLdSt->getOperand(2).getImm();
  return Ofs1 + 1 == Ofs2;
}

bool AArch64InstrInfo::shouldScheduleAdjacent(MachineInstr *First,
                                              MachineInstr *Second) const {
  // Cyclone can fuse CMN, CMP followed by Bcc.

  // FIXME: B0 can also fuse:
  // AND, BIC, ORN, ORR, or EOR (optional S) followed by Bcc or CBZ or CBNZ.
  if (Second->getOpcode() != AArch64::Bcc)
    return false;
  switch (First->getOpcode()) {
  default:
    return false;
  case AArch64::SUBSWri:
  case AArch64::ADDSWri:
  case AArch64::ANDSWri:
  case AArch64::SUBSXri:
  case AArch64::ADDSXri:
  case AArch64::ANDSXri:
    return true;
  }
}

MachineInstr *AArch64InstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                                         int FrameIx,
                                                         uint64_t Offset,
                                                         const MDNode *MDPtr,
                                                         DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(AArch64::DBG_VALUE))
                                .addFrameIndex(FrameIx)
                                .addImm(0)
                                .addImm(Offset)
                                .addMetadata(MDPtr);
  return &*MIB;
}

static const MachineInstrBuilder &AddSubReg(const MachineInstrBuilder &MIB,
                                            unsigned Reg, unsigned SubIdx,
                                            unsigned State,
                                            const TargetRegisterInfo *TRI) {
  if (!SubIdx)
    return MIB.addReg(Reg, State);

  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return MIB.addReg(TRI->getSubReg(Reg, SubIdx), State);
  return MIB.addReg(Reg, State, SubIdx);
}

static bool forwardCopyWillClobberTuple(unsigned DestReg, unsigned SrcReg,
                                        unsigned NumRegs) {
  // We really want the positive remainder mod 32 here, that happens to be
  // easily obtainable with a mask.
  return ((DestReg - SrcReg) & 0x1f) < NumRegs;
}

void AArch64InstrInfo::copyPhysRegTuple(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, DebugLoc DL,
    unsigned DestReg, unsigned SrcReg, bool KillSrc, unsigned Opcode,
    llvm::ArrayRef<unsigned> Indices) const {
  assert(Subtarget.hasNEON() &&
         "Unexpected register copy without NEON");
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  uint16_t DestEncoding = TRI->getEncodingValue(DestReg);
  uint16_t SrcEncoding = TRI->getEncodingValue(SrcReg);
  unsigned NumRegs = Indices.size();

  int SubReg = 0, End = NumRegs, Incr = 1;
  if (forwardCopyWillClobberTuple(DestEncoding, SrcEncoding, NumRegs)) {
    SubReg = NumRegs - 1;
    End = -1;
    Incr = -1;
  }

  for (; SubReg != End; SubReg += Incr) {
    const MachineInstrBuilder &MIB = BuildMI(MBB, I, DL, get(Opcode));
    AddSubReg(MIB, DestReg, Indices[SubReg], RegState::Define, TRI);
    AddSubReg(MIB, SrcReg, Indices[SubReg], 0, TRI);
    AddSubReg(MIB, SrcReg, Indices[SubReg], getKillRegState(KillSrc), TRI);
  }
}

void AArch64InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I, DebugLoc DL,
                                   unsigned DestReg, unsigned SrcReg,
                                   bool KillSrc) const {
  if (AArch64::GPR32spRegClass.contains(DestReg) &&
      (AArch64::GPR32spRegClass.contains(SrcReg) || SrcReg == AArch64::WZR)) {
    const TargetRegisterInfo *TRI = &getRegisterInfo();

    if (DestReg == AArch64::WSP || SrcReg == AArch64::WSP) {
      // If either operand is WSP, expand to ADD #0.
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ADD Xd, Xn, #0" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, AArch64::sub_32,
                                                     &AArch64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, AArch64::sub_32,
                                                    &AArch64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(AArch64::ADDXri), DestRegX)
            .addReg(SrcRegX, RegState::Undef)
            .addImm(0)
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        BuildMI(MBB, I, DL, get(AArch64::ADDWri), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc))
            .addImm(0)
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
      }
    } else if (SrcReg == AArch64::WZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(AArch64::MOVZWi), DestReg).addImm(0).addImm(
          AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else {
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ORR Xd, XZR, Xm" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, AArch64::sub_32,
                                                     &AArch64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, AArch64::sub_32,
                                                    &AArch64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(AArch64::ORRXrr), DestRegX)
            .addReg(AArch64::XZR)
            .addReg(SrcRegX, RegState::Undef)
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        // Otherwise, expand to ORR WZR.
        BuildMI(MBB, I, DL, get(AArch64::ORRWrr), DestReg)
            .addReg(AArch64::WZR)
            .addReg(SrcReg, getKillRegState(KillSrc));
      }
    }
    return;
  }

  if (AArch64::GPR64spRegClass.contains(DestReg) &&
      (AArch64::GPR64spRegClass.contains(SrcReg) || SrcReg == AArch64::XZR)) {
    if (DestReg == AArch64::SP || SrcReg == AArch64::SP) {
      // If either operand is SP, expand to ADD #0.
      BuildMI(MBB, I, DL, get(AArch64::ADDXri), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0)
          .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else if (SrcReg == AArch64::XZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(AArch64::MOVZXi), DestReg).addImm(0).addImm(
          AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else {
      // Otherwise, expand to ORR XZR.
      BuildMI(MBB, I, DL, get(AArch64::ORRXrr), DestReg)
          .addReg(AArch64::XZR)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  // Copy a DDDD register quad by copying the individual sub-registers.
  if (AArch64::DDDDRegClass.contains(DestReg) &&
      AArch64::DDDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::dsub0, AArch64::dsub1,
                                        AArch64::dsub2, AArch64::dsub3 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DDD register triple by copying the individual sub-registers.
  if (AArch64::DDDRegClass.contains(DestReg) &&
      AArch64::DDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::dsub0, AArch64::dsub1,
                                        AArch64::dsub2 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DD register pair by copying the individual sub-registers.
  if (AArch64::DDRegClass.contains(DestReg) &&
      AArch64::DDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::dsub0, AArch64::dsub1 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a QQQQ register quad by copying the individual sub-registers.
  if (AArch64::QQQQRegClass.contains(DestReg) &&
      AArch64::QQQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::qsub0, AArch64::qsub1,
                                        AArch64::qsub2, AArch64::qsub3 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQQ register triple by copying the individual sub-registers.
  if (AArch64::QQQRegClass.contains(DestReg) &&
      AArch64::QQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::qsub0, AArch64::qsub1,
                                        AArch64::qsub2 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQ register pair by copying the individual sub-registers.
  if (AArch64::QQRegClass.contains(DestReg) &&
      AArch64::QQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { AArch64::qsub0, AArch64::qsub1 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  if (AArch64::FPR128RegClass.contains(DestReg) &&
      AArch64::FPR128RegClass.contains(SrcReg)) {
    if(Subtarget.hasNEON()) {
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::STRQpre))
        .addReg(AArch64::SP, RegState::Define)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(AArch64::SP)
        .addImm(-16);
      BuildMI(MBB, I, DL, get(AArch64::LDRQpre))
        .addReg(AArch64::SP, RegState::Define)
        .addReg(DestReg, RegState::Define)
        .addReg(AArch64::SP)
        .addImm(16);
    }
    return;
  }

  if (AArch64::FPR64RegClass.contains(DestReg) &&
      AArch64::FPR64RegClass.contains(SrcReg)) {
    if(Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::dsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::dsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::FMOVDr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR32RegClass.contains(DestReg) &&
      AArch64::FPR32RegClass.contains(SrcReg)) {
    if(Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::ssub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::ssub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR16RegClass.contains(DestReg) &&
      AArch64::FPR16RegClass.contains(SrcReg)) {
    if(Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::hsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::hsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::hsub,
                                       &AArch64::FPR32RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::hsub,
                                      &AArch64::FPR32RegClass);
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR8RegClass.contains(DestReg) &&
      AArch64::FPR8RegClass.contains(SrcReg)) {
    if(Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::bsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::bsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::bsub,
                                       &AArch64::FPR32RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::bsub,
                                      &AArch64::FPR32RegClass);
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  // Copies between GPR64 and FPR64.
  if (AArch64::FPR64RegClass.contains(DestReg) &&
      AArch64::GPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVXDr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (AArch64::GPR64RegClass.contains(DestReg) &&
      AArch64::FPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVDXr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  // Copies between GPR32 and FPR32.
  if (AArch64::FPR32RegClass.contains(DestReg) &&
      AArch64::GPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVWSr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (AArch64::GPR32RegClass.contains(DestReg) &&
      AArch64::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVSWr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (DestReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(SrcReg) && "Invalid NZCV copy");
    BuildMI(MBB, I, DL, get(AArch64::MSR))
      .addImm(AArch64SysReg::NZCV)
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addReg(AArch64::NZCV, RegState::Implicit | RegState::Define);
    return;
  }

  if (SrcReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(DestReg) && "Invalid NZCV copy");
    BuildMI(MBB, I, DL, get(AArch64::MRS))
      .addReg(DestReg)
      .addImm(AArch64SysReg::NZCV)
      .addReg(AArch64::NZCV, RegState::Implicit | getKillRegState(KillSrc));
    return;
  }

  llvm_unreachable("unimplemented reg-to-reg copy");
}

void AArch64InstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned SrcReg,
    bool isKill, int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachinePointerInfo PtrInfo(PseudoSourceValue::getFixedStack(FI));
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOStore, MFI.getObjectSize(FI), Align);
  unsigned Opc = 0;
  bool Offset = true;
  switch (RC->getSize()) {
  case 1:
    if (AArch64::FPR8RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRBui;
    break;
  case 2:
    if (AArch64::FPR16RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRHui;
    break;
  case 4:
    if (AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::STRWui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR32RegClass);
      else
        assert(SrcReg != AArch64::WSP);
    } else if (AArch64::FPR32RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRSui;
    break;
  case 8:
    if (AArch64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::STRXui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR64RegClass);
      else
        assert(SrcReg != AArch64::SP);
    } else if (AArch64::FPR64RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRDui;
    break;
  case 16:
    if (AArch64::FPR128RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRQui;
    else if (AArch64::DDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Twov1d, Offset = false;
    }
    break;
  case 24:
    if (AArch64::DDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Threev1d, Offset = false;
    }
    break;
  case 32:
    if (AArch64::DDDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Fourv1d, Offset = false;
    } else if (AArch64::QQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Twov2d, Offset = false;
    }
    break;
  case 48:
    if (AArch64::QQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Threev2d, Offset = false;
    }
    break;
  case 64:
    if (AArch64::QQQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register store without NEON");
      Opc = AArch64::ST1Fourv2d, Offset = false;
    }
    break;
  }
  assert(Opc && "Unknown register class");

  const MachineInstrBuilder &MI = BuildMI(MBB, MBBI, DL, get(Opc))
                                      .addReg(SrcReg, getKillRegState(isKill))
                                      .addFrameIndex(FI);

  if (Offset)
    MI.addImm(0);
  MI.addMemOperand(MMO);
}

void AArch64InstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned DestReg,
    int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);
  MachinePointerInfo PtrInfo(PseudoSourceValue::getFixedStack(FI));
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, MFI.getObjectSize(FI), Align);

  unsigned Opc = 0;
  bool Offset = true;
  switch (RC->getSize()) {
  case 1:
    if (AArch64::FPR8RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRBui;
    break;
  case 2:
    if (AArch64::FPR16RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRHui;
    break;
  case 4:
    if (AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::LDRWui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &AArch64::GPR32RegClass);
      else
        assert(DestReg != AArch64::WSP);
    } else if (AArch64::FPR32RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRSui;
    break;
  case 8:
    if (AArch64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::LDRXui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &AArch64::GPR64RegClass);
      else
        assert(DestReg != AArch64::SP);
    } else if (AArch64::FPR64RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRDui;
    break;
  case 16:
    if (AArch64::FPR128RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRQui;
    else if (AArch64::DDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Twov1d, Offset = false;
    }
    break;
  case 24:
    if (AArch64::DDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Threev1d, Offset = false;
    }
    break;
  case 32:
    if (AArch64::DDDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Fourv1d, Offset = false;
    } else if (AArch64::QQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Twov2d, Offset = false;
    }
    break;
  case 48:
    if (AArch64::QQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Threev2d, Offset = false;
    }
    break;
  case 64:
    if (AArch64::QQQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() &&
             "Unexpected register load without NEON");
      Opc = AArch64::LD1Fourv2d, Offset = false;
    }
    break;
  }
  assert(Opc && "Unknown register class");

  const MachineInstrBuilder &MI = BuildMI(MBB, MBBI, DL, get(Opc))
                                      .addReg(DestReg, getDefRegState(true))
                                      .addFrameIndex(FI);
  if (Offset)
    MI.addImm(0);
  MI.addMemOperand(MMO);
}

void llvm::emitFrameOffset(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg, int Offset,
                           const TargetInstrInfo *TII,
                           MachineInstr::MIFlag Flag, bool SetNZCV) {
  if (DestReg == SrcReg && Offset == 0)
    return;

  bool isSub = Offset < 0;
  if (isSub)
    Offset = -Offset;

  // FIXME: If the offset won't fit in 24-bits, compute the offset into a
  // scratch register.  If DestReg is a virtual register, use it as the
  // scratch register; otherwise, create a new virtual register (to be
  // replaced by the scavenger at the end of PEI).  That case can be optimized
  // slightly if DestReg is SP which is always 16-byte aligned, so the scratch
  // register can be loaded with offset%8 and the add/sub can use an extending
  // instruction with LSL#3.
  // Currently the function handles any offsets but generates a poor sequence
  // of code.
  //  assert(Offset < (1 << 24) && "unimplemented reg plus immediate");

  unsigned Opc;
  if (SetNZCV)
    Opc = isSub ? AArch64::SUBSXri : AArch64::ADDSXri;
  else
    Opc = isSub ? AArch64::SUBXri : AArch64::ADDXri;
  const unsigned MaxEncoding = 0xfff;
  const unsigned ShiftSize = 12;
  const unsigned MaxEncodableValue = MaxEncoding << ShiftSize;
  while (((unsigned)Offset) >= (1 << ShiftSize)) {
    unsigned ThisVal;
    if (((unsigned)Offset) > MaxEncodableValue) {
      ThisVal = MaxEncodableValue;
    } else {
      ThisVal = Offset & MaxEncodableValue;
    }
    assert((ThisVal >> ShiftSize) <= MaxEncoding &&
           "Encoding cannot handle value that big");
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
        .addReg(SrcReg)
        .addImm(ThisVal >> ShiftSize)
        .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, ShiftSize))
        .setMIFlag(Flag);

    SrcReg = DestReg;
    Offset -= ThisVal;
    if (Offset == 0)
      return;
  }
  BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
      .addReg(SrcReg)
      .addImm(Offset)
      .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
      .setMIFlag(Flag);
}

MachineInstr *
AArch64InstrInfo::foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                                        const SmallVectorImpl<unsigned> &Ops,
                                        int FrameIndex) const {
  // This is a bit of a hack. Consider this instruction:
  //
  //   %vreg0<def> = COPY %SP; GPR64all:%vreg0
  //
  // We explicitly chose GPR64all for the virtual register so such a copy might
  // be eliminated by RegisterCoalescer. However, that may not be possible, and
  // %vreg0 may even spill. We can't spill %SP, and since it is in the GPR64all
  // register class, TargetInstrInfo::foldMemoryOperand() is going to try.
  //
  // To prevent that, we are going to constrain the %vreg0 register class here.
  //
  // <rdar://problem/11522048>
  //
  if (MI->isCopy()) {
    unsigned DstReg = MI->getOperand(0).getReg();
    unsigned SrcReg = MI->getOperand(1).getReg();
    if (SrcReg == AArch64::SP &&
        TargetRegisterInfo::isVirtualRegister(DstReg)) {
      MF.getRegInfo().constrainRegClass(DstReg, &AArch64::GPR64RegClass);
      return nullptr;
    }
    if (DstReg == AArch64::SP &&
        TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR64RegClass);
      return nullptr;
    }
  }

  // Cannot fold.
  return nullptr;
}

int llvm::isAArch64FrameOffsetLegal(const MachineInstr &MI, int &Offset,
                                    bool *OutUseUnscaledOp,
                                    unsigned *OutUnscaledOp,
                                    int *EmittableOffset) {
  int Scale = 1;
  bool IsSigned = false;
  // The ImmIdx should be changed case by case if it is not 2.
  unsigned ImmIdx = 2;
  unsigned UnscaledOp = 0;
  // Set output values in case of early exit.
  if (EmittableOffset)
    *EmittableOffset = 0;
  if (OutUseUnscaledOp)
    *OutUseUnscaledOp = false;
  if (OutUnscaledOp)
    *OutUnscaledOp = 0;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("unhandled opcode in rewriteAArch64FrameIndex");
  // Vector spills/fills can't take an immediate offset.
  case AArch64::LD1Twov2d:
  case AArch64::LD1Threev2d:
  case AArch64::LD1Fourv2d:
  case AArch64::LD1Twov1d:
  case AArch64::LD1Threev1d:
  case AArch64::LD1Fourv1d:
  case AArch64::ST1Twov2d:
  case AArch64::ST1Threev2d:
  case AArch64::ST1Fourv2d:
  case AArch64::ST1Twov1d:
  case AArch64::ST1Threev1d:
  case AArch64::ST1Fourv1d:
    return AArch64FrameOffsetCannotUpdate;
  case AArch64::PRFMui:
    Scale = 8;
    UnscaledOp = AArch64::PRFUMi;
    break;
  case AArch64::LDRXui:
    Scale = 8;
    UnscaledOp = AArch64::LDURXi;
    break;
  case AArch64::LDRWui:
    Scale = 4;
    UnscaledOp = AArch64::LDURWi;
    break;
  case AArch64::LDRBui:
    Scale = 1;
    UnscaledOp = AArch64::LDURBi;
    break;
  case AArch64::LDRHui:
    Scale = 2;
    UnscaledOp = AArch64::LDURHi;
    break;
  case AArch64::LDRSui:
    Scale = 4;
    UnscaledOp = AArch64::LDURSi;
    break;
  case AArch64::LDRDui:
    Scale = 8;
    UnscaledOp = AArch64::LDURDi;
    break;
  case AArch64::LDRQui:
    Scale = 16;
    UnscaledOp = AArch64::LDURQi;
    break;
  case AArch64::LDRBBui:
    Scale = 1;
    UnscaledOp = AArch64::LDURBBi;
    break;
  case AArch64::LDRHHui:
    Scale = 2;
    UnscaledOp = AArch64::LDURHHi;
    break;
  case AArch64::LDRSBXui:
    Scale = 1;
    UnscaledOp = AArch64::LDURSBXi;
    break;
  case AArch64::LDRSBWui:
    Scale = 1;
    UnscaledOp = AArch64::LDURSBWi;
    break;
  case AArch64::LDRSHXui:
    Scale = 2;
    UnscaledOp = AArch64::LDURSHXi;
    break;
  case AArch64::LDRSHWui:
    Scale = 2;
    UnscaledOp = AArch64::LDURSHWi;
    break;
  case AArch64::LDRSWui:
    Scale = 4;
    UnscaledOp = AArch64::LDURSWi;
    break;

  case AArch64::STRXui:
    Scale = 8;
    UnscaledOp = AArch64::STURXi;
    break;
  case AArch64::STRWui:
    Scale = 4;
    UnscaledOp = AArch64::STURWi;
    break;
  case AArch64::STRBui:
    Scale = 1;
    UnscaledOp = AArch64::STURBi;
    break;
  case AArch64::STRHui:
    Scale = 2;
    UnscaledOp = AArch64::STURHi;
    break;
  case AArch64::STRSui:
    Scale = 4;
    UnscaledOp = AArch64::STURSi;
    break;
  case AArch64::STRDui:
    Scale = 8;
    UnscaledOp = AArch64::STURDi;
    break;
  case AArch64::STRQui:
    Scale = 16;
    UnscaledOp = AArch64::STURQi;
    break;
  case AArch64::STRBBui:
    Scale = 1;
    UnscaledOp = AArch64::STURBBi;
    break;
  case AArch64::STRHHui:
    Scale = 2;
    UnscaledOp = AArch64::STURHHi;
    break;

  case AArch64::LDPXi:
  case AArch64::LDPDi:
  case AArch64::STPXi:
  case AArch64::STPDi:
    IsSigned = true;
    Scale = 8;
    break;
  case AArch64::LDPQi:
  case AArch64::STPQi:
    IsSigned = true;
    Scale = 16;
    break;
  case AArch64::LDPWi:
  case AArch64::LDPSi:
  case AArch64::STPWi:
  case AArch64::STPSi:
    IsSigned = true;
    Scale = 4;
    break;

  case AArch64::LDURXi:
  case AArch64::LDURWi:
  case AArch64::LDURBi:
  case AArch64::LDURHi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURHHi:
  case AArch64::LDURBBi:
  case AArch64::LDURSBXi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSHWi:
  case AArch64::LDURSWi:
  case AArch64::STURXi:
  case AArch64::STURWi:
  case AArch64::STURBi:
  case AArch64::STURHi:
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURBBi:
  case AArch64::STURHHi:
    Scale = 1;
    break;
  }

  Offset += MI.getOperand(ImmIdx).getImm() * Scale;

  bool useUnscaledOp = false;
  // If the offset doesn't match the scale, we rewrite the instruction to
  // use the unscaled instruction instead. Likewise, if we have a negative
  // offset (and have an unscaled op to use).
  if ((Offset & (Scale - 1)) != 0 || (Offset < 0 && UnscaledOp != 0))
    useUnscaledOp = true;

  // Use an unscaled addressing mode if the instruction has a negative offset
  // (or if the instruction is already using an unscaled addressing mode).
  unsigned MaskBits;
  if (IsSigned) {
    // ldp/stp instructions.
    MaskBits = 7;
    Offset /= Scale;
  } else if (UnscaledOp == 0 || useUnscaledOp) {
    MaskBits = 9;
    IsSigned = true;
    Scale = 1;
  } else {
    MaskBits = 12;
    IsSigned = false;
    Offset /= Scale;
  }

  // Attempt to fold address computation.
  int MaxOff = (1 << (MaskBits - IsSigned)) - 1;
  int MinOff = (IsSigned ? (-MaxOff - 1) : 0);
  if (Offset >= MinOff && Offset <= MaxOff) {
    if (EmittableOffset)
      *EmittableOffset = Offset;
    Offset = 0;
  } else {
    int NewOff = Offset < 0 ? MinOff : MaxOff;
    if (EmittableOffset)
      *EmittableOffset = NewOff;
    Offset = (Offset - NewOff) * Scale;
  }
  if (OutUseUnscaledOp)
    *OutUseUnscaledOp = useUnscaledOp;
  if (OutUnscaledOp)
    *OutUnscaledOp = UnscaledOp;
  return AArch64FrameOffsetCanUpdate |
         (Offset == 0 ? AArch64FrameOffsetIsLegal : 0);
}

bool llvm::rewriteAArch64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                    unsigned FrameReg, int &Offset,
                                    const AArch64InstrInfo *TII) {
  unsigned Opcode = MI.getOpcode();
  unsigned ImmIdx = FrameRegIdx + 1;

  if (Opcode == AArch64::ADDSXri || Opcode == AArch64::ADDXri) {
    Offset += MI.getOperand(ImmIdx).getImm();
    emitFrameOffset(*MI.getParent(), MI, MI.getDebugLoc(),
                    MI.getOperand(0).getReg(), FrameReg, Offset, TII,
                    MachineInstr::NoFlags, (Opcode == AArch64::ADDSXri));
    MI.eraseFromParent();
    Offset = 0;
    return true;
  }

  int NewOffset;
  unsigned UnscaledOp;
  bool UseUnscaledOp;
  int Status = isAArch64FrameOffsetLegal(MI, Offset, &UseUnscaledOp,
                                         &UnscaledOp, &NewOffset);
  if (Status & AArch64FrameOffsetCanUpdate) {
    if (Status & AArch64FrameOffsetIsLegal)
      // Replace the FrameIndex with FrameReg.
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
    if (UseUnscaledOp)
      MI.setDesc(TII->get(UnscaledOp));

    MI.getOperand(ImmIdx).ChangeToImmediate(NewOffset);
    return Offset == 0;
  }

  return false;
}

void AArch64InstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(AArch64::HINT);
  NopInst.addOperand(MCOperand::CreateImm(0));
}
/// useMachineCombiner - return true when a target supports MachineCombiner
bool AArch64InstrInfo::useMachineCombiner(void) const {
  // AArch64 supports the combiner
  return true;
}
//
// True when Opc sets flag
static bool isCombineInstrSettingFlag(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSXrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBSWri:
  case AArch64::SUBSXri:
    return true;
  default:
    break;
  }
  return false;
}
//
// 32b Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate32(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDWrr:
  case AArch64::ADDWri:
  case AArch64::SUBWrr:
  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::SUBSWrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBWri:
  case AArch64::SUBSWri:
    return true;
  default:
    break;
  }
  return false;
}
//
// 64b Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate64(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDXrr:
  case AArch64::ADDXri:
  case AArch64::SUBXrr:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSXrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBXri:
  case AArch64::SUBSXri:
    return true;
  default:
    break;
  }
  return false;
}
//
// Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate(unsigned Opc) {
  return (isCombineInstrCandidate32(Opc) || isCombineInstrCandidate64(Opc));
}

static bool canCombineWithMUL(MachineBasicBlock &MBB, MachineOperand &MO,
                              unsigned MulOpc, unsigned ZeroReg) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineInstr *MI = nullptr;
  // We need a virtual register definition.
  if (MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    MI = MRI.getUniqueVRegDef(MO.getReg());
  // And it needs to be in the trace (otherwise, it won't have a depth).
  if (!MI || MI->getParent() != &MBB || (unsigned)MI->getOpcode() != MulOpc)
    return false;

  assert(MI->getNumOperands() >= 4 && MI->getOperand(0).isReg() &&
         MI->getOperand(1).isReg() && MI->getOperand(2).isReg() &&
         MI->getOperand(3).isReg() && "MAdd/MSub must have a least 4 regs");

  // The third input reg must be zero.
  if (MI->getOperand(3).getReg() != ZeroReg)
    return false;

  // Must only used by the user we combine with.
  if (!MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
    return false;

  return true;
}

/// hasPattern - return true when there is potentially a faster code sequence
/// for an instruction chain ending in \p Root. All potential patterns are
/// listed
/// in the \p Pattern vector. Pattern should be sorted in priority order since
/// the pattern evaluator stops checking as soon as it finds a faster sequence.

bool AArch64InstrInfo::hasPattern(
    MachineInstr &Root,
    SmallVectorImpl<MachineCombinerPattern::MC_PATTERN> &Pattern) const {
  unsigned Opc = Root.getOpcode();
  MachineBasicBlock &MBB = *Root.getParent();
  bool Found = false;

  if (!isCombineInstrCandidate(Opc))
    return 0;
  if (isCombineInstrSettingFlag(Opc)) {
    int Cmp_NZCV = Root.findRegisterDefOperandIdx(AArch64::NZCV, true);
    // When NZCV is live bail out.
    if (Cmp_NZCV == -1)
      return 0;
    unsigned NewOpc = convertFlagSettingOpcode(&Root);
    // When opcode can't change bail out.
    // CHECKME: do we miss any cases for opcode conversion?
    if (NewOpc == Opc)
      return 0;
    Opc = NewOpc;
  }

  switch (Opc) {
  default:
    break;
  case AArch64::ADDWrr:
    assert(Root.getOperand(1).isReg() && Root.getOperand(2).isReg() &&
           "ADDWrr does not have register operands");
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDW_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDW_OP2);
      Found = true;
    }
    break;
  case AArch64::ADDXrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDX_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDX_OP2);
      Found = true;
    }
    break;
  case AArch64::SUBWrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBW_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBW_OP2);
      Found = true;
    }
    break;
  case AArch64::SUBXrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBX_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBX_OP2);
      Found = true;
    }
    break;
  case AArch64::ADDWri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDWI_OP1);
      Found = true;
    }
    break;
  case AArch64::ADDXri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULADDXI_OP1);
      Found = true;
    }
    break;
  case AArch64::SUBWri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBWI_OP1);
      Found = true;
    }
    break;
  case AArch64::SUBXri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Pattern.push_back(MachineCombinerPattern::MC_MULSUBXI_OP1);
      Found = true;
    }
    break;
  }
  return Found;
}

/// genMadd - Generate madd instruction and combine mul and add.
/// Example:
///  MUL I=A,B,0
///  ADD R,I,C
///  ==> MADD R,A,B,C
/// \param Root is the ADD instruction
/// \param [out] InsInstrs is a vector of machine instructions and will
/// contain the generated madd instruction
/// \param IdxMulOpd is index of operand in Root that is the result of
/// the MUL. In the example above IdxMulOpd is 1.
/// \param MaddOpc the opcode fo the madd instruction
static MachineInstr *genMadd(MachineFunction &MF, MachineRegisterInfo &MRI,
                             const TargetInstrInfo *TII, MachineInstr &Root,
                             SmallVectorImpl<MachineInstr *> &InsInstrs,
                             unsigned IdxMulOpd, unsigned MaddOpc) {
  assert(IdxMulOpd == 1 || IdxMulOpd == 2);

  unsigned IdxOtherOpd = IdxMulOpd == 1 ? 2 : 1;
  MachineInstr *MUL = MRI.getUniqueVRegDef(Root.getOperand(IdxMulOpd).getReg());
  MachineOperand R = Root.getOperand(0);
  MachineOperand A = MUL->getOperand(1);
  MachineOperand B = MUL->getOperand(2);
  MachineOperand C = Root.getOperand(IdxOtherOpd);
  MachineInstrBuilder MIB = BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc))
                                .addOperand(R)
                                .addOperand(A)
                                .addOperand(B)
                                .addOperand(C);
  // Insert the MADD
  InsInstrs.push_back(MIB);
  return MUL;
}

/// genMaddR - Generate madd instruction and combine mul and add using
/// an extra virtual register
/// Example - an ADD intermediate needs to be stored in a register:
///   MUL I=A,B,0
///   ADD R,I,Imm
///   ==> ORR  V, ZR, Imm
///   ==> MADD R,A,B,V
/// \param Root is the ADD instruction
/// \param [out] InsInstrs is a vector of machine instructions and will
/// contain the generated madd instruction
/// \param IdxMulOpd is index of operand in Root that is the result of
/// the MUL. In the example above IdxMulOpd is 1.
/// \param MaddOpc the opcode fo the madd instruction
/// \param VR is a virtual register that holds the value of an ADD operand
/// (V in the example above).
static MachineInstr *genMaddR(MachineFunction &MF, MachineRegisterInfo &MRI,
                              const TargetInstrInfo *TII, MachineInstr &Root,
                              SmallVectorImpl<MachineInstr *> &InsInstrs,
                              unsigned IdxMulOpd, unsigned MaddOpc,
                              unsigned VR) {
  assert(IdxMulOpd == 1 || IdxMulOpd == 2);

  MachineInstr *MUL = MRI.getUniqueVRegDef(Root.getOperand(IdxMulOpd).getReg());
  MachineOperand R = Root.getOperand(0);
  MachineOperand A = MUL->getOperand(1);
  MachineOperand B = MUL->getOperand(2);
  MachineInstrBuilder MIB = BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc))
                                .addOperand(R)
                                .addOperand(A)
                                .addOperand(B)
                                .addReg(VR);
  // Insert the MADD
  InsInstrs.push_back(MIB);
  return MUL;
}
/// genAlternativeCodeSequence - when hasPattern() finds a pattern
/// this function generates the instructions that could replace the
/// original code sequence
void AArch64InstrInfo::genAlternativeCodeSequence(
    MachineInstr &Root, MachineCombinerPattern::MC_PATTERN Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const {
  MachineBasicBlock &MBB = *Root.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getTarget().getSubtargetImpl()->getInstrInfo();

  MachineInstr *MUL;
  unsigned Opc;
  switch (Pattern) {
  default:
    // signal error.
    break;
  case MachineCombinerPattern::MC_MULADDW_OP1:
  case MachineCombinerPattern::MC_MULADDX_OP1:
    // MUL I=A,B,0
    // ADD R,I,C
    // ==> MADD R,A,B,C
    // --- Create(MADD);
    Opc = Pattern == MachineCombinerPattern::MC_MULADDW_OP1 ? AArch64::MADDWrrr
                                                            : AArch64::MADDXrrr;
    MUL = genMadd(MF, MRI, TII, Root, InsInstrs, 1, Opc);
    break;
  case MachineCombinerPattern::MC_MULADDW_OP2:
  case MachineCombinerPattern::MC_MULADDX_OP2:
    // MUL I=A,B,0
    // ADD R,C,I
    // ==> MADD R,A,B,C
    // --- Create(MADD);
    Opc = Pattern == MachineCombinerPattern::MC_MULADDW_OP2 ? AArch64::MADDWrrr
                                                            : AArch64::MADDXrrr;
    MUL = genMadd(MF, MRI, TII, Root, InsInstrs, 2, Opc);
    break;
  case MachineCombinerPattern::MC_MULADDWI_OP1:
  case MachineCombinerPattern::MC_MULADDXI_OP1:
    // MUL I=A,B,0
    // ADD R,I,Imm
    // ==> ORR  V, ZR, Imm
    // ==> MADD R,A,B,V
    // --- Create(MADD);
    {
      const TargetRegisterClass *RC =
          MRI.getRegClass(Root.getOperand(1).getReg());
      unsigned NewVR = MRI.createVirtualRegister(RC);
      unsigned BitSize, OrrOpc, ZeroReg;
      if (Pattern == MachineCombinerPattern::MC_MULADDWI_OP1) {
        BitSize = 32;
        OrrOpc = AArch64::ORRWri;
        ZeroReg = AArch64::WZR;
        Opc = AArch64::MADDWrrr;
      } else {
        OrrOpc = AArch64::ORRXri;
        BitSize = 64;
        ZeroReg = AArch64::XZR;
        Opc = AArch64::MADDXrrr;
      }
      uint64_t Imm = Root.getOperand(2).getImm();

      if (Root.getOperand(3).isImm()) {
        unsigned val = Root.getOperand(3).getImm();
        Imm = Imm << val;
      }
      uint64_t UImm = Imm << (64 - BitSize) >> (64 - BitSize);
      uint64_t Encoding;

      if (AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
        MachineInstrBuilder MIB1 =
            BuildMI(MF, Root.getDebugLoc(), TII->get(OrrOpc))
                .addOperand(MachineOperand::CreateReg(NewVR, true))
                .addReg(ZeroReg)
                .addImm(Encoding);
        InsInstrs.push_back(MIB1);
        InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
        MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR);
      }
    }
    break;
  case MachineCombinerPattern::MC_MULSUBW_OP1:
  case MachineCombinerPattern::MC_MULSUBX_OP1: {
    // MUL I=A,B,0
    // SUB R,I, C
    // ==> SUB  V, 0, C
    // ==> MADD R,A,B,V // = -C + A*B
    // --- Create(MADD);
    const TargetRegisterClass *RC =
        MRI.getRegClass(Root.getOperand(1).getReg());
    unsigned NewVR = MRI.createVirtualRegister(RC);
    unsigned SubOpc, ZeroReg;
    if (Pattern == MachineCombinerPattern::MC_MULSUBW_OP1) {
      SubOpc = AArch64::SUBWrr;
      ZeroReg = AArch64::WZR;
      Opc = AArch64::MADDWrrr;
    } else {
      SubOpc = AArch64::SUBXrr;
      ZeroReg = AArch64::XZR;
      Opc = AArch64::MADDXrrr;
    }
    // SUB NewVR, 0, C
    MachineInstrBuilder MIB1 =
        BuildMI(MF, Root.getDebugLoc(), TII->get(SubOpc))
            .addOperand(MachineOperand::CreateReg(NewVR, true))
            .addReg(ZeroReg)
            .addOperand(Root.getOperand(2));
    InsInstrs.push_back(MIB1);
    InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
    MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR);
  } break;
  case MachineCombinerPattern::MC_MULSUBW_OP2:
  case MachineCombinerPattern::MC_MULSUBX_OP2:
    // MUL I=A,B,0
    // SUB R,C,I
    // ==> MSUB R,A,B,C (computes C - A*B)
    // --- Create(MSUB);
    Opc = Pattern == MachineCombinerPattern::MC_MULSUBW_OP2 ? AArch64::MSUBWrrr
                                                            : AArch64::MSUBXrrr;
    MUL = genMadd(MF, MRI, TII, Root, InsInstrs, 2, Opc);
    break;
  case MachineCombinerPattern::MC_MULSUBWI_OP1:
  case MachineCombinerPattern::MC_MULSUBXI_OP1: {
    // MUL I=A,B,0
    // SUB R,I, Imm
    // ==> ORR  V, ZR, -Imm
    // ==> MADD R,A,B,V // = -Imm + A*B
    // --- Create(MADD);
    const TargetRegisterClass *RC =
        MRI.getRegClass(Root.getOperand(1).getReg());
    unsigned NewVR = MRI.createVirtualRegister(RC);
    unsigned BitSize, OrrOpc, ZeroReg;
    if (Pattern == MachineCombinerPattern::MC_MULSUBWI_OP1) {
      BitSize = 32;
      OrrOpc = AArch64::ORRWri;
      ZeroReg = AArch64::WZR;
      Opc = AArch64::MADDWrrr;
    } else {
      OrrOpc = AArch64::ORRXri;
      BitSize = 64;
      ZeroReg = AArch64::XZR;
      Opc = AArch64::MADDXrrr;
    }
    int Imm = Root.getOperand(2).getImm();
    if (Root.getOperand(3).isImm()) {
      unsigned val = Root.getOperand(3).getImm();
      Imm = Imm << val;
    }
    uint64_t UImm = -Imm << (64 - BitSize) >> (64 - BitSize);
    uint64_t Encoding;
    if (AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
      MachineInstrBuilder MIB1 =
          BuildMI(MF, Root.getDebugLoc(), TII->get(OrrOpc))
              .addOperand(MachineOperand::CreateReg(NewVR, true))
              .addReg(ZeroReg)
              .addImm(Encoding);
      InsInstrs.push_back(MIB1);
      InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
      MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR);
    }
  } break;
  }
  // Record MUL and ADD/SUB for deletion
  DelInstrs.push_back(MUL);
  DelInstrs.push_back(&Root);

  return;
}
