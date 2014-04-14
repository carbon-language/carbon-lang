//===- ARM64InstrInfo.cpp - ARM64 Instruction Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM64InstrInfo.h"
#include "ARM64Subtarget.h"
#include "MCTargetDesc/ARM64AddressingModes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "ARM64GenInstrInfo.inc"

using namespace llvm;

ARM64InstrInfo::ARM64InstrInfo(const ARM64Subtarget &STI)
    : ARM64GenInstrInfo(ARM64::ADJCALLSTACKDOWN, ARM64::ADJCALLSTACKUP),
      RI(this, &STI), Subtarget(STI) {}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
unsigned ARM64InstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
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
  case ARM64::Bcc:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(LastInst->getOperand(0));
    break;
  case ARM64::CBZW:
  case ARM64::CBZX:
  case ARM64::CBNZW:
  case ARM64::CBNZX:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    break;
  case ARM64::TBZ:
  case ARM64::TBNZ:
    Target = LastInst->getOperand(2).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    Cond.push_back(LastInst->getOperand(1));
  }
}

// Branch analysis.
bool ARM64InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
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

bool ARM64InstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    ARM64CC::CondCode CC = (ARM64CC::CondCode)(int)Cond[0].getImm();
    Cond[0].setImm(ARM64CC::getInvertedCondCode(CC));
  } else {
    // Folded compare-and-branch
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown conditional branch!");
    case ARM64::CBZW:
      Cond[1].setImm(ARM64::CBNZW);
      break;
    case ARM64::CBNZW:
      Cond[1].setImm(ARM64::CBZW);
      break;
    case ARM64::CBZX:
      Cond[1].setImm(ARM64::CBNZX);
      break;
    case ARM64::CBNZX:
      Cond[1].setImm(ARM64::CBZX);
      break;
    case ARM64::TBZ:
      Cond[1].setImm(ARM64::TBNZ);
      break;
    case ARM64::TBNZ:
      Cond[1].setImm(ARM64::TBZ);
      break;
    }
  }

  return false;
}

unsigned ARM64InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
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

void ARM64InstrInfo::instantiateCondBranch(
    MachineBasicBlock &MBB, DebugLoc DL, MachineBasicBlock *TBB,
    const SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    BuildMI(&MBB, DL, get(ARM64::Bcc)).addImm(Cond[0].getImm()).addMBB(TBB);
  } else {
    // Folded compare-and-branch
    const MachineInstrBuilder MIB =
        BuildMI(&MBB, DL, get(Cond[1].getImm())).addReg(Cond[2].getReg());
    if (Cond.size() > 3)
      MIB.addImm(Cond[3].getImm());
    MIB.addMBB(TBB);
  }
}

unsigned ARM64InstrInfo::InsertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    const SmallVectorImpl<MachineOperand> &Cond, DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, DL, get(ARM64::B)).addMBB(TBB);
    else
      instantiateCondBranch(MBB, DL, TBB, Cond);
    return 1;
  }

  // Two-way conditional branch.
  instantiateCondBranch(MBB, DL, TBB, Cond);
  BuildMI(&MBB, DL, get(ARM64::B)).addMBB(FBB);
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
                                unsigned *NewVReg = 0) {
  VReg = removeCopies(MRI, VReg);
  if (!TargetRegisterInfo::isVirtualRegister(VReg))
    return 0;

  bool Is64Bit = ARM64::GPR64allRegClass.hasSubClassEq(MRI.getRegClass(VReg));
  const MachineInstr *DefMI = MRI.getVRegDef(VReg);
  unsigned Opc = 0;
  unsigned SrcOpNum = 0;
  switch (DefMI->getOpcode()) {
  case ARM64::ADDSXri:
  case ARM64::ADDSWri:
    // if CPSR is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(ARM64::CPSR, true) == -1)
      return 0;
  // fall-through to ADDXri and ADDWri.
  case ARM64::ADDXri:
  case ARM64::ADDWri:
    // add x, 1 -> csinc.
    if (!DefMI->getOperand(2).isImm() || DefMI->getOperand(2).getImm() != 1 ||
        DefMI->getOperand(3).getImm() != 0)
      return 0;
    SrcOpNum = 1;
    Opc = Is64Bit ? ARM64::CSINCXr : ARM64::CSINCWr;
    break;

  case ARM64::ORNXrr:
  case ARM64::ORNWrr: {
    // not x -> csinv, represented as orn dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != ARM64::XZR && ZReg != ARM64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? ARM64::CSINVXr : ARM64::CSINVWr;
    break;
  }

  case ARM64::SUBSXrr:
  case ARM64::SUBSWrr:
    // if CPSR is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(ARM64::CPSR, true) == -1)
      return 0;
  // fall-through to SUBXrr and SUBWrr.
  case ARM64::SUBXrr:
  case ARM64::SUBWrr: {
    // neg x -> csneg, represented as sub dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != ARM64::XZR && ZReg != ARM64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? ARM64::CSNEGXr : ARM64::CSNEGWr;
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

bool ARM64InstrInfo::canInsertSelect(
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
  if (ARM64::GPR64allRegClass.hasSubClassEq(RC) ||
      ARM64::GPR32allRegClass.hasSubClassEq(RC)) {
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
  if (ARM64::FPR64RegClass.hasSubClassEq(RC) ||
      ARM64::FPR32RegClass.hasSubClassEq(RC)) {
    CondCycles = 5 + ExtraCondLat;
    TrueCycles = FalseCycles = 2;
    return true;
  }

  // Can't do vectors.
  return false;
}

void ARM64InstrInfo::insertSelect(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I, DebugLoc DL,
                                  unsigned DstReg,
                                  const SmallVectorImpl<MachineOperand> &Cond,
                                  unsigned TrueReg, unsigned FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Parse the condition code, see parseCondBranch() above.
  ARM64CC::CondCode CC;
  switch (Cond.size()) {
  default:
    llvm_unreachable("Unknown condition opcode in Cond");
  case 1: // b.cc
    CC = ARM64CC::CondCode(Cond[0].getImm());
    break;
  case 3: { // cbz/cbnz
    // We must insert a compare against 0.
    bool Is64Bit;
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown branch opcode in Cond");
    case ARM64::CBZW:
      Is64Bit = 0;
      CC = ARM64CC::EQ;
      break;
    case ARM64::CBZX:
      Is64Bit = 1;
      CC = ARM64CC::EQ;
      break;
    case ARM64::CBNZW:
      Is64Bit = 0;
      CC = ARM64CC::NE;
      break;
    case ARM64::CBNZX:
      Is64Bit = 1;
      CC = ARM64CC::NE;
      break;
    }
    unsigned SrcReg = Cond[2].getReg();
    if (Is64Bit) {
      // cmp reg, #0 is actually subs xzr, reg, #0.
      MRI.constrainRegClass(SrcReg, &ARM64::GPR64spRegClass);
      BuildMI(MBB, I, DL, get(ARM64::SUBSXri), ARM64::XZR)
          .addReg(SrcReg)
          .addImm(0)
          .addImm(0);
    } else {
      MRI.constrainRegClass(SrcReg, &ARM64::GPR32spRegClass);
      BuildMI(MBB, I, DL, get(ARM64::SUBSWri), ARM64::WZR)
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
    case ARM64::TBZ:
      CC = ARM64CC::EQ;
      break;
    case ARM64::TBNZ:
      CC = ARM64CC::NE;
      break;
    }
    // cmp reg, #foo is actually ands xzr, reg, #1<<foo.
    BuildMI(MBB, I, DL, get(ARM64::ANDSXri), ARM64::XZR)
        .addReg(Cond[2].getReg())
        .addImm(ARM64_AM::encodeLogicalImmediate(1ull << Cond[3].getImm(), 64));
    break;
  }
  }

  unsigned Opc = 0;
  const TargetRegisterClass *RC = 0;
  bool TryFold = false;
  if (MRI.constrainRegClass(DstReg, &ARM64::GPR64RegClass)) {
    RC = &ARM64::GPR64RegClass;
    Opc = ARM64::CSELXr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &ARM64::GPR32RegClass)) {
    RC = &ARM64::GPR32RegClass;
    Opc = ARM64::CSELWr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &ARM64::FPR64RegClass)) {
    RC = &ARM64::FPR64RegClass;
    Opc = ARM64::FCSELDrrr;
  } else if (MRI.constrainRegClass(DstReg, &ARM64::FPR32RegClass)) {
    RC = &ARM64::FPR32RegClass;
    Opc = ARM64::FCSELSrrr;
  }
  assert(RC && "Unsupported regclass");

  // Try folding simple instructions into the csel.
  if (TryFold) {
    unsigned NewVReg = 0;
    unsigned FoldedOpc = canFoldIntoCSel(MRI, TrueReg, &NewVReg);
    if (FoldedOpc) {
      // The folded opcodes csinc, csinc and csneg apply the operation to
      // FalseReg, so we need to invert the condition.
      CC = ARM64CC::getInvertedCondCode(CC);
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

bool ARM64InstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                           unsigned &SrcReg, unsigned &DstReg,
                                           unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default:
    return false;
  case ARM64::SBFMXri: // aka sxtw
  case ARM64::UBFMXri: // aka uxtw
    // Check for the 32 -> 64 bit extension case, these instructions can do
    // much more.
    if (MI.getOperand(2).getImm() != 0 || MI.getOperand(3).getImm() != 31)
      return false;
    // This is a signed or unsigned 32 -> 64 bit extension.
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SubIdx = ARM64::sub_32;
    return true;
  }
}

/// analyzeCompare - For a comparison instruction, return the source registers
/// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
/// Return true if the comparison instruction can be analyzed.
bool ARM64InstrInfo::analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                                    unsigned &SrcReg2, int &CmpMask,
                                    int &CmpValue) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::SUBSWrr:
  case ARM64::SUBSWrs:
  case ARM64::SUBSWrx:
  case ARM64::SUBSXrr:
  case ARM64::SUBSXrs:
  case ARM64::SUBSXrx:
  case ARM64::ADDSWrr:
  case ARM64::ADDSWrs:
  case ARM64::ADDSWrx:
  case ARM64::ADDSXrr:
  case ARM64::ADDSXrs:
  case ARM64::ADDSXrx:
    // Replace SUBSWrr with SUBWrr if CPSR is not used.
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = MI->getOperand(2).getReg();
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case ARM64::SUBSWri:
  case ARM64::ADDSWri:
  case ARM64::ANDSWri:
  case ARM64::SUBSXri:
  case ARM64::ADDSXri:
  case ARM64::ANDSXri:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    CmpValue = MI->getOperand(2).getImm();
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
  const TargetInstrInfo *TII = TM->getInstrInfo();
  const TargetRegisterInfo *TRI = TM->getRegisterInfo();
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

/// optimizeCompareInstr - Convert the instruction supplying the argument to the
/// comparison into one that sets the zero bit in the flags register.
bool ARM64InstrInfo::optimizeCompareInstr(
    MachineInstr *CmpInstr, unsigned SrcReg, unsigned SrcReg2, int CmpMask,
    int CmpValue, const MachineRegisterInfo *MRI) const {

  // Replace SUBSWrr with SUBWrr if CPSR is not used.
  int Cmp_CPSR = CmpInstr->findRegisterDefOperandIdx(ARM64::CPSR, true);
  if (Cmp_CPSR != -1) {
    unsigned NewOpc;
    switch (CmpInstr->getOpcode()) {
    default:
      return false;
    case ARM64::ADDSWrr:      NewOpc = ARM64::ADDWrr; break;
    case ARM64::ADDSWri:      NewOpc = ARM64::ADDWri; break;
    case ARM64::ADDSWrs:      NewOpc = ARM64::ADDWrs; break;
    case ARM64::ADDSWrx:      NewOpc = ARM64::ADDWrx; break;
    case ARM64::ADDSXrr:      NewOpc = ARM64::ADDXrr; break;
    case ARM64::ADDSXri:      NewOpc = ARM64::ADDXri; break;
    case ARM64::ADDSXrs:      NewOpc = ARM64::ADDXrs; break;
    case ARM64::ADDSXrx:      NewOpc = ARM64::ADDXrx; break;
    case ARM64::SUBSWrr:      NewOpc = ARM64::SUBWrr; break;
    case ARM64::SUBSWri:      NewOpc = ARM64::SUBWri; break;
    case ARM64::SUBSWrs:      NewOpc = ARM64::SUBWrs; break;
    case ARM64::SUBSWrx:      NewOpc = ARM64::SUBWrx; break;
    case ARM64::SUBSXrr:      NewOpc = ARM64::SUBXrr; break;
    case ARM64::SUBSXri:      NewOpc = ARM64::SUBXri; break;
    case ARM64::SUBSXrs:      NewOpc = ARM64::SUBXrs; break;
    case ARM64::SUBSXrx:      NewOpc = ARM64::SUBXrx; break;
    }

    const MCInstrDesc &MCID = get(NewOpc);
    CmpInstr->setDesc(MCID);
    CmpInstr->RemoveOperand(Cmp_CPSR);
    bool succeeded = UpdateOperandRegClass(CmpInstr);
    (void)succeeded;
    assert(succeeded && "Some operands reg class are incompatible!");
    return true;
  }

  // Continue only if we have a "ri" where immediate is zero.
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
  // basic block, to check whether CPSR is used or modified in between.
  MachineBasicBlock::iterator I = CmpInstr, E = MI,
                              B = CmpInstr->getParent()->begin();

  // Early exit if CmpInstr is at the beginning of the BB.
  if (I == B)
    return false;

  // Check whether the definition of SrcReg is in the same basic block as
  // Compare. If not, we can't optimize away the Compare.
  if (MI->getParent() != CmpInstr->getParent())
    return false;

  // Check that CPSR isn't set between the comparison instruction and the one we
  // want to change.
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  for (--I; I != E; --I) {
    const MachineInstr &Instr = *I;

    if (Instr.modifiesRegister(ARM64::CPSR, TRI) ||
        Instr.readsRegister(ARM64::CPSR, TRI))
      // This instruction modifies or uses CPSR after the one we want to
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
  case ARM64::ADDSWrr:
  case ARM64::ADDSWri:
  case ARM64::ADDSXrr:
  case ARM64::ADDSXri:
  case ARM64::SUBSWrr:
  case ARM64::SUBSWri:
  case ARM64::SUBSXrr:
  case ARM64::SUBSXri:
    break;
  case ARM64::ADDWrr:    NewOpc = ARM64::ADDSWrr; break;
  case ARM64::ADDWri:    NewOpc = ARM64::ADDSWri; break;
  case ARM64::ADDXrr:    NewOpc = ARM64::ADDSXrr; break;
  case ARM64::ADDXri:    NewOpc = ARM64::ADDSXri; break;
  case ARM64::ADCWr:     NewOpc = ARM64::ADCSWr; break;
  case ARM64::ADCXr:     NewOpc = ARM64::ADCSXr; break;
  case ARM64::SUBWrr:    NewOpc = ARM64::SUBSWrr; break;
  case ARM64::SUBWri:    NewOpc = ARM64::SUBSWri; break;
  case ARM64::SUBXrr:    NewOpc = ARM64::SUBSXrr; break;
  case ARM64::SUBXri:    NewOpc = ARM64::SUBSXri; break;
  case ARM64::SBCWr:     NewOpc = ARM64::SBCSWr; break;
  case ARM64::SBCXr:     NewOpc = ARM64::SBCSXr; break;
  case ARM64::ANDWri:    NewOpc = ARM64::ANDSWri; break;
  case ARM64::ANDXri:    NewOpc = ARM64::ANDSXri; break;
  }

  // Scan forward for the use of CPSR.
  // When checking against MI: if it's a conditional code requires
  // checking of V bit, then this is not safe to do.
  // It is safe to remove CmpInstr if CPSR is redefined or killed.
  // If we are done with the basic block, we need to check whether CPSR is
  // live-out.
  bool IsSafe = false;
  for (MachineBasicBlock::iterator I = CmpInstr,
                                   E = CmpInstr->getParent()->end();
       !IsSafe && ++I != E;) {
    const MachineInstr &Instr = *I;
    for (unsigned IO = 0, EO = Instr.getNumOperands(); !IsSafe && IO != EO;
         ++IO) {
      const MachineOperand &MO = Instr.getOperand(IO);
      if (MO.isRegMask() && MO.clobbersPhysReg(ARM64::CPSR)) {
        IsSafe = true;
        break;
      }
      if (!MO.isReg() || MO.getReg() != ARM64::CPSR)
        continue;
      if (MO.isDef()) {
        IsSafe = true;
        break;
      }

      // Decode the condition code.
      unsigned Opc = Instr.getOpcode();
      ARM64CC::CondCode CC;
      switch (Opc) {
      default:
        return false;
      case ARM64::Bcc:
        CC = (ARM64CC::CondCode)Instr.getOperand(IO - 2).getImm();
        break;
      case ARM64::CSINVWr:
      case ARM64::CSINVXr:
      case ARM64::CSINCWr:
      case ARM64::CSINCXr:
      case ARM64::CSELWr:
      case ARM64::CSELXr:
      case ARM64::CSNEGWr:
      case ARM64::CSNEGXr:
      case ARM64::FCSELSrrr:
      case ARM64::FCSELDrrr:
        CC = (ARM64CC::CondCode)Instr.getOperand(IO - 1).getImm();
        break;
      }

      // It is not safe to remove Compare instruction if Overflow(V) is used.
      switch (CC) {
      default:
        // CPSR can be used multiple times, we should continue.
        break;
      case ARM64CC::VS:
      case ARM64CC::VC:
      case ARM64CC::GE:
      case ARM64CC::LT:
      case ARM64CC::GT:
      case ARM64CC::LE:
        return false;
      }
    }
  }

  // If CPSR is not killed nor re-defined, we should check whether it is
  // live-out. If it is live-out, do not optimize.
  if (!IsSafe) {
    MachineBasicBlock *ParentBlock = CmpInstr->getParent();
    for (auto *MBB : ParentBlock->successors())
      if (MBB->isLiveIn(ARM64::CPSR))
        return false;
  }

  // Update the instruction to set CPSR.
  MI->setDesc(get(NewOpc));
  CmpInstr->eraseFromParent();
  bool succeeded = UpdateOperandRegClass(MI);
  (void)succeeded;
  assert(succeeded && "Some operands reg class are incompatible!");
  MI->addRegisterDefined(ARM64::CPSR, TRI);
  return true;
}

// Return true if this instruction simply sets its single destination register
// to zero. This is equivalent to a register rename of the zero-register.
bool ARM64InstrInfo::isGPRZero(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::MOVZWi:
  case ARM64::MOVZXi: // movz Rd, #0 (LSL #0)
    if (MI->getOperand(1).isImm() && MI->getOperand(1).getImm() == 0) {
      assert(MI->getDesc().getNumOperands() == 3 &&
             MI->getOperand(2).getImm() == 0 && "invalid MOVZi operands");
      return true;
    }
    break;
  case ARM64::ANDWri: // and Rd, Rzr, #imm
    return MI->getOperand(1).getReg() == ARM64::WZR;
  case ARM64::ANDXri:
    return MI->getOperand(1).getReg() == ARM64::XZR;
  case TargetOpcode::COPY:
    return MI->getOperand(1).getReg() == ARM64::WZR;
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool ARM64InstrInfo::isGPRCopy(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // GPR32 copies will by lowered to ORRXrs
    unsigned DstReg = MI->getOperand(0).getReg();
    return (ARM64::GPR32RegClass.contains(DstReg) ||
            ARM64::GPR64RegClass.contains(DstReg));
  }
  case ARM64::ORRXrs: // orr Xd, Xzr, Xm (LSL #0)
    if (MI->getOperand(1).getReg() == ARM64::XZR) {
      assert(MI->getDesc().getNumOperands() == 4 &&
             MI->getOperand(3).getImm() == 0 && "invalid ORRrs operands");
      return true;
    }
  case ARM64::ADDXri: // add Xd, Xn, #0 (LSL #0)
    if (MI->getOperand(2).getImm() == 0) {
      assert(MI->getDesc().getNumOperands() == 4 &&
             MI->getOperand(3).getImm() == 0 && "invalid ADDXri operands");
      return true;
    }
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool ARM64InstrInfo::isFPRCopy(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // FPR64 copies will by lowered to ORR.16b
    unsigned DstReg = MI->getOperand(0).getReg();
    return (ARM64::FPR64RegClass.contains(DstReg) ||
            ARM64::FPR128RegClass.contains(DstReg));
  }
  case ARM64::ORRv16i8:
    if (MI->getOperand(1).getReg() == MI->getOperand(2).getReg()) {
      assert(MI->getDesc().getNumOperands() == 3 && MI->getOperand(0).isReg() &&
             "invalid ORRv16i8 operands");
      return true;
    }
  }
  return false;
}

unsigned ARM64InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::LDRWui:
  case ARM64::LDRXui:
  case ARM64::LDRBui:
  case ARM64::LDRHui:
  case ARM64::LDRSui:
  case ARM64::LDRDui:
  case ARM64::LDRQui:
    if (MI->getOperand(0).getSubReg() == 0 && MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() && MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

unsigned ARM64InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::STRWui:
  case ARM64::STRXui:
  case ARM64::STRBui:
  case ARM64::STRHui:
  case ARM64::STRSui:
  case ARM64::STRDui:
  case ARM64::STRQui:
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
bool ARM64InstrInfo::isScaledAddr(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::LDRBBro:
  case ARM64::LDRBro:
  case ARM64::LDRDro:
  case ARM64::LDRHHro:
  case ARM64::LDRHro:
  case ARM64::LDRQro:
  case ARM64::LDRSBWro:
  case ARM64::LDRSBXro:
  case ARM64::LDRSHWro:
  case ARM64::LDRSHXro:
  case ARM64::LDRSWro:
  case ARM64::LDRSro:
  case ARM64::LDRWro:
  case ARM64::LDRXro:
  case ARM64::STRBBro:
  case ARM64::STRBro:
  case ARM64::STRDro:
  case ARM64::STRHHro:
  case ARM64::STRHro:
  case ARM64::STRQro:
  case ARM64::STRSro:
  case ARM64::STRWro:
  case ARM64::STRXro:
    unsigned Val = MI->getOperand(3).getImm();
    ARM64_AM::ExtendType ExtType = ARM64_AM::getMemExtendType(Val);
    return (ExtType != ARM64_AM::UXTX) || ARM64_AM::getMemDoShift(Val);
  }
  return false;
}

/// Check all MachineMemOperands for a hint to suppress pairing.
bool ARM64InstrInfo::isLdStPairSuppressed(const MachineInstr *MI) const {
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
void ARM64InstrInfo::suppressLdStPair(MachineInstr *MI) const {
  if (MI->memoperands_empty())
    return;

  assert(MOSuppressPair < (1 << MachineMemOperand::MOTargetNumBits) &&
         "Too many target MO flags");
  (*MI->memoperands_begin())
      ->setFlags(MOSuppressPair << MachineMemOperand::MOTargetStartBit);
}

bool ARM64InstrInfo::getLdStBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                                          unsigned &Offset,
                                          const TargetRegisterInfo *TRI) const {
  switch (LdSt->getOpcode()) {
  default:
    return false;
  case ARM64::STRSui:
  case ARM64::STRDui:
  case ARM64::STRQui:
  case ARM64::STRXui:
  case ARM64::STRWui:
  case ARM64::LDRSui:
  case ARM64::LDRDui:
  case ARM64::LDRQui:
  case ARM64::LDRXui:
  case ARM64::LDRWui:
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
bool ARM64InstrInfo::shouldClusterLoads(MachineInstr *FirstLdSt,
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

bool ARM64InstrInfo::shouldScheduleAdjacent(MachineInstr *First,
                                            MachineInstr *Second) const {
  // Cyclone can fuse CMN, CMP followed by Bcc.

  // FIXME: B0 can also fuse:
  // AND, BIC, ORN, ORR, or EOR (optional S) followed by Bcc or CBZ or CBNZ.
  if (Second->getOpcode() != ARM64::Bcc)
    return false;
  switch (First->getOpcode()) {
  default:
    return false;
  case ARM64::SUBSWri:
  case ARM64::ADDSWri:
  case ARM64::ANDSWri:
  case ARM64::SUBSXri:
  case ARM64::ADDSXri:
  case ARM64::ANDSXri:
    return true;
  }
}

MachineInstr *ARM64InstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                                       int FrameIx,
                                                       uint64_t Offset,
                                                       const MDNode *MDPtr,
                                                       DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(ARM64::DBG_VALUE))
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

void ARM64InstrInfo::copyPhysRegTuple(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      DebugLoc DL, unsigned DestReg,
                                      unsigned SrcReg, bool KillSrc,
                                      unsigned Opcode,
                                      llvm::ArrayRef<unsigned> Indices) const {
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

void ARM64InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I, DebugLoc DL,
                                 unsigned DestReg, unsigned SrcReg,
                                 bool KillSrc) const {
  if (ARM64::GPR32spRegClass.contains(DestReg) &&
      (ARM64::GPR32spRegClass.contains(SrcReg) || SrcReg == ARM64::WZR)) {
    const TargetRegisterInfo *TRI = &getRegisterInfo();

    if (DestReg == ARM64::WSP || SrcReg == ARM64::WSP) {
      // If either operand is WSP, expand to ADD #0.
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ADD Xd, Xn, #0" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, ARM64::sub_32,
                                                     &ARM64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, ARM64::sub_32,
                                                    &ARM64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(ARM64::ADDXri), DestRegX)
            .addReg(SrcRegX, RegState::Undef)
            .addImm(0)
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, 0))
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        BuildMI(MBB, I, DL, get(ARM64::ADDWri), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc))
            .addImm(0)
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, 0));
      }
    } else if (SrcReg == ARM64::WZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(ARM64::MOVZWi), DestReg).addImm(0).addImm(
          ARM64_AM::getShifterImm(ARM64_AM::LSL, 0));
    } else {
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ORR Xd, XZR, Xm" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, ARM64::sub_32,
                                                     &ARM64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, ARM64::sub_32,
                                                    &ARM64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(ARM64::ORRXrr), DestRegX)
            .addReg(ARM64::XZR)
            .addReg(SrcRegX, RegState::Undef)
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        // Otherwise, expand to ORR WZR.
        BuildMI(MBB, I, DL, get(ARM64::ORRWrr), DestReg)
            .addReg(ARM64::WZR)
            .addReg(SrcReg, getKillRegState(KillSrc));
      }
    }
    return;
  }

  if (ARM64::GPR64spRegClass.contains(DestReg) &&
      (ARM64::GPR64spRegClass.contains(SrcReg) || SrcReg == ARM64::XZR)) {
    if (DestReg == ARM64::SP || SrcReg == ARM64::SP) {
      // If either operand is SP, expand to ADD #0.
      BuildMI(MBB, I, DL, get(ARM64::ADDXri), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0)
          .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, 0));
    } else if (SrcReg == ARM64::XZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(ARM64::MOVZXi), DestReg).addImm(0).addImm(
          ARM64_AM::getShifterImm(ARM64_AM::LSL, 0));
    } else {
      // Otherwise, expand to ORR XZR.
      BuildMI(MBB, I, DL, get(ARM64::ORRXrr), DestReg)
          .addReg(ARM64::XZR)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  // Copy a DDDD register quad by copying the individual sub-registers.
  if (ARM64::DDDDRegClass.contains(DestReg) &&
      ARM64::DDDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::dsub0, ARM64::dsub1,
                                        ARM64::dsub2, ARM64::dsub3 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DDD register triple by copying the individual sub-registers.
  if (ARM64::DDDRegClass.contains(DestReg) &&
      ARM64::DDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::dsub0, ARM64::dsub1,
                                        ARM64::dsub2 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DD register pair by copying the individual sub-registers.
  if (ARM64::DDRegClass.contains(DestReg) &&
      ARM64::DDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::dsub0, ARM64::dsub1 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a QQQQ register quad by copying the individual sub-registers.
  if (ARM64::QQQQRegClass.contains(DestReg) &&
      ARM64::QQQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::qsub0, ARM64::qsub1,
                                        ARM64::qsub2, ARM64::qsub3 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQQ register triple by copying the individual sub-registers.
  if (ARM64::QQQRegClass.contains(DestReg) &&
      ARM64::QQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::qsub0, ARM64::qsub1,
                                        ARM64::qsub2 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQ register pair by copying the individual sub-registers.
  if (ARM64::QQRegClass.contains(DestReg) &&
      ARM64::QQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = { ARM64::qsub0, ARM64::qsub1 };
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, ARM64::ORRv16i8,
                     Indices);
    return;
  }

  if (ARM64::FPR128RegClass.contains(DestReg) &&
      ARM64::FPR128RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(ARM64::ORRv16i8), DestReg).addReg(SrcReg).addReg(
        SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (ARM64::FPR64RegClass.contains(DestReg) &&
      ARM64::FPR64RegClass.contains(SrcReg)) {
    DestReg =
        RI.getMatchingSuperReg(DestReg, ARM64::dsub, &ARM64::FPR128RegClass);
    SrcReg =
        RI.getMatchingSuperReg(SrcReg, ARM64::dsub, &ARM64::FPR128RegClass);
    BuildMI(MBB, I, DL, get(ARM64::ORRv16i8), DestReg).addReg(SrcReg).addReg(
        SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (ARM64::FPR32RegClass.contains(DestReg) &&
      ARM64::FPR32RegClass.contains(SrcReg)) {
    DestReg =
        RI.getMatchingSuperReg(DestReg, ARM64::ssub, &ARM64::FPR128RegClass);
    SrcReg =
        RI.getMatchingSuperReg(SrcReg, ARM64::ssub, &ARM64::FPR128RegClass);
    BuildMI(MBB, I, DL, get(ARM64::ORRv16i8), DestReg).addReg(SrcReg).addReg(
        SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (ARM64::FPR16RegClass.contains(DestReg) &&
      ARM64::FPR16RegClass.contains(SrcReg)) {
    DestReg =
        RI.getMatchingSuperReg(DestReg, ARM64::hsub, &ARM64::FPR128RegClass);
    SrcReg =
        RI.getMatchingSuperReg(SrcReg, ARM64::hsub, &ARM64::FPR128RegClass);
    BuildMI(MBB, I, DL, get(ARM64::ORRv16i8), DestReg).addReg(SrcReg).addReg(
        SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (ARM64::FPR8RegClass.contains(DestReg) &&
      ARM64::FPR8RegClass.contains(SrcReg)) {
    DestReg =
        RI.getMatchingSuperReg(DestReg, ARM64::bsub, &ARM64::FPR128RegClass);
    SrcReg =
        RI.getMatchingSuperReg(SrcReg, ARM64::bsub, &ARM64::FPR128RegClass);
    BuildMI(MBB, I, DL, get(ARM64::ORRv16i8), DestReg).addReg(SrcReg).addReg(
        SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Copies between GPR64 and FPR64.
  if (ARM64::FPR64RegClass.contains(DestReg) &&
      ARM64::GPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(ARM64::FMOVXDr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (ARM64::GPR64RegClass.contains(DestReg) &&
      ARM64::FPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(ARM64::FMOVDXr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  // Copies between GPR32 and FPR32.
  if (ARM64::FPR32RegClass.contains(DestReg) &&
      ARM64::GPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(ARM64::FMOVWSr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (ARM64::GPR32RegClass.contains(DestReg) &&
      ARM64::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(ARM64::FMOVSWr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  assert(0 && "unimplemented reg-to-reg copy");
}

void ARM64InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         unsigned SrcReg, bool isKill, int FI,
                                         const TargetRegisterClass *RC,
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
    if (ARM64::FPR8RegClass.hasSubClassEq(RC))
      Opc = ARM64::STRBui;
    break;
  case 2:
    if (ARM64::FPR16RegClass.hasSubClassEq(RC))
      Opc = ARM64::STRHui;
    break;
  case 4:
    if (ARM64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = ARM64::STRWui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &ARM64::GPR32RegClass);
      else
        assert(SrcReg != ARM64::WSP);
    } else if (ARM64::FPR32RegClass.hasSubClassEq(RC))
      Opc = ARM64::STRSui;
    break;
  case 8:
    if (ARM64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = ARM64::STRXui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &ARM64::GPR64RegClass);
      else
        assert(SrcReg != ARM64::SP);
    } else if (ARM64::FPR64RegClass.hasSubClassEq(RC))
      Opc = ARM64::STRDui;
    break;
  case 16:
    if (ARM64::FPR128RegClass.hasSubClassEq(RC))
      Opc = ARM64::STRQui;
    else if (ARM64::DDRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Twov1d, Offset = false;
    break;
  case 24:
    if (ARM64::DDDRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Threev1d, Offset = false;
    break;
  case 32:
    if (ARM64::DDDDRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Fourv1d, Offset = false;
    else if (ARM64::QQRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Twov2d, Offset = false;
    break;
  case 48:
    if (ARM64::QQQRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Threev2d, Offset = false;
    break;
  case 64:
    if (ARM64::QQQQRegClass.hasSubClassEq(RC))
      Opc = ARM64::ST1Fourv2d, Offset = false;
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

void ARM64InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned DestReg, int FI,
                                          const TargetRegisterClass *RC,
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
    if (ARM64::FPR8RegClass.hasSubClassEq(RC))
      Opc = ARM64::LDRBui;
    break;
  case 2:
    if (ARM64::FPR16RegClass.hasSubClassEq(RC))
      Opc = ARM64::LDRHui;
    break;
  case 4:
    if (ARM64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = ARM64::LDRWui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &ARM64::GPR32RegClass);
      else
        assert(DestReg != ARM64::WSP);
    } else if (ARM64::FPR32RegClass.hasSubClassEq(RC))
      Opc = ARM64::LDRSui;
    break;
  case 8:
    if (ARM64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = ARM64::LDRXui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &ARM64::GPR64RegClass);
      else
        assert(DestReg != ARM64::SP);
    } else if (ARM64::FPR64RegClass.hasSubClassEq(RC))
      Opc = ARM64::LDRDui;
    break;
  case 16:
    if (ARM64::FPR128RegClass.hasSubClassEq(RC))
      Opc = ARM64::LDRQui;
    else if (ARM64::DDRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Twov1d, Offset = false;
    break;
  case 24:
    if (ARM64::DDDRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Threev1d, Offset = false;
    break;
  case 32:
    if (ARM64::DDDDRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Fourv1d, Offset = false;
    else if (ARM64::QQRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Twov2d, Offset = false;
    break;
  case 48:
    if (ARM64::QQQRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Threev2d, Offset = false;
    break;
  case 64:
    if (ARM64::QQQQRegClass.hasSubClassEq(RC))
      Opc = ARM64::LD1Fourv2d, Offset = false;
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
                           const ARM64InstrInfo *TII, MachineInstr::MIFlag Flag,
                           bool SetCPSR) {
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
  if (SetCPSR)
    Opc = isSub ? ARM64::SUBSXri : ARM64::ADDSXri;
  else
    Opc = isSub ? ARM64::SUBXri : ARM64::ADDXri;
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
        .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, ShiftSize))
        .setMIFlag(Flag);

    SrcReg = DestReg;
    Offset -= ThisVal;
    if (Offset == 0)
      return;
  }
  BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
      .addReg(SrcReg)
      .addImm(Offset)
      .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, 0))
      .setMIFlag(Flag);
}

MachineInstr *
ARM64InstrInfo::foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
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
    if (SrcReg == ARM64::SP && TargetRegisterInfo::isVirtualRegister(DstReg)) {
      MF.getRegInfo().constrainRegClass(DstReg, &ARM64::GPR64RegClass);
      return 0;
    }
    if (DstReg == ARM64::SP && TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      MF.getRegInfo().constrainRegClass(SrcReg, &ARM64::GPR64RegClass);
      return 0;
    }
  }

  // Cannot fold.
  return 0;
}

int llvm::isARM64FrameOffsetLegal(const MachineInstr &MI, int &Offset,
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
    assert(0 && "unhandled opcode in rewriteARM64FrameIndex");
  // Vector spills/fills can't take an immediate offset.
  case ARM64::LD1Twov2d:
  case ARM64::LD1Threev2d:
  case ARM64::LD1Fourv2d:
  case ARM64::LD1Twov1d:
  case ARM64::LD1Threev1d:
  case ARM64::LD1Fourv1d:
  case ARM64::ST1Twov2d:
  case ARM64::ST1Threev2d:
  case ARM64::ST1Fourv2d:
  case ARM64::ST1Twov1d:
  case ARM64::ST1Threev1d:
  case ARM64::ST1Fourv1d:
    return ARM64FrameOffsetCannotUpdate;
  case ARM64::PRFMui:
    Scale = 8;
    UnscaledOp = ARM64::PRFUMi;
    break;
  case ARM64::LDRXui:
    Scale = 8;
    UnscaledOp = ARM64::LDURXi;
    break;
  case ARM64::LDRWui:
    Scale = 4;
    UnscaledOp = ARM64::LDURWi;
    break;
  case ARM64::LDRBui:
    Scale = 1;
    UnscaledOp = ARM64::LDURBi;
    break;
  case ARM64::LDRHui:
    Scale = 2;
    UnscaledOp = ARM64::LDURHi;
    break;
  case ARM64::LDRSui:
    Scale = 4;
    UnscaledOp = ARM64::LDURSi;
    break;
  case ARM64::LDRDui:
    Scale = 8;
    UnscaledOp = ARM64::LDURDi;
    break;
  case ARM64::LDRQui:
    Scale = 16;
    UnscaledOp = ARM64::LDURQi;
    break;
  case ARM64::LDRBBui:
    Scale = 1;
    UnscaledOp = ARM64::LDURBBi;
    break;
  case ARM64::LDRHHui:
    Scale = 2;
    UnscaledOp = ARM64::LDURHHi;
    break;
  case ARM64::LDRSBXui:
    Scale = 1;
    UnscaledOp = ARM64::LDURSBXi;
    break;
  case ARM64::LDRSBWui:
    Scale = 1;
    UnscaledOp = ARM64::LDURSBWi;
    break;
  case ARM64::LDRSHXui:
    Scale = 2;
    UnscaledOp = ARM64::LDURSHXi;
    break;
  case ARM64::LDRSHWui:
    Scale = 2;
    UnscaledOp = ARM64::LDURSHWi;
    break;
  case ARM64::LDRSWui:
    Scale = 4;
    UnscaledOp = ARM64::LDURSWi;
    break;

  case ARM64::STRXui:
    Scale = 8;
    UnscaledOp = ARM64::STURXi;
    break;
  case ARM64::STRWui:
    Scale = 4;
    UnscaledOp = ARM64::STURWi;
    break;
  case ARM64::STRBui:
    Scale = 1;
    UnscaledOp = ARM64::STURBi;
    break;
  case ARM64::STRHui:
    Scale = 2;
    UnscaledOp = ARM64::STURHi;
    break;
  case ARM64::STRSui:
    Scale = 4;
    UnscaledOp = ARM64::STURSi;
    break;
  case ARM64::STRDui:
    Scale = 8;
    UnscaledOp = ARM64::STURDi;
    break;
  case ARM64::STRQui:
    Scale = 16;
    UnscaledOp = ARM64::STURQi;
    break;
  case ARM64::STRBBui:
    Scale = 1;
    UnscaledOp = ARM64::STURBBi;
    break;
  case ARM64::STRHHui:
    Scale = 2;
    UnscaledOp = ARM64::STURHHi;
    break;

  case ARM64::LDPXi:
  case ARM64::LDPDi:
  case ARM64::STPXi:
  case ARM64::STPDi:
    IsSigned = true;
    Scale = 8;
    break;
  case ARM64::LDPQi:
  case ARM64::STPQi:
    IsSigned = true;
    Scale = 16;
    break;
  case ARM64::LDPWi:
  case ARM64::LDPSi:
  case ARM64::STPWi:
  case ARM64::STPSi:
    IsSigned = true;
    Scale = 4;
    break;

  case ARM64::LDURXi:
  case ARM64::LDURWi:
  case ARM64::LDURBi:
  case ARM64::LDURHi:
  case ARM64::LDURSi:
  case ARM64::LDURDi:
  case ARM64::LDURQi:
  case ARM64::LDURHHi:
  case ARM64::LDURBBi:
  case ARM64::LDURSBXi:
  case ARM64::LDURSBWi:
  case ARM64::LDURSHXi:
  case ARM64::LDURSHWi:
  case ARM64::LDURSWi:
  case ARM64::STURXi:
  case ARM64::STURWi:
  case ARM64::STURBi:
  case ARM64::STURHi:
  case ARM64::STURSi:
  case ARM64::STURDi:
  case ARM64::STURQi:
  case ARM64::STURBBi:
  case ARM64::STURHHi:
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
  return ARM64FrameOffsetCanUpdate |
         (Offset == 0 ? ARM64FrameOffsetIsLegal : 0);
}

bool llvm::rewriteARM64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                  unsigned FrameReg, int &Offset,
                                  const ARM64InstrInfo *TII) {
  unsigned Opcode = MI.getOpcode();
  unsigned ImmIdx = FrameRegIdx + 1;

  if (Opcode == ARM64::ADDSXri || Opcode == ARM64::ADDXri) {
    Offset += MI.getOperand(ImmIdx).getImm();
    emitFrameOffset(*MI.getParent(), MI, MI.getDebugLoc(),
                    MI.getOperand(0).getReg(), FrameReg, Offset, TII,
                    MachineInstr::NoFlags, (Opcode == ARM64::ADDSXri));
    MI.eraseFromParent();
    Offset = 0;
    return true;
  }

  int NewOffset;
  unsigned UnscaledOp;
  bool UseUnscaledOp;
  int Status = isARM64FrameOffsetLegal(MI, Offset, &UseUnscaledOp, &UnscaledOp,
                                       &NewOffset);
  if (Status & ARM64FrameOffsetCanUpdate) {
    if (Status & ARM64FrameOffsetIsLegal)
      // Replace the FrameIndex with FrameReg.
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
    if (UseUnscaledOp)
      MI.setDesc(TII->get(UnscaledOp));

    MI.getOperand(ImmIdx).ChangeToImmediate(NewOffset);
    return Offset == 0;
  }

  return false;
}

void ARM64InstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(ARM64::HINT);
  NopInst.addOperand(MCOperand::CreateImm(0));
}
