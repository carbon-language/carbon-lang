//===- Thumb1RegisterInfo.cpp - Thumb-1 Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "Thumb1InstrInfo.h"
#include "Thumb1RegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

Thumb1RegisterInfo::Thumb1RegisterInfo(const ARMBaseInstrInfo &tii,
                                       const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void Thumb1RegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator &MBBI,
                                           DebugLoc dl,
                                           unsigned DestReg, unsigned SubIdx,
                                           int Val,
                                           ARMCC::CondCodes Pred,
                                           unsigned PredReg) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C = ConstantInt::get(
          Type::getInt32Ty(MBB.getParent()->getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::tLDRcp))
          .addReg(DestReg, getDefRegState(true), SubIdx)
          .addConstantPoolIndex(Idx).addImm(Pred).addReg(PredReg);
}

const TargetRegisterClass*
Thumb1RegisterInfo::getPhysicalRegisterRegClass(unsigned Reg, EVT VT) const {
  if (isARMLowRegister(Reg))
    return ARM::tGPRRegisterClass;
  switch (Reg) {
   default:
    break;
   case ARM::R8:  case ARM::R9:  case ARM::R10:  case ARM::R11:
   case ARM::R12: case ARM::SP:  case ARM::LR:   case ARM::PC:
    return ARM::GPRRegisterClass;
  }

  return TargetRegisterInfo::getPhysicalRegisterRegClass(Reg, VT);
}

bool
Thumb1RegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

bool
Thumb1RegisterInfo::requiresFrameIndexScavenging(const MachineFunction &MF)
  const {
  return true;
}


bool Thumb1RegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (CFSize >= ((1 << 8) - 1) * 4 / 2) // Half of imm8 * 4
    return false;

  return !MF.getFrameInfo()->hasVarSizedObjects();
}


/// emitThumbRegPlusImmInReg - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code. Materialize the immediate
/// in a register using mov / mvn sequences or load the immediate from a
/// constpool entry.
static
void emitThumbRegPlusImmInReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned DestReg, unsigned BaseReg,
                              int NumBytes, bool CanChangeCC,
                              const TargetInstrInfo &TII,
                              const Thumb1RegisterInfo& MRI,
                              DebugLoc dl) {
    MachineFunction &MF = *MBB.getParent();
    bool isHigh = !isARMLowRegister(DestReg) ||
                  (BaseReg != 0 && !isARMLowRegister(BaseReg));
    bool isSub = false;
    // Subtract doesn't have high register version. Load the negative value
    // if either base or dest register is a high register. Also, if do not
    // issue sub as part of the sequence if condition register is to be
    // preserved.
    if (NumBytes < 0 && !isHigh && CanChangeCC) {
      isSub = true;
      NumBytes = -NumBytes;
    }
    unsigned LdReg = DestReg;
    if (DestReg == ARM::SP) {
      assert(BaseReg == ARM::SP && "Unexpected!");
      LdReg = MF.getRegInfo().createVirtualRegister(ARM::tGPRRegisterClass);
    }

    if (NumBytes <= 255 && NumBytes >= 0)
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8), LdReg))
        .addImm(NumBytes);
    else if (NumBytes < 0 && NumBytes >= -255) {
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8), LdReg))
        .addImm(NumBytes);
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tRSB), LdReg))
        .addReg(LdReg, RegState::Kill);
    } else
      MRI.emitLoadConstPool(MBB, MBBI, dl, LdReg, 0, NumBytes);

    // Emit add / sub.
    int Opc = (isSub) ? ARM::tSUBrr : (isHigh ? ARM::tADDhirr : ARM::tADDrr);
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
    if (Opc != ARM::tADDhirr)
      MIB = AddDefaultT1CC(MIB);
    if (DestReg == ARM::SP || isSub)
      MIB.addReg(BaseReg).addReg(LdReg, RegState::Kill);
    else
      MIB.addReg(LdReg).addReg(BaseReg, RegState::Kill);
    AddDefaultPred(MIB);
}

/// calcNumMI - Returns the number of instructions required to materialize
/// the specific add / sub r, c instruction.
static unsigned calcNumMI(int Opc, int ExtraOpc, unsigned Bytes,
                          unsigned NumBits, unsigned Scale) {
  unsigned NumMIs = 0;
  unsigned Chunk = ((1 << NumBits) - 1) * Scale;

  if (Opc == ARM::tADDrSPi) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;
    NumMIs++;
    NumBits = 8;
    Scale = 1;  // Followed by a number of tADDi8.
    Chunk = ((1 << NumBits) - 1) * Scale;
  }

  NumMIs += Bytes / Chunk;
  if ((Bytes % Chunk) != 0)
    NumMIs++;
  if (ExtraOpc)
    NumMIs++;
  return NumMIs;
}

/// emitThumbRegPlusImmediate - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code.
static
void emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI,
                               unsigned DestReg, unsigned BaseReg,
                               int NumBytes, const TargetInstrInfo &TII,
                               const Thumb1RegisterInfo& MRI,
                               DebugLoc dl) {
  bool isSub = NumBytes < 0;
  unsigned Bytes = (unsigned)NumBytes;
  if (isSub) Bytes = -NumBytes;
  bool isMul4 = (Bytes & 3) == 0;
  bool isTwoAddr = false;
  bool DstNotEqBase = false;
  unsigned NumBits = 1;
  unsigned Scale = 1;
  int Opc = 0;
  int ExtraOpc = 0;
  bool NeedCC = false;
  bool NeedPred = false;

  if (DestReg == BaseReg && BaseReg == ARM::SP) {
    assert(isMul4 && "Thumb sp inc / dec size must be multiple of 4!");
    NumBits = 7;
    Scale = 4;
    Opc = isSub ? ARM::tSUBspi : ARM::tADDspi;
    isTwoAddr = true;
  } else if (!isSub && BaseReg == ARM::SP) {
    // r1 = add sp, 403
    // =>
    // r1 = add sp, 100 * 4
    // r1 = add r1, 3
    if (!isMul4) {
      Bytes &= ~3;
      ExtraOpc = ARM::tADDi3;
    }
    NumBits = 8;
    Scale = 4;
    Opc = ARM::tADDrSPi;
  } else {
    // sp = sub sp, c
    // r1 = sub sp, c
    // r8 = sub sp, c
    if (DestReg != BaseReg)
      DstNotEqBase = true;
    NumBits = 8;
    if (DestReg == ARM::SP) {
      Opc = isSub ? ARM::tSUBspi : ARM::tADDspi;
      assert(isMul4 && "Thumb sp inc / dec size must be multiple of 4!");
      NumBits = 7;
      Scale = 4;
    } else {
      Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
      NumBits = 8;
      NeedPred = NeedCC = true;
    }
    isTwoAddr = true;
  }

  unsigned NumMIs = calcNumMI(Opc, ExtraOpc, Bytes, NumBits, Scale);
  unsigned Threshold = (DestReg == ARM::SP) ? 3 : 2;
  if (NumMIs > Threshold) {
    // This will expand into too many instructions. Load the immediate from a
    // constpool entry.
    emitThumbRegPlusImmInReg(MBB, MBBI, DestReg, BaseReg, NumBytes, true, TII,
                             MRI, dl);
    return;
  }

  if (DstNotEqBase) {
    if (isARMLowRegister(DestReg) && isARMLowRegister(BaseReg)) {
      // If both are low registers, emit DestReg = add BaseReg, max(Imm, 7)
      unsigned Chunk = (1 << 3) - 1;
      unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
      Bytes -= ThisVal;
      const TargetInstrDesc &TID = TII.get(isSub ? ARM::tSUBi3 : ARM::tADDi3);
      const MachineInstrBuilder MIB =
        AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TID, DestReg));
      AddDefaultPred(MIB.addReg(BaseReg, RegState::Kill).addImm(ThisVal));
    } else {
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr), DestReg)
        .addReg(BaseReg, RegState::Kill);
    }
    BaseReg = DestReg;
  }

  unsigned Chunk = ((1 << NumBits) - 1) * Scale;
  while (Bytes) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;
    ThisVal /= Scale;
    // Build the new tADD / tSUB.
    if (isTwoAddr) {
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
      if (NeedCC)
        MIB = AddDefaultT1CC(MIB);
      MIB .addReg(DestReg).addImm(ThisVal);
      if (NeedPred)
        MIB = AddDefaultPred(MIB);
    }
    else {
      bool isKill = BaseReg != ARM::SP;
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
      if (NeedCC)
        MIB = AddDefaultT1CC(MIB);
      MIB.addReg(BaseReg, getKillRegState(isKill)).addImm(ThisVal);
      if (NeedPred)
        MIB = AddDefaultPred(MIB);
      BaseReg = DestReg;

      if (Opc == ARM::tADDrSPi) {
        // r4 = add sp, imm
        // r4 = add r4, imm
        // ...
        NumBits = 8;
        Scale = 1;
        Chunk = ((1 << NumBits) - 1) * Scale;
        Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
        NeedPred = NeedCC = isTwoAddr = true;
      }
    }
  }

  if (ExtraOpc) {
    const TargetInstrDesc &TID = TII.get(ExtraOpc);
    AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TID, DestReg))
                   .addReg(DestReg, RegState::Kill)
                   .addImm(((unsigned)NumBytes) & 3));
  }
}

static void emitSPUpdate(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         const TargetInstrInfo &TII, DebugLoc dl,
                         const Thumb1RegisterInfo &MRI,
                         int NumBytes) {
  emitThumbRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes, TII,
                            MRI, dl);
}

void Thumb1RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If we have alloca, convert as follows:
    // ADJCALLSTACKDOWN -> sub, sp, sp, amount
    // ADJCALLSTACKUP   -> add, sp, sp, amount
    MachineInstr *Old = I;
    DebugLoc dl = Old->getDebugLoc();
    unsigned Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      unsigned Opc = Old->getOpcode();
      if (Opc == ARM::ADJCALLSTACKDOWN || Opc == ARM::tADJCALLSTACKDOWN) {
        emitSPUpdate(MBB, I, TII, dl, *this, -Amount);
      } else {
        assert(Opc == ARM::ADJCALLSTACKUP || Opc == ARM::tADJCALLSTACKUP);
        emitSPUpdate(MBB, I, TII, dl, *this, Amount);
      }
    }
  }
  MBB.erase(I);
}

/// emitThumbConstant - Emit a series of instructions to materialize a
/// constant.
static void emitThumbConstant(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned DestReg, int Imm,
                              const TargetInstrInfo &TII,
                              const Thumb1RegisterInfo& MRI,
                              DebugLoc dl) {
  bool isSub = Imm < 0;
  if (isSub) Imm = -Imm;

  int Chunk = (1 << 8) - 1;
  int ThisVal = (Imm > Chunk) ? Chunk : Imm;
  Imm -= ThisVal;
  AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8),
                                        DestReg))
                 .addImm(ThisVal));
  if (Imm > 0)
    emitThumbRegPlusImmediate(MBB, MBBI, DestReg, DestReg, Imm, TII, MRI, dl);
  if (isSub) {
    const TargetInstrDesc &TID = TII.get(ARM::tRSB);
    AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TID, DestReg))
                   .addReg(DestReg, RegState::Kill));
  }
}

static void removeOperands(MachineInstr &MI, unsigned i) {
  unsigned Op = i;
  for (unsigned e = MI.getNumOperands(); i != e; ++i)
    MI.RemoveOperand(Op);
}

int Thumb1RegisterInfo::
rewriteFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                  unsigned FrameReg, int Offset,
                  unsigned MOVOpc, unsigned ADDriOpc, unsigned SUBriOpc) const
{
  // if/when eliminateFrameIndex() conforms with ARMBaseRegisterInfo
  // version then can pull out Thumb1 specific parts here
  return 0;
}

/// saveScavengerRegister - Spill the register so it can be used by the
/// register scavenger. Return true.
bool
Thumb1RegisterInfo::saveScavengerRegister(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          MachineBasicBlock::iterator &UseMI,
                                          const TargetRegisterClass *RC,
                                          unsigned Reg) const {
  // Thumb1 can't use the emergency spill slot on the stack because
  // ldr/str immediate offsets must be positive, and if we're referencing
  // off the frame pointer (if, for example, there are alloca() calls in
  // the function, the offset will be negative. Use R12 instead since that's
  // a call clobbered register that we know won't be used in Thumb1 mode.
  DebugLoc DL = DebugLoc::getUnknownLoc();
  BuildMI(MBB, I, DL, TII.get(ARM::tMOVtgpr2gpr)).
    addReg(ARM::R12, RegState::Define).addReg(Reg, RegState::Kill);

  // The UseMI is where we would like to restore the register. If there's
  // interference with R12 before then, however, we'll need to restore it
  // before that instead and adjust the UseMI.
  bool done = false;
  for (MachineBasicBlock::iterator II = I; !done && II != UseMI ; ++II) {
    // If this instruction affects R12, adjust our restore point.
    for (unsigned i = 0, e = II->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = II->getOperand(i);
      if (!MO.isReg() || MO.isUndef() || !MO.getReg() ||
          TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;
      if (MO.getReg() == ARM::R12) {
        UseMI = II;
        done = true;
        break;
      }
    }
  }
  // Restore the register from R12
  BuildMI(MBB, UseMI, DL, TII.get(ARM::tMOVgpr2tgpr)).
    addReg(Reg, RegState::Define).addReg(ARM::R12, RegState::Kill);

  return true;
}

unsigned
Thumb1RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, int *Value,
                                        RegScavenger *RS) const{
  unsigned VReg = 0;
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  DebugLoc dl = MI.getDebugLoc();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  unsigned FrameReg = ARM::SP;
  int FrameIndex = MI.getOperand(i).getIndex();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MF.getFrameInfo()->getStackSize() + SPAdj;

  if (AFI->isGPRCalleeSavedArea1Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea1Offset();
  else if (AFI->isGPRCalleeSavedArea2Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea2Offset();
  else if (hasFP(MF)) {
    assert(SPAdj == 0 && "Unexpected");
    // There is alloca()'s in this function, must reference off the frame
    // pointer instead.
    FrameReg = getFrameRegister(MF);
    Offset -= AFI->getFramePtrSpillOffset();
  }

  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);

  if (Opcode == ARM::tADDrSPi) {
    Offset += MI.getOperand(i+1).getImm();

    // Can't use tADDrSPi if it's based off the frame pointer.
    unsigned NumBits = 0;
    unsigned Scale = 1;
    if (FrameReg != ARM::SP) {
      Opcode = ARM::tADDi3;
      MI.setDesc(TII.get(Opcode));
      NumBits = 3;
    } else {
      NumBits = 8;
      Scale = 4;
      assert((Offset & 3) == 0 &&
             "Thumb add/sub sp, #imm immediate must be multiple of 4!");
    }

    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::tMOVgpr2tgpr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return 0;
    }

    // Common case: small offset, fits into instruction.
    unsigned Mask = (1 << NumBits) - 1;
    if (((Offset / Scale) & ~Mask) == 0) {
      // Replace the FrameIndex with sp / fp
      if (Opcode == ARM::tADDi3) {
        removeOperands(MI, i);
        MachineInstrBuilder MIB(&MI);
        AddDefaultPred(AddDefaultT1CC(MIB).addReg(FrameReg)
                       .addImm(Offset / Scale));
      } else {
        MI.getOperand(i).ChangeToRegister(FrameReg, false);
        MI.getOperand(i+1).ChangeToImmediate(Offset / Scale);
      }
      return 0;
    }

    unsigned DestReg = MI.getOperand(0).getReg();
    unsigned Bytes = (Offset > 0) ? Offset : -Offset;
    unsigned NumMIs = calcNumMI(Opcode, 0, Bytes, NumBits, Scale);
    // MI would expand into a large number of instructions. Don't try to
    // simplify the immediate.
    if (NumMIs > 2) {
      emitThumbRegPlusImmediate(MBB, II, DestReg, FrameReg, Offset, TII,
                                *this, dl);
      MBB.erase(II);
      return 0;
    }

    if (Offset > 0) {
      // Translate r0 = add sp, imm to
      // r0 = add sp, 255*4
      // r0 = add r0, (imm - 255*4)
      if (Opcode == ARM::tADDi3) {
        removeOperands(MI, i);
        MachineInstrBuilder MIB(&MI);
        AddDefaultPred(AddDefaultT1CC(MIB).addReg(FrameReg).addImm(Mask));
      } else {
        MI.getOperand(i).ChangeToRegister(FrameReg, false);
        MI.getOperand(i+1).ChangeToImmediate(Mask);
      }
      Offset = (Offset - Mask * Scale);
      MachineBasicBlock::iterator NII = next(II);
      emitThumbRegPlusImmediate(MBB, NII, DestReg, DestReg, Offset, TII,
                                *this, dl);
    } else {
      // Translate r0 = add sp, -imm to
      // r0 = -imm (this is then translated into a series of instructons)
      // r0 = add r0, sp
      emitThumbConstant(MBB, II, DestReg, Offset, TII, *this, dl);

      MI.setDesc(TII.get(ARM::tADDhirr));
      MI.getOperand(i).ChangeToRegister(DestReg, false, false, true);
      MI.getOperand(i+1).ChangeToRegister(FrameReg, false);
      if (Opcode == ARM::tADDi3) {
        MachineInstrBuilder MIB(&MI);
        AddDefaultPred(MIB);
      }
    }
    return 0;
  } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrModeT1_s: {
      ImmIdx = i+1;
      InstrOffs = MI.getOperand(ImmIdx).getImm();
      NumBits = (FrameReg == ARM::SP) ? 8 : 5;
      Scale = 4;
      break;
    }
    default:
      llvm_unreachable("Unsupported addressing mode!");
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale-1)) == 0 && "Can't encode this offset!");

    // Common case: small offset, fits into instruction.
    MachineOperand &ImmOp = MI.getOperand(ImmIdx);
    int ImmedOffset = Offset / Scale;
    unsigned Mask = (1 << NumBits) - 1;
    if ((unsigned)Offset <= Mask * Scale) {
      // Replace the FrameIndex with sp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      ImmOp.ChangeToImmediate(ImmedOffset);
      return 0;
    }

    bool isThumSpillRestore = Opcode == ARM::tRestore || Opcode == ARM::tSpill;
    if (AddrMode == ARMII::AddrModeT1_s) {
      // Thumb tLDRspi, tSTRspi. These will change to instructions that use
      // a different base register.
      NumBits = 5;
      Mask = (1 << NumBits) - 1;
    }
    // If this is a thumb spill / restore, we will be using a constpool load to
    // materialize the offset.
    if (AddrMode == ARMII::AddrModeT1_s && isThumSpillRestore)
      ImmOp.ChangeToImmediate(0);
    else {
      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask*Scale);
    }
  }

  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert(Offset && "This code isn't needed if offset already handled!");

  // Remove predicate first.
  int PIdx = MI.findFirstPredOperandIdx();
  if (PIdx != -1)
    removeOperands(MI, PIdx);

  if (Desc.mayLoad()) {
    // Use the destination register to materialize sp + offset.
    unsigned TmpReg = MI.getOperand(0).getReg();
    bool UseRR = false;
    if (Opcode == ARM::tRestore) {
      if (FrameReg == ARM::SP)
        emitThumbRegPlusImmInReg(MBB, II, TmpReg, FrameReg,
                                 Offset, false, TII, *this, dl);
      else {
        emitLoadConstPool(MBB, II, dl, TmpReg, 0, Offset);
        UseRR = true;
      }
    } else {
      emitThumbRegPlusImmediate(MBB, II, TmpReg, FrameReg, Offset, TII,
                                *this, dl);
    }

    MI.setDesc(TII.get(ARM::tLDR));
    MI.getOperand(i).ChangeToRegister(TmpReg, false, false, true);
    if (UseRR)
      // Use [reg, reg] addrmode.
      MI.addOperand(MachineOperand::CreateReg(FrameReg, false));
    else  // tLDR has an extra register operand.
      MI.addOperand(MachineOperand::CreateReg(0, false));
  } else if (Desc.mayStore()) {
      VReg = MF.getRegInfo().createVirtualRegister(ARM::tGPRRegisterClass);
      assert (Value && "Frame index virtual allocated, but Value arg is NULL!");
      *Value = Offset;
      bool UseRR = false;

      if (Opcode == ARM::tSpill) {
        if (FrameReg == ARM::SP)
          emitThumbRegPlusImmInReg(MBB, II, VReg, FrameReg,
                                   Offset, false, TII, *this, dl);
        else {
          emitLoadConstPool(MBB, II, dl, VReg, 0, Offset);
          UseRR = true;
        }
      } else
        emitThumbRegPlusImmediate(MBB, II, VReg, FrameReg, Offset, TII,
                                  *this, dl);
      MI.setDesc(TII.get(ARM::tSTR));
      MI.getOperand(i).ChangeToRegister(VReg, false, false, true);
      if (UseRR)  // Use [reg, reg] addrmode.
        MI.addOperand(MachineOperand::CreateReg(FrameReg, false));
      else // tSTR has an extra register operand.
        MI.addOperand(MachineOperand::CreateReg(0, false));
  } else
    assert(false && "Unexpected opcode!");

  // Add predicate back if it's needed.
  if (MI.getDesc().isPredicable()) {
    MachineInstrBuilder MIB(&MI);
    AddDefaultPred(MIB);
  }
  return VReg;
}

void Thumb1RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = (MBBI != MBB.end() ?
                 MBBI->getDebugLoc() : DebugLoc::getUnknownLoc());

  // Thumb add/sub sp, imm8 instructions implicitly multiply the offset by 4.
  NumBytes = (NumBytes + 3) & ~3;
  MFI->setStackSize(NumBytes);

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  if (VARegSaveSize)
    emitSPUpdate(MBB, MBBI, TII, dl, *this, -VARegSaveSize);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, *this, -NumBytes);
    return;
  }

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    int FI = CSI[i].getFrameIdx();
    switch (Reg) {
    case ARM::R4:
    case ARM::R5:
    case ARM::R6:
    case ARM::R7:
    case ARM::LR:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      AFI->addGPRCalleeSavedArea1Frame(FI);
      GPRCS1Size += 4;
      break;
    case ARM::R8:
    case ARM::R9:
    case ARM::R10:
    case ARM::R11:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      if (STI.isTargetDarwin()) {
        AFI->addGPRCalleeSavedArea2Frame(FI);
        GPRCS2Size += 4;
      } else {
        AFI->addGPRCalleeSavedArea1Frame(FI);
        GPRCS1Size += 4;
      }
      break;
    default:
      AFI->addDPRCalleeSavedAreaFrame(FI);
      DPRCSSize += 8;
    }
  }

  if (MBBI != MBB.end() && MBBI->getOpcode() == ARM::tPUSH) {
    ++MBBI;
    if (MBBI != MBB.end())
      dl = MBBI->getDebugLoc();
  }

  // Darwin ABI requires FP to point to the stack slot that contains the
  // previous FP.
  if (STI.isTargetDarwin() || hasFP(MF)) {
    BuildMI(MBB, MBBI, dl, TII.get(ARM::tADDrSPi), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);
  }

  // Determine starting offsets of spill areas.
  unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
  unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
  unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
  AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) + NumBytes);
  AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
  AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
  AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);

  NumBytes = DPRCSOffset;
  if (NumBytes) {
    // Insert it after all the callee-save spills.
    emitSPUpdate(MBB, MBBI, TII, dl, *this, -NumBytes);
  }

  if (STI.isTargetELF() && hasFP(MF)) {
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());
  }

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);
}

static bool isCalleeSavedRegister(unsigned Reg, const unsigned *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI, const unsigned *CSRegs) {
  return (MI->getOpcode() == ARM::tRestore &&
          MI->getOperand(1).isFI() &&
          isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs));
}

void Thumb1RegisterInfo::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert((MBBI->getOpcode() == ARM::tBX_RET ||
          MBBI->getOpcode() == ARM::tPOP_RET) &&
         "Can only insert epilog into returning blocks");
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, *this, NumBytes);
  } else {
    // Unwind MBBI to point to first LDR / FLDD.
    const unsigned *CSRegs = getCalleeSavedRegs();
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, CSRegs));
      if (!isCSRestore(MBBI, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());

    if (hasFP(MF)) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      // Reset SP based on frame pointer only if the stack frame extends beyond
      // frame pointer stack slot or target is ELF and the function has FP.
      if (NumBytes)
        emitThumbRegPlusImmediate(MBB, MBBI, ARM::SP, FramePtr, -NumBytes,
                                  TII, *this, dl);
      else
        BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVtgpr2gpr), ARM::SP)
          .addReg(FramePtr);
    } else {
      if (MBBI->getOpcode() == ARM::tBX_RET &&
          &MBB.front() != MBBI &&
          prior(MBBI)->getOpcode() == ARM::tPOP) {
        MachineBasicBlock::iterator PMBBI = prior(MBBI);
        emitSPUpdate(MBB, PMBBI, TII, dl, *this, NumBytes);
      } else
        emitSPUpdate(MBB, MBBI, TII, dl, *this, NumBytes);
    }
  }

  if (VARegSaveSize) {
    // Epilogue for vararg functions: pop LR to R3 and branch off it.
    // FIXME: Verify this is still ok when R3 is no longer being reserved.
    AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tPOP)))
      .addReg(0) // No write back.
      .addReg(ARM::R3, RegState::Define);

    emitSPUpdate(MBB, MBBI, TII, dl, *this, VARegSaveSize);

    BuildMI(MBB, MBBI, dl, TII.get(ARM::tBX_RET_vararg))
      .addReg(ARM::R3, RegState::Kill);
    MBB.erase(MBBI);
  }
}
