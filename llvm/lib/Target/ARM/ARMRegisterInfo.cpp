//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

ARMRegisterInfo::ARMRegisterInfo(const TargetInstrInfo &tii,
                                 const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void ARMRegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator &MBBI,
                                        const TargetInstrInfo *TII, DebugLoc dl,
                                        unsigned DestReg, int Val,
                                        ARMCC::CondCodes Pred,
                                        unsigned PredReg) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C = ConstantInt::get(Type::Int32Ty, Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII->get(ARM::LDRcp), DestReg)
    .addConstantPoolIndex(Idx)
    .addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
}

bool
ARMRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
// not required, we reserve argument space for call sites in the function
// immediately on entry to the current function. This eliminates the need for
// add/sub sp brackets around call sites. Returns true if the call frame is
// included as part of the stack frame.
bool ARMRegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (CFSize >= ((1 << 12) - 1) / 2)  // Half of imm12
    return false;

  return !MF.getFrameInfo()->hasVarSizedObjects();
}

/// emitARMRegPlusImmediate - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in ARM code.
static
void emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI,
                             unsigned DestReg, unsigned BaseReg, int NumBytes,
                             ARMCC::CondCodes Pred, unsigned PredReg,
                             const TargetInstrInfo &TII,
                             DebugLoc dl) {
  bool isSub = NumBytes < 0;
  if (isSub) NumBytes = -NumBytes;

  while (NumBytes) {
    unsigned RotAmt = ARM_AM::getSOImmValRotate(NumBytes);
    unsigned ThisVal = NumBytes & ARM_AM::rotr32(0xFF, RotAmt);
    assert(ThisVal && "Didn't extract field correctly");

    // We will handle these bits from offset, clear them.
    NumBytes &= ~ThisVal;

    // Get the properly encoded SOImmVal field.
    int SOImmVal = ARM_AM::getSOImmVal(ThisVal);
    assert(SOImmVal != -1 && "Bit extraction didn't work?");

    // Build the new ADD / SUB.
    BuildMI(MBB, MBBI, dl, TII.get(isSub ? ARM::SUBri : ARM::ADDri), DestReg)
      .addReg(BaseReg, RegState::Kill).addImm(SOImmVal)
      .addImm((unsigned)Pred).addReg(PredReg).addReg(0);
    BaseReg = DestReg;
  }
}

static void
emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
             const TargetInstrInfo &TII, DebugLoc dl,
             int NumBytes,
             ARMCC::CondCodes Pred = ARMCC::AL, unsigned PredReg = 0) {
  emitARMRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes,
                          Pred, PredReg, TII, dl);
}

void ARMRegisterInfo::
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
      ARMCC::CondCodes Pred = (ARMCC::CondCodes)Old->getOperand(1).getImm();
      if (Opc == ARM::ADJCALLSTACKDOWN || Opc == ARM::tADJCALLSTACKDOWN) {
        // Note: PredReg is operand 2 for ADJCALLSTACKDOWN.
        unsigned PredReg = Old->getOperand(2).getReg();
        emitSPUpdate(MBB, I, TII, dl, -Amount, Pred, PredReg);
      } else {
        // Note: PredReg is operand 3 for ADJCALLSTACKUP.
        unsigned PredReg = Old->getOperand(3).getReg();
        assert(Opc == ARM::ADJCALLSTACKUP || Opc == ARM::tADJCALLSTACKUP);
        emitSPUpdate(MBB, I, TII, dl, Amount, Pred, PredReg);
      }
    }
  }
  MBB.erase(I);
}

/// findScratchRegister - Find a 'free' ARM register. If register scavenger
/// is not being used, R12 is available. Otherwise, try for a call-clobbered
/// register first and then a spilled callee-saved register if that fails.
static
unsigned findScratchRegister(RegScavenger *RS, const TargetRegisterClass *RC,
                             ARMFunctionInfo *AFI) {
  unsigned Reg = RS ? RS->FindUnusedReg(RC, true) : (unsigned) ARM::R12;
  assert (!AFI->isThumbFunction());
  if (Reg == 0)
    // Try a already spilled CS register.
    Reg = RS->FindUnusedReg(RC, AFI->getSpilledCSRegisters());

  return Reg;
}

void ARMRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, RegScavenger *RS) const{
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
  else if (AFI->isDPRCalleeSavedAreaFrame(FrameIndex))
    Offset -= AFI->getDPRCalleeSavedAreaOffset();
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
  bool isSub = false;

  // Memory operands in inline assembly always use AddrMode2.
  if (Opcode == ARM::INLINEASM)
    AddrMode = ARMII::AddrMode2;

  if (Opcode == ARM::ADDri) {
    Offset += MI.getOperand(i+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::MOVr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setDesc(TII.get(ARM::SUBri));
    }

    // Common case: small offset, fits into instruction.
    int ImmedOffset = ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(ImmedOffset);
      return;
    }

    // Otherwise, we fallback to common code below to form the imm offset with
    // a sequence of ADDri instructions.  First though, pull as much of the imm
    // into this ADDri as possible.
    unsigned RotAmt = ARM_AM::getSOImmValRotate(Offset);
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xFF, RotAmt);

    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;

    // Get the properly encoded SOImmVal field.
    int ThisSOImmVal = ARM_AM::getSOImmVal(ThisImmVal);
    assert(ThisSOImmVal != -1 && "Bit extraction didn't work?");
    MI.getOperand(i+1).ChangeToImmediate(ThisSOImmVal);
  } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrMode2: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM2Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM2Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 12;
      break;
    }
    case ARMII::AddrMode3: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM3Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM3Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      break;
    }
    case ARMII::AddrMode5: {
      ImmIdx = i+1;
      InstrOffs = ARM_AM::getAM5Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM5Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      Scale = 4;
      break;
    }
    default:
      assert(0 && "Unsupported addressing mode!");
      abort();
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale-1)) == 0 && "Can't encode this offset!");
    if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
    }

    // Common case: small offset, fits into instruction.
    MachineOperand &ImmOp = MI.getOperand(ImmIdx);
    int ImmedOffset = Offset / Scale;
    unsigned Mask = (1 << NumBits) - 1;
    if ((unsigned)Offset <= Mask * Scale) {
      // Replace the FrameIndex with sp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      return;
    }

    // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
    ImmedOffset = ImmedOffset & Mask;
    if (isSub)
      ImmedOffset |= 1 << NumBits;
    ImmOp.ChangeToImmediate(ImmedOffset);
    Offset &= ~(Mask*Scale);
  }

  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert(Offset && "This code isn't needed if offset already handled!");

  // Insert a set of r12 with the full address: r12 = sp + offset
  // If the offset we have is too large to fit into the instruction, we need
  // to form it with a series of ADDri's.  Do this by taking 8-bit chunks
  // out of 'Offset'.
  unsigned ScratchReg = findScratchRegister(RS, &ARM::GPRRegClass, AFI);
  if (ScratchReg == 0)
    // No register is "free". Scavenge a register.
    ScratchReg = RS->scavengeRegister(&ARM::GPRRegClass, II, SPAdj);
  int PIdx = MI.findFirstPredOperandIdx();
  ARMCC::CondCodes Pred = (PIdx == -1)
    ? ARMCC::AL : (ARMCC::CondCodes)MI.getOperand(PIdx).getImm();
  unsigned PredReg = (PIdx == -1) ? 0 : MI.getOperand(PIdx+1).getReg();
  emitARMRegPlusImmediate(MBB, II, ScratchReg, FrameReg,
                          isSub ? -Offset : Offset, Pred, PredReg, TII, dl);
  MI.getOperand(i).ChangeToRegister(ScratchReg, false, false, true);
}

/// Move iterator pass the next bunch of callee save load / store ops for
/// the particular spill area (1: integer area 1, 2: integer area 2,
/// 3: fp area, 0: don't care).
static void movePastCSLoadStoreOps(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   int Opc, unsigned Area,
                                   const ARMSubtarget &STI) {
  while (MBBI != MBB.end() &&
         MBBI->getOpcode() == Opc && MBBI->getOperand(1).isFI()) {
    if (Area != 0) {
      bool Done = false;
      unsigned Category = 0;
      switch (MBBI->getOperand(0).getReg()) {
      case ARM::R4:  case ARM::R5:  case ARM::R6: case ARM::R7:
      case ARM::LR:
        Category = 1;
        break;
      case ARM::R8:  case ARM::R9:  case ARM::R10: case ARM::R11:
        Category = STI.isTargetDarwin() ? 2 : 1;
        break;
      case ARM::D8:  case ARM::D9:  case ARM::D10: case ARM::D11:
      case ARM::D12: case ARM::D13: case ARM::D14: case ARM::D15:
        Category = 3;
        break;
      default:
        Done = true;
        break;
      }
      if (Done || Category != Area)
        break;
    }

    ++MBBI;
  }
}

void ARMRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = (MBBI != MBB.end() ?
                 MBBI->getDebugLoc() : DebugLoc::getUnknownLoc());

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  if (VARegSaveSize)
    emitSPUpdate(MBB, MBBI, TII, dl, -VARegSaveSize);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, -NumBytes);
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

  // Build the new SUBri to adjust SP for integer callee-save spill area 1.
  emitSPUpdate(MBB, MBBI, TII, dl, -GPRCS1Size);
  movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 1, STI);

  // Darwin ABI requires FP to point to the stack slot that contains the
  // previous FP.
  if (STI.isTargetDarwin() || hasFP(MF)) {
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(ARM::ADDri), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);
    AddDefaultCC(AddDefaultPred(MIB));
  }

  // Build the new SUBri to adjust SP for integer callee-save spill area 2.
  emitSPUpdate(MBB, MBBI, TII, dl, -GPRCS2Size);

  // Build the new SUBri to adjust SP for FP callee-save spill area.
  movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 2, STI);
  emitSPUpdate(MBB, MBBI, TII, dl, -DPRCSSize);

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
    movePastCSLoadStoreOps(MBB, MBBI, ARM::FSTD, 3, STI);
    emitSPUpdate(MBB, MBBI, TII, dl, -NumBytes);
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
  return ((MI->getOpcode() == ARM::FLDD ||
           MI->getOpcode() == ARM::LDR) &&
          MI->getOperand(1).isFI() &&
          isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs));
}

void ARMRegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == ARM::BX_RET &&
         "Can only insert epilog into returning blocks");
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, NumBytes);
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

    // Darwin ABI requires FP to point to the stack slot that contains the
    // previous FP.
    if ((STI.isTargetDarwin() && NumBytes) || hasFP(MF)) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      // Reset SP based on frame pointer only if the stack frame extends beyond
      // frame pointer stack slot or target is ELF and the function has FP.
      if (AFI->getGPRCalleeSavedArea2Size() ||
          AFI->getDPRCalleeSavedAreaSize()  ||
          AFI->getDPRCalleeSavedAreaOffset()||
          hasFP(MF)) {
        if (NumBytes)
          BuildMI(MBB, MBBI, dl, TII.get(ARM::SUBri), ARM::SP).addReg(FramePtr)
            .addImm(NumBytes)
            .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
        else
          BuildMI(MBB, MBBI, dl, TII.get(ARM::MOVr), ARM::SP).addReg(FramePtr)
            .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
      }
    } else if (NumBytes) {
      emitSPUpdate(MBB, MBBI, TII, dl, NumBytes);
    }

    // Move SP to start of integer callee save spill area 2.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::FLDD, 3, STI);
    emitSPUpdate(MBB, MBBI, TII, dl, AFI->getDPRCalleeSavedAreaSize());

    // Move SP to start of integer callee save spill area 1.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 2, STI);
    emitSPUpdate(MBB, MBBI, TII, dl, AFI->getGPRCalleeSavedArea2Size());

    // Move SP to SP upon entry to the function.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 1, STI);
    emitSPUpdate(MBB, MBBI, TII, dl, AFI->getGPRCalleeSavedArea1Size());
  }

  if (VARegSaveSize)
    emitSPUpdate(MBB, MBBI, TII, dl, VARegSaveSize);

}

