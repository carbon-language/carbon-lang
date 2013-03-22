//===- AArch64FrameLowering.cpp - AArch64 Frame Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64FrameLowering.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64InstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void AArch64FrameLowering::splitSPAdjustments(uint64_t Total,
                                              uint64_t &Initial,
                                              uint64_t &Residual) const {
  // 0x1f0 here is a pessimistic (i.e. realistic) boundary: x-register LDP
  // instructions have a 7-bit signed immediate scaled by 8, giving a reach of
  // 0x1f8, but stack adjustment should always be a multiple of 16.
  if (Total <= 0x1f0) {
    Initial = Total;
    Residual = 0;
  } else {
    Initial = 0x1f0;
    Residual = Total - Initial;
  }
}

void AArch64FrameLowering::emitPrologue(MachineFunction &MF) const {
  AArch64MachineFunctionInfo *FuncInfo =
    MF.getInfo<AArch64MachineFunctionInfo>();
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  MachineModuleInfo &MMI = MF.getMMI();
  std::vector<MachineMove> &Moves = MMI.getFrameMoves();
  bool NeedsFrameMoves = MMI.hasDebugInfo()
    || MF.getFunction()->needsUnwindTableEntry();

  uint64_t NumInitialBytes, NumResidualBytes;

  // Currently we expect the stack to be laid out by
  //     sub sp, sp, #initial
  //     stp x29, x30, [sp, #offset]
  //     ...
  //     str xxx, [sp, #offset]
  //     sub sp, sp, #rest (possibly via extra instructions).
  if (MFI->getCalleeSavedInfo().size()) {
    // If there are callee-saved registers, we want to store them efficiently as
    // a block, and virtual base assignment happens too early to do it for us so
    // we adjust the stack in two phases: first just for callee-saved fiddling,
    // then to allocate the rest of the frame.
    splitSPAdjustments(MFI->getStackSize(), NumInitialBytes, NumResidualBytes);
  } else {
    // If there aren't any callee-saved registers, two-phase adjustment is
    // inefficient. It's more efficient to adjust with NumInitialBytes too
    // because when we're in a "callee pops argument space" situation, that pop
    // must be tacked onto Initial for correctness.
    NumInitialBytes = MFI->getStackSize();
    NumResidualBytes = 0;
  }

  // Tell everyone else how much adjustment we're expecting them to use. In
  // particular if an adjustment is required for a tail call the epilogue could
  // have a different view of things.
  FuncInfo->setInitialStackAdjust(NumInitialBytes);

  emitSPUpdate(MBB, MBBI, DL, TII, AArch64::X16, -NumInitialBytes,
               MachineInstr::FrameSetup);

  if (NeedsFrameMoves && NumInitialBytes) {
    // We emit this update even if the CFA is set from a frame pointer later so
    // that the CFA is valid in the interim.
    MCSymbol *SPLabel = MMI.getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::PROLOG_LABEL))
      .addSym(SPLabel);

    MachineLocation Dst(MachineLocation::VirtualFP);
    MachineLocation Src(AArch64::XSP, NumInitialBytes);
    Moves.push_back(MachineMove(SPLabel, Dst, Src));
  }

  // Otherwise we need to set the frame pointer and/or add a second stack
  // adjustment.

  bool FPNeedsSetting = hasFP(MF);
  for (; MBBI != MBB.end(); ++MBBI) {
    // Note that this search makes strong assumptions about the operation used
    // to store the frame-pointer: it must be "STP x29, x30, ...". This could
    // change in future, but until then there's no point in implementing
    // untestable more generic cases.
    if (FPNeedsSetting && MBBI->getOpcode() == AArch64::LSPair64_STR
                       && MBBI->getOperand(0).getReg() == AArch64::X29) {
      int64_t X29FrameIdx = MBBI->getOperand(2).getIndex();
      FuncInfo->setFramePointerOffset(MFI->getObjectOffset(X29FrameIdx));

      ++MBBI;
      emitRegUpdate(MBB, MBBI, DL, TII, AArch64::X29, AArch64::XSP,
                    AArch64::X29,
                    NumInitialBytes + MFI->getObjectOffset(X29FrameIdx),
                    MachineInstr::FrameSetup);

      // The offset adjustment used when emitting debugging locations relative
      // to whatever frame base is set. AArch64 uses the default frame base (FP
      // or SP) and this adjusts the calculations to be correct.
      MFI->setOffsetAdjustment(- MFI->getObjectOffset(X29FrameIdx)
                               - MFI->getStackSize());

      if (NeedsFrameMoves) {
        MCSymbol *FPLabel = MMI.getContext().CreateTempSymbol();
        BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::PROLOG_LABEL))
          .addSym(FPLabel);
        MachineLocation Dst(MachineLocation::VirtualFP);
        MachineLocation Src(AArch64::X29, -MFI->getObjectOffset(X29FrameIdx));
        Moves.push_back(MachineMove(FPLabel, Dst, Src));
      }

      FPNeedsSetting = false;
    }

    if (!MBBI->getFlag(MachineInstr::FrameSetup))
      break;
  }

  assert(!FPNeedsSetting && "Frame pointer couldn't be set");

  emitSPUpdate(MBB, MBBI, DL, TII, AArch64::X16, -NumResidualBytes,
               MachineInstr::FrameSetup);

  // Now we emit the rest of the frame setup information, if necessary: we've
  // already noted the FP and initial SP moves so we're left with the prologue's
  // final SP update and callee-saved register locations.
  if (!NeedsFrameMoves)
    return;

  // Reuse the label if appropriate, so create it in this outer scope.
  MCSymbol *CSLabel = 0;

  // The rest of the stack adjustment
  if (!hasFP(MF) && NumResidualBytes) {
    CSLabel = MMI.getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::PROLOG_LABEL))
      .addSym(CSLabel);

    MachineLocation Dst(MachineLocation::VirtualFP);
    MachineLocation Src(AArch64::XSP, NumResidualBytes + NumInitialBytes);
    Moves.push_back(MachineMove(CSLabel, Dst, Src));
  }

  // And any callee-saved registers (it's fine to leave them to the end here,
  // because the old values are still valid at this point.
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  if (CSI.size()) {
    if (!CSLabel) {
      CSLabel = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::PROLOG_LABEL))
        .addSym(CSLabel);
    }

    for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
           E = CSI.end(); I != E; ++I) {
      MachineLocation Dst(MachineLocation::VirtualFP,
                          MFI->getObjectOffset(I->getFrameIdx()));
      MachineLocation Src(I->getReg());
      Moves.push_back(MachineMove(CSLabel, Dst, Src));
    }
  }
}

void
AArch64FrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  AArch64MachineFunctionInfo *FuncInfo =
    MF.getInfo<AArch64MachineFunctionInfo>();

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned RetOpcode = MBBI->getOpcode();

  // Initial and residual are named for consitency with the prologue. Note that
  // in the epilogue, the residual adjustment is executed first.
  uint64_t NumInitialBytes = FuncInfo->getInitialStackAdjust();
  uint64_t NumResidualBytes = MFI.getStackSize() - NumInitialBytes;
  uint64_t ArgumentPopSize = 0;
  if (RetOpcode == AArch64::TC_RETURNdi ||
      RetOpcode == AArch64::TC_RETURNxi) {
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(1);

    MachineInstrBuilder MIB;
    if (RetOpcode == AArch64::TC_RETURNdi) {
      MIB = BuildMI(MBB, MBBI, DL, TII.get(AArch64::TAIL_Bimm));
      if (JumpTarget.isGlobal()) {
        MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                             JumpTarget.getTargetFlags());
      } else {
        assert(JumpTarget.isSymbol() && "unexpected tail call destination");
        MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                              JumpTarget.getTargetFlags());
      }
    } else {
      assert(RetOpcode == AArch64::TC_RETURNxi && JumpTarget.isReg()
             && "Unexpected tail call");

      MIB = BuildMI(MBB, MBBI, DL, TII.get(AArch64::TAIL_BRx));
      MIB.addReg(JumpTarget.getReg(), RegState::Kill);
    }

    // Add the extra operands onto the new tail call instruction even though
    // they're not used directly (so that liveness is tracked properly etc).
    for (unsigned i = 2, e = MBBI->getNumOperands(); i != e; ++i)
        MIB->addOperand(MBBI->getOperand(i));


    // Delete the pseudo instruction TC_RETURN.
    MachineInstr *NewMI = prior(MBBI);
    MBB.erase(MBBI);
    MBBI = NewMI;

    // For a tail-call in a callee-pops-arguments environment, some or all of
    // the stack may actually be in use for the call's arguments, this is
    // calculated during LowerCall and consumed here...
    ArgumentPopSize = StackAdjust.getImm();
  } else {
    // ... otherwise the amount to pop is *all* of the argument space,
    // conveniently stored in the MachineFunctionInfo by
    // LowerFormalArguments. This will, of course, be zero for the C calling
    // convention.
    ArgumentPopSize = FuncInfo->getArgumentStackToRestore();
  }

  assert(NumInitialBytes % 16 == 0 && NumResidualBytes % 16 == 0
         && "refusing to adjust stack by misaligned amt");

  // We may need to address callee-saved registers differently, so find out the
  // bound on the frame indices.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  // The "residual" stack update comes first from this direction and guarantees
  // that SP is NumInitialBytes below its value on function entry, either by a
  // direct update or restoring it from the frame pointer.
  if (NumInitialBytes + ArgumentPopSize != 0) {
    emitSPUpdate(MBB, MBBI, DL, TII, AArch64::X16,
                 NumInitialBytes + ArgumentPopSize);
    --MBBI;
  }


  // MBBI now points to the instruction just past the last callee-saved
  // restoration (either RET/B if NumInitialBytes == 0, or the "ADD sp, sp"
  // otherwise).

  // Now we need to find out where to put the bulk of the stack adjustment
  MachineBasicBlock::iterator FirstEpilogue = MBBI;
  while (MBBI != MBB.begin()) {
    --MBBI;

    unsigned FrameOp;
    for (FrameOp = 0; FrameOp < MBBI->getNumOperands(); ++FrameOp) {
      if (MBBI->getOperand(FrameOp).isFI())
        break;
    }

    // If this instruction doesn't have a frame index we've reached the end of
    // the callee-save restoration.
    if (FrameOp == MBBI->getNumOperands())
      break;

    // Likewise if it *is* a local reference, but not to a callee-saved object.
    int FrameIdx = MBBI->getOperand(FrameOp).getIndex();
    if (FrameIdx < MinCSFI || FrameIdx > MaxCSFI)
      break;

    FirstEpilogue = MBBI;
  }

  if (MF.getFrameInfo()->hasVarSizedObjects()) {
    int64_t StaticFrameBase;
    StaticFrameBase = -(NumInitialBytes + FuncInfo->getFramePointerOffset());
    emitRegUpdate(MBB, FirstEpilogue, DL, TII,
                  AArch64::XSP, AArch64::X29, AArch64::NoRegister,
                  StaticFrameBase);
  } else {
    emitSPUpdate(MBB, FirstEpilogue, DL,TII, AArch64::X16, NumResidualBytes);
  }
}

int64_t
AArch64FrameLowering::resolveFrameIndexReference(MachineFunction &MF,
                                                 int FrameIndex,
                                                 unsigned &FrameReg,
                                                 int SPAdj,
                                                 bool IsCalleeSaveOp) const {
  AArch64MachineFunctionInfo *FuncInfo =
    MF.getInfo<AArch64MachineFunctionInfo>();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  int64_t TopOfFrameOffset = MFI->getObjectOffset(FrameIndex);

  assert(!(IsCalleeSaveOp && FuncInfo->getInitialStackAdjust() == 0)
         && "callee-saved register in unexpected place");

  // If the frame for this function is particularly large, we adjust the stack
  // in two phases which means the callee-save related operations see a
  // different (intermediate) stack size.
  int64_t FrameRegPos;
  if (IsCalleeSaveOp) {
    FrameReg = AArch64::XSP;
    FrameRegPos = -static_cast<int64_t>(FuncInfo->getInitialStackAdjust());
  } else if (useFPForAddressing(MF)) {
    // Have to use the frame pointer since we have no idea where SP is.
    FrameReg = AArch64::X29;
    FrameRegPos = FuncInfo->getFramePointerOffset();
  } else {
    FrameReg = AArch64::XSP;
    FrameRegPos = -static_cast<int64_t>(MFI->getStackSize()) + SPAdj;
  }

  return TopOfFrameOffset - FrameRegPos;
}

void
AArch64FrameLowering::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                       RegScavenger *RS) const {
  const AArch64RegisterInfo *RegInfo =
    static_cast<const AArch64RegisterInfo *>(MF.getTarget().getRegisterInfo());
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const AArch64InstrInfo &TII =
    *static_cast<const AArch64InstrInfo *>(MF.getTarget().getInstrInfo());

  if (hasFP(MF)) {
    MF.getRegInfo().setPhysRegUsed(AArch64::X29);
    MF.getRegInfo().setPhysRegUsed(AArch64::X30);
  }

  // If addressing of local variables is going to be more complicated than
  // shoving a base register and an offset into the instruction then we may well
  // need to scavenge registers. We should either specifically add an
  // callee-save register for this purpose or allocate an extra spill slot.

  bool BigStack =
    (RS && MFI->estimateStackSize(MF) >= TII.estimateRSStackLimit(MF))
    || MFI->hasVarSizedObjects() // Access will be from X29: messes things up
    || (MFI->adjustsStack() && !hasReservedCallFrame(MF));

  if (!BigStack)
    return;

  // We certainly need some slack space for the scavenger, preferably an extra
  // register.
  const uint16_t *CSRegs = RegInfo->getCalleeSavedRegs();
  uint16_t ExtraReg = AArch64::NoRegister;

  for (unsigned i = 0; CSRegs[i]; ++i) {
    if (AArch64::GPR64RegClass.contains(CSRegs[i]) &&
        !MF.getRegInfo().isPhysRegUsed(CSRegs[i])) {
      ExtraReg = CSRegs[i];
      break;
    }
  }

  if (ExtraReg != 0) {
    MF.getRegInfo().setPhysRegUsed(ExtraReg);
  } else {
    // Create a stack slot for scavenging purposes. PrologEpilogInserter
    // helpfully places it near either SP or FP for us to avoid
    // infinitely-regression during scavenging.
    const TargetRegisterClass *RC = &AArch64::GPR64RegClass;
    RS->addScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment(),
                                                       false));
  }
}

bool AArch64FrameLowering::determinePrologueDeath(MachineBasicBlock &MBB,
                                                  unsigned Reg) const {
  // If @llvm.returnaddress is called then it will refer to X30 by some means;
  // the prologue store does not kill the register.
  if (Reg == AArch64::X30) {
    if (MBB.getParent()->getFrameInfo()->isReturnAddressTaken()
        && MBB.getParent()->getRegInfo().isLiveIn(Reg))
    return false;
  }

  // In all other cases, physical registers are dead after they've been saved
  // but live at the beginning of the prologue block.
  MBB.addLiveIn(Reg);
  return true;
}

void
AArch64FrameLowering::emitFrameMemOps(bool isPrologue, MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      const std::vector<CalleeSavedInfo> &CSI,
                                      const TargetRegisterInfo *TRI,
                                      LoadStoreMethod PossClasses[],
                                      unsigned NumClasses) const {
  DebugLoc DL = MBB.findDebugLoc(MBBI);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  // A certain amount of implicit contract is present here. The actual stack
  // offsets haven't been allocated officially yet, so for strictly correct code
  // we rely on the fact that the elements of CSI are allocated in order
  // starting at SP, purely as dictated by size and alignment. In practice since
  // this function handles the only accesses to those slots it's not quite so
  // important.
  //
  // We have also ordered the Callee-saved register list in AArch64CallingConv
  // so that the above scheme puts registers in order: in particular we want
  // &X30 to be &X29+8 for an ABI-correct frame record (PCS 5.2.2)
  for (unsigned i = 0, e = CSI.size(); i < e; ++i) {
    unsigned Reg = CSI[i].getReg();

    // First we need to find out which register class the register belongs to so
    // that we can use the correct load/store instrucitons.
    unsigned ClassIdx;
    for (ClassIdx = 0; ClassIdx < NumClasses; ++ClassIdx) {
      if (PossClasses[ClassIdx].RegClass->contains(Reg))
        break;
    }
    assert(ClassIdx != NumClasses
           && "Asked to store register in unexpected class");
    const TargetRegisterClass &TheClass = *PossClasses[ClassIdx].RegClass;

    // Now we need to decide whether it's possible to emit a paired instruction:
    // for this we want the next register to be in the same class.
    MachineInstrBuilder NewMI;
    bool Pair = false;
    if (i + 1 < CSI.size() && TheClass.contains(CSI[i+1].getReg())) {
      Pair = true;
      unsigned StLow = 0, StHigh = 0;
      if (isPrologue) {
        // Most of these registers will be live-in to the MBB and killed by our
        // store, though there are exceptions (see determinePrologueDeath).
        StLow = getKillRegState(determinePrologueDeath(MBB, CSI[i+1].getReg()));
        StHigh = getKillRegState(determinePrologueDeath(MBB, CSI[i].getReg()));
      } else {
        StLow = RegState::Define;
        StHigh = RegState::Define;
      }

      NewMI = BuildMI(MBB, MBBI, DL, TII.get(PossClasses[ClassIdx].PairOpcode))
                .addReg(CSI[i+1].getReg(), StLow)
                .addReg(CSI[i].getReg(), StHigh);

      // If it's a paired op, we've consumed two registers
      ++i;
    } else {
      unsigned State;
      if (isPrologue) {
        State = getKillRegState(determinePrologueDeath(MBB, CSI[i].getReg()));
      } else {
        State = RegState::Define;
      }

      NewMI = BuildMI(MBB, MBBI, DL,
                      TII.get(PossClasses[ClassIdx].SingleOpcode))
                .addReg(CSI[i].getReg(), State);
    }

    // Note that the FrameIdx refers to the second register in a pair: it will
    // be allocated the smaller numeric address and so is the one an LDP/STP
    // address must use.
    int FrameIdx = CSI[i].getFrameIdx();
    MachineMemOperand::MemOperandFlags Flags;
    Flags = isPrologue ? MachineMemOperand::MOStore : MachineMemOperand::MOLoad;
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                             Flags,
                             Pair ? TheClass.getSize() * 2 : TheClass.getSize(),
                             MFI.getObjectAlignment(FrameIdx));

    NewMI.addFrameIndex(FrameIdx)
      .addImm(0)                  // address-register offset
      .addMemOperand(MMO);

    if (isPrologue)
      NewMI.setMIFlags(MachineInstr::FrameSetup);

    // For aesthetic reasons, during an epilogue we want to emit complementary
    // operations to the prologue, but in the opposite order. So we still
    // iterate through the CalleeSavedInfo list in order, but we put the
    // instructions successively earlier in the MBB.
    if (!isPrologue)
      --MBBI;
  }
}

bool
AArch64FrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  static LoadStoreMethod PossibleClasses[] = {
    {&AArch64::GPR64RegClass, AArch64::LSPair64_STR, AArch64::LS64_STR},
    {&AArch64::FPR64RegClass, AArch64::LSFPPair64_STR, AArch64::LSFP64_STR},
  };
  unsigned NumClasses = llvm::array_lengthof(PossibleClasses);

  emitFrameMemOps(/* isPrologue = */ true, MBB, MBBI, CSI, TRI,
                  PossibleClasses, NumClasses);

  return true;
}

bool
AArch64FrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {

  if (CSI.empty())
    return false;

  static LoadStoreMethod PossibleClasses[] = {
    {&AArch64::GPR64RegClass, AArch64::LSPair64_LDR, AArch64::LS64_LDR},
    {&AArch64::FPR64RegClass, AArch64::LSFPPair64_LDR, AArch64::LSFP64_LDR},
  };
  unsigned NumClasses = llvm::array_lengthof(PossibleClasses);

  emitFrameMemOps(/* isPrologue = */ false, MBB, MBBI, CSI, TRI,
                  PossibleClasses, NumClasses);

  return true;
}

bool
AArch64FrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getTarget().getRegisterInfo();

  // This is a decision of ABI compliance. The AArch64 PCS gives various options
  // for conformance, and even at the most stringent level more or less permits
  // elimination for leaf functions because there's no loss of functionality
  // (for debugging etc)..
  if (MF.getTarget().Options.DisableFramePointerElim(MF) && MFI->hasCalls())
    return true;

  // The following are hard-limits: incorrect code will be generated if we try
  // to omit the frame.
  return (RI->needsStackRealignment(MF) ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken());
}

bool
AArch64FrameLowering::useFPForAddressing(const MachineFunction &MF) const {
  return MF.getFrameInfo()->hasVarSizedObjects();
}

bool
AArch64FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Of the various reasons for having a frame pointer, it's actually only
  // variable-sized objects that prevent reservation of a call frame.
  return !(hasFP(MF) && MFI->hasVarSizedObjects());
}

void
AArch64FrameLowering::eliminateCallFramePseudoInstr(
                                MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
  const AArch64InstrInfo &TII =
    *static_cast<const AArch64InstrInfo *>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MI->getDebugLoc();
  int Opcode = MI->getOpcode();
  bool IsDestroy = Opcode == TII.getCallFrameDestroyOpcode();
  uint64_t CalleePopAmount = IsDestroy ? MI->getOperand(1).getImm() : 0;

  if (!hasReservedCallFrame(MF)) {
    unsigned Align = getStackAlignment();

    int64_t Amount = MI->getOperand(0).getImm();
    Amount = RoundUpToAlignment(Amount, Align);
    if (!IsDestroy) Amount = -Amount;

    // N.b. if CalleePopAmount is valid but zero (i.e. callee would pop, but it
    // doesn't have to pop anything), then the first operand will be zero too so
    // this adjustment is a no-op.
    if (CalleePopAmount == 0) {
      // FIXME: in-function stack adjustment for calls is limited to 12-bits
      // because there's no guaranteed temporary register available. Mostly call
      // frames will be allocated at the start of a function so this is OK, but
      // it is a limitation that needs dealing with.
      assert(Amount > -0xfff && Amount < 0xfff && "call frame too large");
      emitSPUpdate(MBB, MI, dl, TII, AArch64::NoRegister, Amount);
    }
  } else if (CalleePopAmount != 0) {
    // If the calling convention demands that the callee pops arguments from the
    // stack, we want to add it back if we have a reserved call frame.
    assert(CalleePopAmount < 0xfff && "call frame too large");
    emitSPUpdate(MBB, MI, dl, TII, AArch64::NoRegister, -CalleePopAmount);
  }

  MBB.erase(MI);
}
