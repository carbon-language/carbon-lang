//=======- ARMFrameInfo.cpp - ARM Frame Information ------------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of TargetFrameInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMFrameInfo.h"
#include "ARMBaseInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

static bool isCalleeSavedRegister(unsigned Reg, const unsigned *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI,
                        const ARMBaseInstrInfo &TII,
                        const unsigned *CSRegs) {
  // Integer spill area is handled with "pop".
  if (MI->getOpcode() == ARM::LDMIA_RET ||
      MI->getOpcode() == ARM::t2LDMIA_RET ||
      MI->getOpcode() == ARM::LDMIA_UPD ||
      MI->getOpcode() == ARM::t2LDMIA_UPD ||
      MI->getOpcode() == ARM::VLDMDIA_UPD) {
    // The first two operands are predicates. The last two are
    // imp-def and imp-use of SP. Check everything in between.
    for (int i = 5, e = MI->getNumOperands(); i != e; ++i)
      if (!isCalleeSavedRegister(MI->getOperand(i).getReg(), CSRegs))
        return false;
    return true;
  }

  return false;
}

static void
emitSPUpdate(bool isARM,
             MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
             DebugLoc dl, const ARMBaseInstrInfo &TII,
             int NumBytes,
             ARMCC::CondCodes Pred = ARMCC::AL, unsigned PredReg = 0) {
  if (isARM)
    emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                            Pred, PredReg, TII);
  else
    emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                           Pred, PredReg, TII);
}

void ARMFrameInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const ARMBaseRegisterInfo *RegInfo =
    static_cast<const ARMBaseRegisterInfo*>(MF.getTarget().getRegisterInfo());
  const ARMBaseInstrInfo &TII =
    *static_cast<const ARMBaseInstrInfo*>(MF.getTarget().getInstrInfo());
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitPrologue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  // Allocate the vararg register save area. This is not counted in NumBytes.
  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -VARegSaveSize);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes);
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
  
  // Move past area 1.
  if (GPRCS1Size > 0) MBBI++;
  
  // Set FP to point to the stack slot that contains the previous FP.
  // For Darwin, FP is R7, which has now been stored in spill area 1.
  // Otherwise, if this is not Darwin, all the callee-saved registers go
  // into spill area 1, including the FP in R11.  In either case, it is
  // now safe to emit this assignment.
  bool HasFP = RegInfo->hasFP(MF);
  if (HasFP) {
    unsigned ADDriOpc = !AFI->isThumbFunction() ? ARM::ADDri : ARM::t2ADDri;
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(ADDriOpc), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);
    AddDefaultCC(AddDefaultPred(MIB));
  }
  
  // Move past area 2.
  if (GPRCS2Size > 0) MBBI++;
  
  // Determine starting offsets of spill areas.
  unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
  unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
  unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
  if (HasFP)
    AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) +
                                NumBytes);
  AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
  AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
  AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);

  // Move past area 3.
  if (DPRCSSize > 0) MBBI++;
  
  NumBytes = DPRCSOffset;
  if (NumBytes) {
    // Adjust SP after all the callee-save spills.
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes);
    if (HasFP)
      AFI->setShouldRestoreSPFromFP(true);
  }

  if (STI.isTargetELF() && RegInfo->hasFP(MF)) {
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());
    AFI->setShouldRestoreSPFromFP(true);
  }

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);

  // If we need dynamic stack realignment, do it here. Be paranoid and make
  // sure if we also have VLAs, we have a base pointer for frame access.
  if (RegInfo->needsStackRealignment(MF)) {
    unsigned MaxAlign = MFI->getMaxAlignment();
    assert (!AFI->isThumb1OnlyFunction());
    if (!AFI->isThumbFunction()) {
      // Emit bic sp, sp, MaxAlign
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::BICri), ARM::SP)
                                  .addReg(ARM::SP, RegState::Kill)
                                  .addImm(MaxAlign-1)));
    } else {
      // We cannot use sp as source/dest register here, thus we're emitting the
      // following sequence:
      // mov r4, sp
      // bic r4, r4, MaxAlign
      // mov sp, r4
      // FIXME: It will be better just to find spare register here.
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVgpr2tgpr), ARM::R4)
        .addReg(ARM::SP, RegState::Kill);
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::t2BICri), ARM::R4)
                                  .addReg(ARM::R4, RegState::Kill)
                                  .addImm(MaxAlign-1)));
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVtgpr2gpr), ARM::SP)
        .addReg(ARM::R4, RegState::Kill);
    }

    AFI->setShouldRestoreSPFromFP(true);
  }

  // If we need a base pointer, set it up here. It's whatever the value
  // of the stack pointer is at this point. Any variable size objects
  // will be allocated after this, so we can still use the base pointer
  // to reference locals.
  if (RegInfo->hasBasePointer(MF)) {
    if (isARM)
      BuildMI(MBB, MBBI, dl,
              TII.get(ARM::MOVr), RegInfo->getBaseRegister())
        .addReg(ARM::SP)
        .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
    else
      BuildMI(MBB, MBBI, dl,
              TII.get(ARM::tMOVgpr2gpr), RegInfo->getBaseRegister())
        .addReg(ARM::SP);
  }

  // If the frame has variable sized objects then the epilogue must restore
  // the sp from fp.
  if (!AFI->shouldRestoreSPFromFP() && MFI->hasVarSizedObjects())
    AFI->setShouldRestoreSPFromFP(true);
}

void ARMFrameInfo::emitEpilogue(MachineFunction &MF,
                                MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getDesc().isReturn() &&
         "Can only insert epilog into returning blocks");
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  const ARMBaseInstrInfo &TII =
    *static_cast<const ARMBaseInstrInfo*>(MF.getTarget().getInstrInfo());
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitEpilogue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();

  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);
  } else {
    // Unwind MBBI to point to first LDR / VLDRD.
    const unsigned *CSRegs = RegInfo->getCalleeSavedRegs();
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, TII, CSRegs));
      if (!isCSRestore(MBBI, TII, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());

    // Reset SP based on frame pointer only if the stack frame extends beyond
    // frame pointer stack slot or target is ELF and the function has FP.
    if (AFI->shouldRestoreSPFromFP()) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      if (NumBytes) {
        if (isARM)
          emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, FramePtr, -NumBytes,
                                  ARMCC::AL, 0, TII);
        else
          emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::SP, FramePtr, -NumBytes,
                                 ARMCC::AL, 0, TII);
      } else {
        // Thumb2 or ARM.
        if (isARM)
          BuildMI(MBB, MBBI, dl, TII.get(ARM::MOVr), ARM::SP)
            .addReg(FramePtr).addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
        else
          BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVgpr2gpr), ARM::SP)
            .addReg(FramePtr);
      }
    } else if (NumBytes)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);

    // Increment past our save areas.
    if (AFI->getDPRCalleeSavedAreaSize()) MBBI++;
    if (AFI->getGPRCalleeSavedArea2Size()) MBBI++;
    if (AFI->getGPRCalleeSavedArea1Size()) MBBI++;
  }

  if (RetOpcode == ARM::TCRETURNdi || RetOpcode == ARM::TCRETURNdiND ||
      RetOpcode == ARM::TCRETURNri || RetOpcode == ARM::TCRETURNriND) {
    // Tail call return: adjust the stack pointer and jump to callee.
    MBBI = prior(MBB.end());
    MachineOperand &JumpTarget = MBBI->getOperand(0);

    // Jump to label or value in register.
    if (RetOpcode == ARM::TCRETURNdi) {
      BuildMI(MBB, MBBI, dl,
            TII.get(STI.isThumb() ? ARM::TAILJMPdt : ARM::TAILJMPd)).
        addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                         JumpTarget.getTargetFlags());
    } else if (RetOpcode == ARM::TCRETURNdiND) {
      BuildMI(MBB, MBBI, dl,
            TII.get(STI.isThumb() ? ARM::TAILJMPdNDt : ARM::TAILJMPdND)).
        addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                         JumpTarget.getTargetFlags());
    } else if (RetOpcode == ARM::TCRETURNri) {
      BuildMI(MBB, MBBI, dl, TII.get(ARM::TAILJMPr)).
        addReg(JumpTarget.getReg(), RegState::Kill);
    } else if (RetOpcode == ARM::TCRETURNriND) {
      BuildMI(MBB, MBBI, dl, TII.get(ARM::TAILJMPrND)).
        addReg(JumpTarget.getReg(), RegState::Kill);
    }

    MachineInstr *NewMI = prior(MBBI);
    for (unsigned i = 1, e = MBBI->getNumOperands(); i != e; ++i)
      NewMI->addOperand(MBBI->getOperand(i));

    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
  }

  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, VARegSaveSize);
}
