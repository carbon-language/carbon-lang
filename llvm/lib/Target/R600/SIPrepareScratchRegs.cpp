//===-- SIPrepareScratchRegs.cpp - Use predicates for control flow --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This pass loads scratch pointer and scratch offset into a register or a
/// frame index which can be used anywhere in the program.  These values will
/// be used for spilling VGPRs.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace {

class SIPrepareScratchRegs : public MachineFunctionPass {

private:
  static char ID;

public:
  SIPrepareScratchRegs() : MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI prepare scratch registers";
  }

};

} // End anonymous namespace

char SIPrepareScratchRegs::ID = 0;

FunctionPass *llvm::createSIPrepareScratchRegs() {
  return new SIPrepareScratchRegs();
}

bool SIPrepareScratchRegs::runOnMachineFunction(MachineFunction &MF) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  MachineBasicBlock *Entry = MF.begin();
  MachineBasicBlock::iterator I = Entry->begin();
  DebugLoc DL = I->getDebugLoc();

  // FIXME: If we don't have enough VGPRs for SGPR spilling we will need to
  // run this pass.
  if (!MFI->hasSpilledVGPRs())
    return false;

  unsigned ScratchPtrPreloadReg =
      TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_PTR);
  unsigned ScratchOffsetPreloadReg =
      TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_WAVE_OFFSET);

  if (!Entry->isLiveIn(ScratchPtrPreloadReg))
    Entry->addLiveIn(ScratchPtrPreloadReg);

  if (!Entry->isLiveIn(ScratchOffsetPreloadReg))
    Entry->addLiveIn(ScratchOffsetPreloadReg);

  // Load the scratch pointer
  unsigned ScratchPtrReg =
      TRI->findUnusedRegister(MRI, &AMDGPU::SGPR_64RegClass);
  int ScratchPtrFI = -1;

  if (ScratchPtrReg != AMDGPU::NoRegister) {
    // Found an SGPR to use.
    MRI.setPhysRegUsed(ScratchPtrReg);
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::S_MOV_B64), ScratchPtrReg)
            .addReg(ScratchPtrPreloadReg);
  } else {
    // No SGPR is available, we must spill.
    ScratchPtrFI = FrameInfo->CreateSpillStackObject(8, 4);
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::SI_SPILL_S64_SAVE))
            .addReg(ScratchPtrPreloadReg)
            .addFrameIndex(ScratchPtrFI);
  }

  // Load the scratch offset.
  unsigned ScratchOffsetReg =
      TRI->findUnusedRegister(MRI, &AMDGPU::SGPR_32RegClass);
  int ScratchOffsetFI = ~0;

  if (ScratchOffsetReg != AMDGPU::NoRegister) {
    // Found an SGPR to use
    MRI.setPhysRegUsed(ScratchOffsetReg);
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::S_MOV_B32), ScratchOffsetReg)
            .addReg(ScratchOffsetPreloadReg);
  } else {
    // No SGPR is available, we must spill.
    ScratchOffsetFI = FrameInfo->CreateSpillStackObject(4,4);
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::SI_SPILL_S32_SAVE))
            .addReg(ScratchOffsetPreloadReg)
            .addFrameIndex(ScratchOffsetFI);
  }


  // Now that we have the scratch pointer and offset values, we need to
  // add them to all the SI_SPILL_V* instructions.

  RegScavenger RS;
  bool UseRegScavenger =
      (ScratchPtrReg == AMDGPU::NoRegister ||
      ScratchOffsetReg == AMDGPU::NoRegister);
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    if (UseRegScavenger)
      RS.enterBasicBlock(&MBB);

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      MachineInstr &MI = *I;
      DebugLoc DL = MI.getDebugLoc();
      switch(MI.getOpcode()) {
        default: break;;
        case AMDGPU::SI_SPILL_V512_SAVE:
        case AMDGPU::SI_SPILL_V256_SAVE:
        case AMDGPU::SI_SPILL_V128_SAVE:
        case AMDGPU::SI_SPILL_V96_SAVE:
        case AMDGPU::SI_SPILL_V64_SAVE:
        case AMDGPU::SI_SPILL_V32_SAVE:
        case AMDGPU::SI_SPILL_V32_RESTORE:
        case AMDGPU::SI_SPILL_V64_RESTORE:
        case AMDGPU::SI_SPILL_V128_RESTORE:
        case AMDGPU::SI_SPILL_V256_RESTORE:
        case AMDGPU::SI_SPILL_V512_RESTORE:

          // Scratch Pointer
          if (ScratchPtrReg == AMDGPU::NoRegister) {
            ScratchPtrReg = RS.scavengeRegister(&AMDGPU::SGPR_64RegClass, 0);
            BuildMI(MBB, I, DL, TII->get(AMDGPU::SI_SPILL_S64_RESTORE),
                    ScratchPtrReg)
                    .addFrameIndex(ScratchPtrFI)
                    .addReg(AMDGPU::NoRegister)
                    .addReg(AMDGPU::NoRegister);
          } else if (!MBB.isLiveIn(ScratchPtrReg)) {
            MBB.addLiveIn(ScratchPtrReg);
          }

          if (ScratchOffsetReg == AMDGPU::NoRegister) {
            ScratchOffsetReg = RS.scavengeRegister(&AMDGPU::SGPR_32RegClass, 0);
            BuildMI(MBB, I, DL, TII->get(AMDGPU::SI_SPILL_S32_RESTORE),
                    ScratchOffsetReg)
                    .addFrameIndex(ScratchOffsetFI)
                    .addReg(AMDGPU::NoRegister)
                    .addReg(AMDGPU::NoRegister);
          } else if (!MBB.isLiveIn(ScratchOffsetReg)) {
            MBB.addLiveIn(ScratchOffsetReg);
          }

          if (ScratchPtrReg == AMDGPU::NoRegister ||
              ScratchOffsetReg == AMDGPU::NoRegister) {
            LLVMContext &Ctx = MF.getFunction()->getContext();
            Ctx.emitError("ran out of SGPRs for spilling VGPRs");
            ScratchPtrReg = AMDGPU::SGPR0;
            ScratchOffsetReg = AMDGPU::SGPR0;
          }
          MI.getOperand(2).setReg(ScratchPtrReg);
          MI.getOperand(3).setReg(ScratchOffsetReg);

          break;
      }
      if (UseRegScavenger)
        RS.forward();
    }
  }
  return true;
}
