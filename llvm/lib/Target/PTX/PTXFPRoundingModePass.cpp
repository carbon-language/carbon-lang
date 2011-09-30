//===-- PTXFPRoundingModePass.cpp - Assign rounding modes pass ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a machine function pass that sets appropriate FP rounding
// modes for all relevant instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ptx-fp-rounding-mode"

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

// NOTE: PTXFPRoundingModePass should be executed just before emission.

namespace llvm {
  /// PTXFPRoundingModePass - Pass to assign appropriate FP rounding modes to
  /// all FP instructions. Essentially, this pass just looks for all FP
  /// instructions that have a rounding mode set to RndDefault, and sets an
  /// appropriate rounding mode based on the target device.
  ///
  class PTXFPRoundingModePass : public MachineFunctionPass {
    private:
      static char ID;
      PTXTargetMachine& TargetMachine;

    public:
      PTXFPRoundingModePass(PTXTargetMachine &TM, CodeGenOpt::Level OptLevel)
        : MachineFunctionPass(ID),
          TargetMachine(TM) {}

      virtual bool runOnMachineFunction(MachineFunction &MF);

      virtual const char *getPassName() const {
        return "PTX FP Rounding Mode Pass";
      }

    private:

      void processInstruction(MachineInstr &MI);
  }; // class PTXFPRoundingModePass
} // namespace llvm

using namespace llvm;

char PTXFPRoundingModePass::ID = 0;

bool PTXFPRoundingModePass::runOnMachineFunction(MachineFunction &MF) {

  // Look at each basic block
  for (MachineFunction::iterator bbi = MF.begin(), bbe = MF.end(); bbi != bbe;
       ++bbi) {
    MachineBasicBlock &MBB = *bbi;
    // Look at each instruction
    for (MachineBasicBlock::iterator ii = MBB.begin(), ie = MBB.end();
         ii != ie; ++ii) {
      MachineInstr &MI = *ii;
      processInstruction(MI);
    }
  }
  return false;
}

void PTXFPRoundingModePass::processInstruction(MachineInstr &MI) {
  // If the instruction has a rounding mode set to RndDefault, then assign an
  // appropriate rounding mode based on the target device.
  const PTXSubtarget& ST = TargetMachine.getSubtarget<PTXSubtarget>();
  switch (MI.getOpcode()) {
  case PTX::FADDrr32:
  case PTX::FADDri32:
  case PTX::FADDrr64:
  case PTX::FADDri64:
  case PTX::FSUBrr32:
  case PTX::FSUBri32:
  case PTX::FSUBrr64:
  case PTX::FSUBri64:
  case PTX::FMULrr32:
  case PTX::FMULri32:
  case PTX::FMULrr64:
  case PTX::FMULri64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      MI.getOperand(1).setImm(PTXRoundingMode::RndNearestEven);
    }
    break;
  case PTX::FNEGrr32:
  case PTX::FNEGri32:
  case PTX::FNEGrr64:
  case PTX::FNEGri64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      MI.getOperand(1).setImm(PTXRoundingMode::RndNone);
    }
    break;
  case PTX::FDIVrr32:
  case PTX::FDIVri32:
  case PTX::FDIVrr64:
  case PTX::FDIVri64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      if (ST.fdivNeedsRoundingMode())
        MI.getOperand(1).setImm(PTXRoundingMode::RndNearestEven);
      else
        MI.getOperand(1).setImm(PTXRoundingMode::RndNone);
    }
    break;
  case PTX::FMADrrr32:
  case PTX::FMADrri32:
  case PTX::FMADrii32:
  case PTX::FMADrrr64:
  case PTX::FMADrri64:
  case PTX::FMADrii64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      if (ST.fmadNeedsRoundingMode())
        MI.getOperand(1).setImm(PTXRoundingMode::RndNearestEven);
      else
        MI.getOperand(1).setImm(PTXRoundingMode::RndNone);
    }
    break;
  case PTX::FSQRTrr32:
  case PTX::FSQRTri32:
  case PTX::FSQRTrr64:
  case PTX::FSQRTri64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      MI.getOperand(1).setImm(PTXRoundingMode::RndNearestEven);
    }
    break;
  case PTX::FSINrr32:
  case PTX::FSINri32:
  case PTX::FSINrr64:
  case PTX::FSINri64:
  case PTX::FCOSrr32:
  case PTX::FCOSri32:
  case PTX::FCOSrr64:
  case PTX::FCOSri64:
    if (MI.getOperand(1).getImm() == PTXRoundingMode::RndDefault) {
      MI.getOperand(1).setImm(PTXRoundingMode::RndApprox);
    }
    break;
  }
}

FunctionPass *llvm::createPTXFPRoundingModePass(PTXTargetMachine &TM,
                                                CodeGenOpt::Level OptLevel) {
  return new PTXFPRoundingModePass(TM, OptLevel);
}

