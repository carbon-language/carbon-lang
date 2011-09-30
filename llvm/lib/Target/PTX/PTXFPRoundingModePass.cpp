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
#include "llvm/ADT/DenseMap.h"
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

      typedef std::pair<unsigned, unsigned> RndModeDesc;

      PTXTargetMachine& TargetMachine;
      DenseMap<unsigned, RndModeDesc> Instrs;

    public:
      PTXFPRoundingModePass(PTXTargetMachine &TM, CodeGenOpt::Level OptLevel)
        : MachineFunctionPass(ID),
          TargetMachine(TM) {
        initializeMap();
      }

      virtual bool runOnMachineFunction(MachineFunction &MF);

      virtual const char *getPassName() const {
        return "PTX FP Rounding Mode Pass";
      }

    private:

      void initializeMap();
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

void PTXFPRoundingModePass::initializeMap() {
  using namespace PTXRoundingMode;
  const PTXSubtarget& ST = TargetMachine.getSubtarget<PTXSubtarget>();

  // Build a map of default rounding mode for all instructions that need a
  // rounding mode.
  Instrs[PTX::FADDrr32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FADDri32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FADDrr64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FADDri64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSUBrr32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSUBri32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSUBrr64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSUBri64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FMULrr32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FMULri32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FMULrr64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FMULri64] = std::make_pair(1U, (unsigned)RndNearestEven);

  Instrs[PTX::FNEGrr32] = std::make_pair(1U, (unsigned)RndNone);
  Instrs[PTX::FNEGri32] = std::make_pair(1U, (unsigned)RndNone);
  Instrs[PTX::FNEGrr64] = std::make_pair(1U, (unsigned)RndNone);
  Instrs[PTX::FNEGri64] = std::make_pair(1U, (unsigned)RndNone);

  unsigned FDivRndMode = ST.fdivNeedsRoundingMode() ? RndNearestEven : RndNone;
  Instrs[PTX::FDIVrr32] = std::make_pair(1U, FDivRndMode);
  Instrs[PTX::FDIVri32] = std::make_pair(1U, FDivRndMode);
  Instrs[PTX::FDIVrr64] = std::make_pair(1U, FDivRndMode);
  Instrs[PTX::FDIVri64] = std::make_pair(1U, FDivRndMode);

  unsigned FMADRndMode = ST.fmadNeedsRoundingMode() ? RndNearestEven : RndNone;
  Instrs[PTX::FMADrrr32] = std::make_pair(1U, FMADRndMode);
  Instrs[PTX::FMADrri32] = std::make_pair(1U, FMADRndMode);
  Instrs[PTX::FMADrii32] = std::make_pair(1U, FMADRndMode);
  Instrs[PTX::FMADrrr64] = std::make_pair(1U, FMADRndMode);
  Instrs[PTX::FMADrri64] = std::make_pair(1U, FMADRndMode);
  Instrs[PTX::FMADrii64] = std::make_pair(1U, FMADRndMode);

  Instrs[PTX::FSQRTrr32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSQRTri32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSQRTrr64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::FSQRTri64] = std::make_pair(1U, (unsigned)RndNearestEven);

  Instrs[PTX::FSINrr32] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FSINri32] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FSINrr64] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FSINri64] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FCOSrr32] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FCOSri32] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FCOSrr64] = std::make_pair(1U, (unsigned)RndApprox);
  Instrs[PTX::FCOSri64] = std::make_pair(1U, (unsigned)RndApprox);

  Instrs[PTX::CVTu16f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs16f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTu16f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs16f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTu32f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs32f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTu32f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs32f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTu64f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs64f32] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTu64f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);
  Instrs[PTX::CVTs64f64] = std::make_pair(1U, (unsigned)RndTowardsZeroInt);

  Instrs[PTX::CVTf32u16] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32s16] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32u32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32s32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32u64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32s64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf32f64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64u16] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64s16] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64u32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64s32] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64u64] = std::make_pair(1U, (unsigned)RndNearestEven);
  Instrs[PTX::CVTf64s64] = std::make_pair(1U, (unsigned)RndNearestEven);
}

void PTXFPRoundingModePass::processInstruction(MachineInstr &MI) {
  // Is this an instruction that needs a rounding mode?
  if (Instrs.count(MI.getOpcode())) {
    const RndModeDesc &Desc = Instrs[MI.getOpcode()];
    // Get the rounding mode operand
    MachineOperand &Op = MI.getOperand(Desc.first);
    // Update the rounding mode if needed
    if (Op.getImm() == PTXRoundingMode::RndDefault) {
      Op.setImm(Desc.second);
    }
  }
}

FunctionPass *llvm::createPTXFPRoundingModePass(PTXTargetMachine &TM,
                                                CodeGenOpt::Level OptLevel) {
  return new PTXFPRoundingModePass(TM, OptLevel);
}

