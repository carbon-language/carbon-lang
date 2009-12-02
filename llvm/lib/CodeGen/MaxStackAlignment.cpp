//===-- MaxStackAlignment.cpp - Compute the required stack alignment -- ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass looks for vector register usage and aligned local objects to
// calculate the maximum required alignment for a function. This is used by
// targets which support it to determine if dynamic stack realignment is
// necessary.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

namespace {
  struct MaximalStackAlignmentCalculator : public MachineFunctionPass {
    static char ID;
    MaximalStackAlignmentCalculator() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      MachineFrameInfo *FFI = MF.getFrameInfo();
      MachineRegisterInfo &RI = MF.getRegInfo();

      // Calculate max stack alignment of all already allocated stack objects.
      FFI->calculateMaxStackAlignment();
      unsigned MaxAlign = FFI->getMaxAlignment();

      // Be over-conservative: scan over all vreg defs and find whether vector
      // registers are used. If yes, there is probability that vector registers
      // will be spilled and thus the stack needs to be aligned properly.
      // FIXME: It would be better to only do this if a spill actually
      // happens rather than conseratively aligning the stack regardless.
      for (unsigned RegNum = TargetRegisterInfo::FirstVirtualRegister;
           RegNum < RI.getLastVirtReg(); ++RegNum)
        MaxAlign = std::max(MaxAlign, RI.getRegClass(RegNum)->getAlignment());

      if (FFI->getMaxAlignment() == MaxAlign)
        return false;

      FFI->setMaxAlignment(MaxAlign);
      return true;
    }

    virtual const char *getPassName() const {
      return "Stack Alignment Requirements Auto-Detector";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };

  char MaximalStackAlignmentCalculator::ID = 0;
}

FunctionPass*
llvm::createMaxStackAlignmentCalculatorPass() {
  return new MaximalStackAlignmentCalculator();
}

