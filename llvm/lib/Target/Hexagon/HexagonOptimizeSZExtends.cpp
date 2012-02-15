//===-- HexagonOptimizeSZExtends.cpp - Identify and remove sign and -------===//
//===--                                zero extends.                -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/PassSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Debug.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include <algorithm>
#include "Hexagon.h"
#include "HexagonTargetMachine.h"

using namespace llvm;

namespace {
  struct HexagonOptimizeSZExtends : public MachineFunctionPass {

  public:
    static char ID;
    HexagonOptimizeSZExtends() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const {
      return "Hexagon remove redundant zero and size extends";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineFunctionAnalysis>();
      AU.addPreserved<MachineFunctionAnalysis>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
  };
}

char HexagonOptimizeSZExtends::ID = 0;

// This is a brain dead pass to get rid of redundant sign extends for the
// following case:
//
// Transform the following pattern
// %vreg170<def> = SXTW %vreg166
// ...
// %vreg176<def> = COPY %vreg170:subreg_loreg
//
// Into
// %vreg176<def> = COPY vreg166

bool HexagonOptimizeSZExtends::runOnMachineFunction(MachineFunction &MF) {
  DenseMap<unsigned, unsigned> SExtMap;

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBBb = MF.begin(), MBBe = MF.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    SExtMap.clear();

    // Traverse the basic block.
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         ++MII) {
      MachineInstr *MI = MII;
      // Look for sign extends:
      //   %vreg170<def> = SXTW %vreg166
      if (MI->getOpcode() == Hexagon::SXTW) {
        assert (MI->getNumOperands() == 2);
        MachineOperand &Dst = MI->getOperand(0);
        MachineOperand &Src  = MI->getOperand(1);
        unsigned DstReg = Dst.getReg();
        unsigned SrcReg = Src.getReg();
        // Just handle virtual registers.
        if (TargetRegisterInfo::isVirtualRegister(DstReg) &&
            TargetRegisterInfo::isVirtualRegister(SrcReg)) {
          // Map the following:
          //  %vreg170<def> = SXTW %vreg166
          //  SExtMap[170] = vreg166
          SExtMap[DstReg] = SrcReg;
        }
      }
      // Look for copy:
      //   %vreg176<def> = COPY %vreg170:subreg_loreg
      if (MI->isCopy()) {
        assert (MI->getNumOperands() == 2);
        MachineOperand &Dst = MI->getOperand(0);
        MachineOperand &Src  = MI->getOperand(1);

        // Make sure we are copying the lower 32 bits.
        if (Src.getSubReg() != Hexagon::subreg_loreg)
          continue;

        unsigned DstReg = Dst.getReg();
        unsigned SrcReg = Src.getReg();
        if (TargetRegisterInfo::isVirtualRegister(DstReg) &&
            TargetRegisterInfo::isVirtualRegister(SrcReg)) {
          // Try to find in the map.
          if (unsigned SextSrc = SExtMap.lookup(SrcReg)) {
            // Change the 1st operand.
            MI->RemoveOperand(1);
            MI->addOperand(MachineOperand::CreateReg(SextSrc, false));
          }
        }
      }
    }
  }
  return true;
}

FunctionPass *llvm::createHexagonOptimizeSZExtends() {
  return new HexagonOptimizeSZExtends();
}
