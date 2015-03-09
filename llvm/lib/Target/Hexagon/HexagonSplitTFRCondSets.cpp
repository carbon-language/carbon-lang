//===-- HexagonSplitTFRCondSets.cpp - split TFR condsets into xfers -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//===----------------------------------------------------------------------===//
// This pass tries to provide opportunities for better optimization of muxes.
// The default code generated for something like: flag = (a == b) ? 1 : 3;
// would be:
//
//   {p0 = cmp.eq(r0,r1)}
//   {r3 = mux(p0,#1,#3)}
//
// This requires two packets.  If we use .new predicated immediate transfers,
// then we can do this in a single packet, e.g.:
//
//   {p0 = cmp.eq(r0,r1)
//    if (p0.new) r3 = #1
//    if (!p0.new) r3 = #3}
//
// Note that the conditional assignments are not generated in .new form here.
// We assume opptimisically that they will be formed later.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "xfer"

namespace llvm {
  void initializeHexagonSplitTFRCondSetsPass(PassRegistry&);
}


namespace {

class HexagonSplitTFRCondSets : public MachineFunctionPass {
 public:
    static char ID;
    HexagonSplitTFRCondSets() : MachineFunctionPass(ID) {
      initializeHexagonSplitTFRCondSetsPass(*PassRegistry::getPassRegistry());
    }

    const char *getPassName() const override {
      return "Hexagon Split TFRCondSets";
    }
    bool runOnMachineFunction(MachineFunction &Fn) override;
};


char HexagonSplitTFRCondSets::ID = 0;


bool HexagonSplitTFRCondSets::runOnMachineFunction(MachineFunction &Fn) {

  const TargetInstrInfo *TII = Fn.getSubtarget().getInstrInfo();

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    // Traverse the basic block.
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         ++MII) {
      MachineInstr *MI = MII;
      switch(MI->getOpcode()) {
        case Hexagon::TFR_condset_ii: {
          int DestReg = MI->getOperand(0).getReg();
          int SrcReg1 = MI->getOperand(1).getReg();

          int Immed1 = MI->getOperand(2).getImm();
          int Immed2 = MI->getOperand(3).getImm();
          BuildMI(*MBB, MII, MI->getDebugLoc(),
                  TII->get(Hexagon::C2_cmoveit),
                  DestReg).addReg(SrcReg1).addImm(Immed1);
          BuildMI(*MBB, MII, MI->getDebugLoc(),
                  TII->get(Hexagon::C2_cmoveif),
                  DestReg).addReg(SrcReg1).addImm(Immed2);
          MII = MBB->erase(MI);
          --MII;
          break;
        }
      }
    }
  }
  return true;
}

}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

static void initializePassOnce(PassRegistry &Registry) {
  const char *Name = "Hexagon Split TFRCondSets";
  PassInfo *PI = new PassInfo(Name, "hexagon-split-tfr",
                              &HexagonSplitTFRCondSets::ID, nullptr, false,
                              false);
  Registry.registerPass(*PI, true);
}

void llvm::initializeHexagonSplitTFRCondSetsPass(PassRegistry &Registry) {
  CALL_ONCE_INITIALIZATION(initializePassOnce)
}

FunctionPass *llvm::createHexagonSplitTFRCondSets() {
  return new HexagonSplitTFRCondSets();
}
