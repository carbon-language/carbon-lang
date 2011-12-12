//===---- HexagonSplitTFRCondSets.cpp - split TFR condsets into xfers -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//===----------------------------------------------------------------------===////
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

#define DEBUG_TYPE "xfer"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "HexagonTargetMachine.h"
#include "HexagonSubtarget.h"
#include "HexagonMachineFunctionInfo.h"
#include <map>
#include <iostream>

#include "llvm/Support/CommandLine.h"
#define DEBUG_TYPE "xfer"


using namespace llvm;

namespace {

class HexagonSplitTFRCondSets : public MachineFunctionPass {
    HexagonTargetMachine& QTM;
    const HexagonSubtarget &QST;

 public:
    static char ID;
    HexagonSplitTFRCondSets(HexagonTargetMachine& TM) :
      MachineFunctionPass(ID), QTM(TM), QST(*TM.getSubtargetImpl()) {}

    const char *getPassName() const {
      return "Hexagon Split TFRCondSets";
    }
    bool runOnMachineFunction(MachineFunction &Fn);
};


char HexagonSplitTFRCondSets::ID = 0;


bool HexagonSplitTFRCondSets::runOnMachineFunction(MachineFunction &Fn) {

  const TargetInstrInfo *TII = QTM.getInstrInfo();

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    // Traverse the basic block.
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         ++MII) {
      MachineInstr *MI = MII;
      int Opc = MI->getOpcode();
      if (Opc == Hexagon::TFR_condset_rr) {

        int DestReg = MI->getOperand(0).getReg();
        int SrcReg1 = MI->getOperand(2).getReg();
        int SrcReg2 = MI->getOperand(3).getReg();

        // Minor optimization: do not emit the predicated copy if the source and
        // the destination is the same register
        if (DestReg != SrcReg1) {
          BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::TFR_cPt),
                  DestReg).addReg(MI->getOperand(1).getReg()).addReg(SrcReg1);
        }
        if (DestReg != SrcReg2) {
          BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::TFR_cNotPt),
                  DestReg).addReg(MI->getOperand(1).getReg()).addReg(SrcReg2);
        }
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::TFR_condset_ii) {
        int DestReg = MI->getOperand(0).getReg();
        int SrcReg1 = MI->getOperand(1).getReg();
        int Immed1 = MI->getOperand(2).getImm();
        int Immed2 = MI->getOperand(3).getImm();
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::TFRI_cPt),
                DestReg).addReg(SrcReg1).addImm(Immed1);
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::TFRI_cNotPt),
                DestReg).addReg(SrcReg1).addImm(Immed2);
        MII = MBB->erase(MI);
        --MII;
      }
    }
  }

  return true;
}

}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonSplitTFRCondSets(HexagonTargetMachine &TM) {
  return new HexagonSplitTFRCondSets(TM);
}
