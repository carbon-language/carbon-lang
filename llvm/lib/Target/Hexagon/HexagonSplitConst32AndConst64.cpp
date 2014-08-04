//=== HexagonSplitConst32AndConst64.cpp - split CONST32/Const64 into HI/LO ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// When the compiler is invoked with no small data, for instance, with the -G0
// command line option, then all CONST32_* opcodes should be broken down into
// appropriate LO and HI instructions. This splitting is done by this pass.
// The only reason this is not done in the DAG lowering itself is that there
// is no simple way of getting the register allocator to allot the same hard
// register to the result of LO and HI instructions. This pass is always
// scheduled after register allocation.
//
//===----------------------------------------------------------------------===//

#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonTargetObjectFile.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "xfer"

namespace {

class HexagonSplitConst32AndConst64 : public MachineFunctionPass {
  const HexagonTargetMachine &QTM;

 public:
    static char ID;
    HexagonSplitConst32AndConst64(const HexagonTargetMachine &TM)
        : MachineFunctionPass(ID), QTM(TM) {}

    const char *getPassName() const override {
      return "Hexagon Split Const32s and Const64s";
    }
    bool runOnMachineFunction(MachineFunction &Fn) override;
};


char HexagonSplitConst32AndConst64::ID = 0;


bool HexagonSplitConst32AndConst64::runOnMachineFunction(MachineFunction &Fn) {

  const HexagonTargetObjectFile &TLOF =
      (const HexagonTargetObjectFile &)QTM.getSubtargetImpl()
          ->getTargetLowering()
          ->getObjFileLowering();
  if (TLOF.IsSmallDataEnabled())
    return true;

  const TargetInstrInfo *TII = QTM.getSubtargetImpl()->getInstrInfo();

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    // Traverse the basic block
    MachineBasicBlock::iterator MII = MBB->begin();
    MachineBasicBlock::iterator MIE = MBB->end ();
    while (MII != MIE) {
      MachineInstr *MI = MII;
      int Opc = MI->getOpcode();
      if (Opc == Hexagon::CONST32_set) {
        int DestReg = MI->getOperand(0).getReg();
        MachineOperand &Symbol = MI->getOperand (1);

        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LO), DestReg).addOperand(Symbol);
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HI), DestReg).addOperand(Symbol);
        // MBB->erase returns the iterator to the next instruction, which is the
        // one we want to process next
        MII = MBB->erase (MI);
        continue;
      }
      else if (Opc == Hexagon::CONST32_set_jt) {
        int DestReg = MI->getOperand(0).getReg();
        MachineOperand &Symbol = MI->getOperand (1);

        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LO_jt), DestReg).addOperand(Symbol);
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HI_jt), DestReg).addOperand(Symbol);
        // MBB->erase returns the iterator to the next instruction, which is the
        // one we want to process next
        MII = MBB->erase (MI);
        continue;
      }
      else if (Opc == Hexagon::CONST32_Label) {
        int DestReg = MI->getOperand(0).getReg();
        MachineOperand &Symbol = MI->getOperand (1);

        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LO_label), DestReg).addOperand(Symbol);
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HI_label), DestReg).addOperand(Symbol);
        // MBB->erase returns the iterator to the next instruction, which is the
        // one we want to process next
        MII = MBB->erase (MI);
        continue;
      }
      else if (Opc == Hexagon::CONST32_Int_Real) {
        int DestReg = MI->getOperand(0).getReg();
        int64_t ImmValue = MI->getOperand(1).getImm ();

        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LOi), DestReg).addImm(ImmValue);
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HIi), DestReg).addImm(ImmValue);
        MII = MBB->erase (MI);
        continue;
      }
      else if (Opc == Hexagon::CONST64_Int_Real) {
        int DestReg = MI->getOperand(0).getReg();
        int64_t ImmValue = MI->getOperand(1).getImm ();
        unsigned DestLo = QTM.getSubtargetImpl()->getRegisterInfo()->getSubReg(
            DestReg, Hexagon::subreg_loreg);
        unsigned DestHi = QTM.getSubtargetImpl()->getRegisterInfo()->getSubReg(
            DestReg, Hexagon::subreg_hireg);

        int32_t LowWord = (ImmValue & 0xFFFFFFFF);
        int32_t HighWord = (ImmValue >> 32) & 0xFFFFFFFF;

        // Lower Registers Lower Half
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LOi), DestLo).addImm(LowWord);
        // Lower Registers Higher Half
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HIi), DestLo).addImm(LowWord);
        // Higher Registers Lower Half
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::LOi), DestHi).addImm(HighWord);
        // Higher Registers Higher Half.
        BuildMI (*MBB, MII, MI->getDebugLoc(),
                 TII->get(Hexagon::HIi), DestHi).addImm(HighWord);
        MII = MBB->erase (MI);
        continue;
       }
      ++MII;
    }
  }

  return true;
}

}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *
llvm::createHexagonSplitConst32AndConst64(const HexagonTargetMachine &TM) {
  return new HexagonSplitConst32AndConst64(TM);
}
