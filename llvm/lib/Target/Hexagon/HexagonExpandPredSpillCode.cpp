//===-- HexagonExpandPredSpillCode.cpp - Expand Predicate Spill Code ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The Hexagon processor has no instructions that load or store predicate
// registers directly.  So, when these registers must be spilled a general
// purpose register must be found and the value copied to/from it from/to
// the predicate register.  This code currently does not use the register
// scavenger mechanism available in the allocator.  There are two registers
// reserved to allow spilling/restoring predicate registers.  One is used to
// hold the predicate value.  The other is used when stack frame offsets are
// too large.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/Statistic.h"
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


namespace llvm {
  void initializeHexagonExpandPredSpillCodePass(PassRegistry&);
}


namespace {

class HexagonExpandPredSpillCode : public MachineFunctionPass {
 public:
    static char ID;
    HexagonExpandPredSpillCode() : MachineFunctionPass(ID) {
      PassRegistry &Registry = *PassRegistry::getPassRegistry();
      initializeHexagonExpandPredSpillCodePass(Registry);
    }

    const char *getPassName() const override {
      return "Hexagon Expand Predicate Spill Code";
    }
    bool runOnMachineFunction(MachineFunction &Fn) override;
};


char HexagonExpandPredSpillCode::ID = 0;


bool HexagonExpandPredSpillCode::runOnMachineFunction(MachineFunction &Fn) {

  const HexagonSubtarget &QST = Fn.getSubtarget<HexagonSubtarget>();
  const HexagonInstrInfo *TII = QST.getInstrInfo();

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    // Traverse the basic block.
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         ++MII) {
      MachineInstr *MI = MII;
      int Opc = MI->getOpcode();
      if (Opc == Hexagon::S2_storerb_pci_pseudo ||
          Opc == Hexagon::S2_storerh_pci_pseudo ||
          Opc == Hexagon::S2_storeri_pci_pseudo ||
          Opc == Hexagon::S2_storerd_pci_pseudo ||
          Opc == Hexagon::S2_storerf_pci_pseudo) {
        unsigned Opcode;
        if (Opc == Hexagon::S2_storerd_pci_pseudo)
          Opcode = Hexagon::S2_storerd_pci;
        else if (Opc == Hexagon::S2_storeri_pci_pseudo)
          Opcode = Hexagon::S2_storeri_pci;
        else if (Opc == Hexagon::S2_storerh_pci_pseudo)
          Opcode = Hexagon::S2_storerh_pci;
        else if (Opc == Hexagon::S2_storerf_pci_pseudo)
          Opcode = Hexagon::S2_storerf_pci;
        else if (Opc == Hexagon::S2_storerb_pci_pseudo)
          Opcode = Hexagon::S2_storerb_pci;
        else
          llvm_unreachable("wrong Opc");
        MachineOperand &Op0 = MI->getOperand(0);
        MachineOperand &Op1 = MI->getOperand(1);
        MachineOperand &Op2 = MI->getOperand(2);
        MachineOperand &Op3 = MI->getOperand(3); // Modifier value.
        MachineOperand &Op4 = MI->getOperand(4);
        // Emit a "C6 = Rn, C6 is the control register for M0".
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_tfrrcr),
                Hexagon::C6)->addOperand(Op3);
        // Replace the pseude circ_ldd by the real circ_ldd.
        MachineInstr *NewMI = BuildMI(*MBB, MII, MI->getDebugLoc(),
                                      TII->get(Opcode));
        NewMI->addOperand(Op0);
        NewMI->addOperand(Op1);
        NewMI->addOperand(Op4);
        NewMI->addOperand(MachineOperand::CreateReg(Hexagon::M0,
                                                    false, /*isDef*/
                                                    false, /*isImpl*/
                                                    true   /*isKill*/));
        NewMI->addOperand(Op2);
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::L2_loadrd_pci_pseudo ||
                 Opc == Hexagon::L2_loadri_pci_pseudo ||
                 Opc == Hexagon::L2_loadrh_pci_pseudo ||
                 Opc == Hexagon::L2_loadruh_pci_pseudo||
                 Opc == Hexagon::L2_loadrb_pci_pseudo ||
                 Opc == Hexagon::L2_loadrub_pci_pseudo) {
        unsigned Opcode;
        if (Opc == Hexagon::L2_loadrd_pci_pseudo)
          Opcode = Hexagon::L2_loadrd_pci;
        else if (Opc == Hexagon::L2_loadri_pci_pseudo)
          Opcode = Hexagon::L2_loadri_pci;
        else if (Opc == Hexagon::L2_loadrh_pci_pseudo)
          Opcode = Hexagon::L2_loadrh_pci;
        else if (Opc == Hexagon::L2_loadruh_pci_pseudo)
          Opcode = Hexagon::L2_loadruh_pci;
        else if (Opc == Hexagon::L2_loadrb_pci_pseudo)
          Opcode = Hexagon::L2_loadrb_pci;
        else if (Opc == Hexagon::L2_loadrub_pci_pseudo)
          Opcode = Hexagon::L2_loadrub_pci;
        else
          llvm_unreachable("wrong Opc");

        MachineOperand &Op0 = MI->getOperand(0);
        MachineOperand &Op1 = MI->getOperand(1);
        MachineOperand &Op2 = MI->getOperand(2);
        MachineOperand &Op4 = MI->getOperand(4); // Modifier value.
        MachineOperand &Op5 = MI->getOperand(5);
        // Emit a "C6 = Rn, C6 is the control register for M0".
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_tfrrcr),
                Hexagon::C6)->addOperand(Op4);
        // Replace the pseude circ_ldd by the real circ_ldd.
        MachineInstr *NewMI = BuildMI(*MBB, MII, MI->getDebugLoc(),
                                      TII->get(Opcode));
        NewMI->addOperand(Op1);
        NewMI->addOperand(Op0);
        NewMI->addOperand(Op2);
        NewMI->addOperand(Op5);
        NewMI->addOperand(MachineOperand::CreateReg(Hexagon::M0,
                                                    false, /*isDef*/
                                                    false, /*isImpl*/
                                                    true   /*isKill*/));
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::L2_loadrd_pbr_pseudo ||
                 Opc == Hexagon::L2_loadri_pbr_pseudo ||
                 Opc == Hexagon::L2_loadrh_pbr_pseudo ||
                 Opc == Hexagon::L2_loadruh_pbr_pseudo||
                 Opc == Hexagon::L2_loadrb_pbr_pseudo ||
                 Opc == Hexagon::L2_loadrub_pbr_pseudo) {
        unsigned Opcode;
        if (Opc == Hexagon::L2_loadrd_pbr_pseudo)
          Opcode = Hexagon::L2_loadrd_pbr;
        else if (Opc == Hexagon::L2_loadri_pbr_pseudo)
          Opcode = Hexagon::L2_loadri_pbr;
        else if (Opc == Hexagon::L2_loadrh_pbr_pseudo)
          Opcode = Hexagon::L2_loadrh_pbr;
        else if (Opc == Hexagon::L2_loadruh_pbr_pseudo)
          Opcode = Hexagon::L2_loadruh_pbr;
        else if (Opc == Hexagon::L2_loadrb_pbr_pseudo)
          Opcode = Hexagon::L2_loadrb_pbr;
        else if (Opc == Hexagon::L2_loadrub_pbr_pseudo)
          Opcode = Hexagon::L2_loadrub_pbr;
        else
          llvm_unreachable("wrong Opc");
        MachineOperand &Op0 = MI->getOperand(0);
        MachineOperand &Op1 = MI->getOperand(1);
        MachineOperand &Op2 = MI->getOperand(2);
        MachineOperand &Op4 = MI->getOperand(4); // Modifier value.
        // Emit a "C6 = Rn, C6 is the control register for M0".
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_tfrrcr),
                Hexagon::C6)->addOperand(Op4);
        // Replace the pseudo brev_ldd by the real brev_ldd.
        MachineInstr *NewMI = BuildMI(*MBB, MII, MI->getDebugLoc(),
                                      TII->get(Opcode));
        NewMI->addOperand(Op1);
        NewMI->addOperand(Op0);
        NewMI->addOperand(Op2);
        NewMI->addOperand(MachineOperand::CreateReg(Hexagon::M0,
                                                    false, /*isDef*/
                                                    false, /*isImpl*/
                                                    true   /*isKill*/));
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::S2_storerd_pbr_pseudo ||
                 Opc == Hexagon::S2_storeri_pbr_pseudo ||
                 Opc == Hexagon::S2_storerh_pbr_pseudo ||
                 Opc == Hexagon::S2_storerb_pbr_pseudo ||
                 Opc == Hexagon::S2_storerf_pbr_pseudo) {
        unsigned Opcode;
        if (Opc == Hexagon::S2_storerd_pbr_pseudo)
          Opcode = Hexagon::S2_storerd_pbr;
        else if (Opc == Hexagon::S2_storeri_pbr_pseudo)
          Opcode = Hexagon::S2_storeri_pbr;
        else if (Opc == Hexagon::S2_storerh_pbr_pseudo)
          Opcode = Hexagon::S2_storerh_pbr;
        else if (Opc == Hexagon::S2_storerf_pbr_pseudo)
          Opcode = Hexagon::S2_storerf_pbr;
        else if (Opc == Hexagon::S2_storerb_pbr_pseudo)
          Opcode = Hexagon::S2_storerb_pbr;
        else
          llvm_unreachable("wrong Opc");
        MachineOperand &Op0 = MI->getOperand(0);
        MachineOperand &Op1 = MI->getOperand(1);
        MachineOperand &Op2 = MI->getOperand(2);
        MachineOperand &Op3 = MI->getOperand(3); // Modifier value.
        // Emit a "C6 = Rn, C6 is the control register for M0".
        BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_tfrrcr),
                Hexagon::C6)->addOperand(Op3);
        // Replace the pseudo brev_ldd by the real brev_ldd.
        MachineInstr *NewMI = BuildMI(*MBB, MII, MI->getDebugLoc(),
                                      TII->get(Opcode));
        NewMI->addOperand(Op0);
        NewMI->addOperand(Op1);
        NewMI->addOperand(MachineOperand::CreateReg(Hexagon::M0,
                                                    false, /*isDef*/
                                                    false, /*isImpl*/
                                                    true   /*isKill*/));
        NewMI->addOperand(Op2);
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::STriw_pred) {
        // STriw_pred [R30], ofst, SrcReg;
        unsigned FP = MI->getOperand(0).getReg();
        assert(FP == QST.getRegisterInfo()->getFrameRegister() &&
               "Not a Frame Pointer, Nor a Spill Slot");
        assert(MI->getOperand(1).isImm() && "Not an offset");
        int Offset = MI->getOperand(1).getImm();
        int SrcReg = MI->getOperand(2).getReg();
        assert(Hexagon::PredRegsRegClass.contains(SrcReg) &&
               "Not a predicate register");
        if (!TII->isValidOffset(Hexagon::S2_storeri_io, Offset)) {
          if (!TII->isValidOffset(Hexagon::A2_addi, Offset)) {
            BuildMI(*MBB, MII, MI->getDebugLoc(),
                    TII->get(Hexagon::CONST32_Int_Real),
                      HEXAGON_RESERVED_REG_1).addImm(Offset);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_add),
                    HEXAGON_RESERVED_REG_1)
              .addReg(FP).addReg(HEXAGON_RESERVED_REG_1);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrpr),
                      HEXAGON_RESERVED_REG_2).addReg(SrcReg);
            BuildMI(*MBB, MII, MI->getDebugLoc(),
                    TII->get(Hexagon::S2_storeri_io))
              .addReg(HEXAGON_RESERVED_REG_1)
              .addImm(0).addReg(HEXAGON_RESERVED_REG_2);
          } else {
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_addi),
                      HEXAGON_RESERVED_REG_1).addReg(FP).addImm(Offset);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrpr),
                      HEXAGON_RESERVED_REG_2).addReg(SrcReg);
            BuildMI(*MBB, MII, MI->getDebugLoc(),
                          TII->get(Hexagon::S2_storeri_io))
              .addReg(HEXAGON_RESERVED_REG_1)
              .addImm(0)
              .addReg(HEXAGON_RESERVED_REG_2);
          }
        } else {
          BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrpr),
                    HEXAGON_RESERVED_REG_2).addReg(SrcReg);
          BuildMI(*MBB, MII, MI->getDebugLoc(),
                        TII->get(Hexagon::S2_storeri_io)).
                    addReg(FP).addImm(Offset).addReg(HEXAGON_RESERVED_REG_2);
        }
        MII = MBB->erase(MI);
        --MII;
      } else if (Opc == Hexagon::LDriw_pred) {
        // DstReg = LDriw_pred [R30], ofst.
        int DstReg = MI->getOperand(0).getReg();
        assert(Hexagon::PredRegsRegClass.contains(DstReg) &&
               "Not a predicate register");
        unsigned FP = MI->getOperand(1).getReg();
        assert(FP == QST.getRegisterInfo()->getFrameRegister() &&
               "Not a Frame Pointer, Nor a Spill Slot");
        assert(MI->getOperand(2).isImm() && "Not an offset");
        int Offset = MI->getOperand(2).getImm();
        if (!TII->isValidOffset(Hexagon::L2_loadri_io, Offset)) {
          if (!TII->isValidOffset(Hexagon::A2_addi, Offset)) {
            BuildMI(*MBB, MII, MI->getDebugLoc(),
                    TII->get(Hexagon::CONST32_Int_Real),
                      HEXAGON_RESERVED_REG_1).addImm(Offset);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_add),
                    HEXAGON_RESERVED_REG_1)
              .addReg(FP)
              .addReg(HEXAGON_RESERVED_REG_1);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::L2_loadri_io),
                      HEXAGON_RESERVED_REG_2)
              .addReg(HEXAGON_RESERVED_REG_1)
              .addImm(0);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrrp),
                      DstReg).addReg(HEXAGON_RESERVED_REG_2);
          } else {
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::A2_addi),
                      HEXAGON_RESERVED_REG_1).addReg(FP).addImm(Offset);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::L2_loadri_io),
                      HEXAGON_RESERVED_REG_2)
              .addReg(HEXAGON_RESERVED_REG_1)
              .addImm(0);
            BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrrp),
                      DstReg).addReg(HEXAGON_RESERVED_REG_2);
          }
        } else {
          BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::L2_loadri_io),
                    HEXAGON_RESERVED_REG_2).addReg(FP).addImm(Offset);
          BuildMI(*MBB, MII, MI->getDebugLoc(), TII->get(Hexagon::C2_tfrrp),
                    DstReg).addReg(HEXAGON_RESERVED_REG_2);
        }
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

static void initializePassOnce(PassRegistry &Registry) {
  const char *Name = "Hexagon Expand Predicate Spill Code";
  PassInfo *PI = new PassInfo(Name, "hexagon-spill-pred",
                              &HexagonExpandPredSpillCode::ID,
                              nullptr, false, false);
  Registry.registerPass(*PI, true);
}

void llvm::initializeHexagonExpandPredSpillCodePass(PassRegistry &Registry) {
  CALL_ONCE_INITIALIZATION(initializePassOnce)
}

FunctionPass*
llvm::createHexagonExpandPredSpillCode() {
  return new HexagonExpandPredSpillCode();
}
