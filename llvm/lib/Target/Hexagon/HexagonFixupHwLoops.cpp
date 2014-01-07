//===---- HexagonFixupHwLoops.cpp - Fixup HW loops too far from LOOPn. ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// The loop start address in the LOOPn instruction is encoded as a distance
// from the LOOPn instruction itself.  If the start address is too far from
// the LOOPn instruction, the loop needs to be set up manually, i.e. via
// direct transfers to SAn and LCn.
// This pass will identify and convert such LOOPn instructions to a proper
// form.
//===----------------------------------------------------------------------===//


#include "llvm/ADT/DenseMap.h"
#include "Hexagon.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/PassSupport.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

namespace llvm {
  void initializeHexagonFixupHwLoopsPass(PassRegistry&);
}

namespace {
  struct HexagonFixupHwLoops : public MachineFunctionPass {
  public:
    static char ID;

    HexagonFixupHwLoops() : MachineFunctionPass(ID) {
      initializeHexagonFixupHwLoopsPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "Hexagon Hardware Loop Fixup"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// \brief Maximum distance between the loop instr and the basic block.
    /// Just an estimate.
    static const unsigned MAX_LOOP_DISTANCE = 200;

    /// \brief Check the offset between each loop instruction and
    /// the loop basic block to determine if we can use the LOOP instruction
    /// or if we need to set the LC/SA registers explicitly.
    bool fixupLoopInstrs(MachineFunction &MF);

    /// \brief Add the instruction to set the LC and SA registers explicitly.
    void convertLoopInstr(MachineFunction &MF,
                          MachineBasicBlock::iterator &MII,
                          RegScavenger &RS);

  };

  char HexagonFixupHwLoops::ID = 0;
}

INITIALIZE_PASS(HexagonFixupHwLoops, "hwloopsfixup",
                "Hexagon Hardware Loops Fixup", false, false)

FunctionPass *llvm::createHexagonFixupHwLoops() {
  return new HexagonFixupHwLoops();
}


/// \brief Returns true if the instruction is a hardware loop instruction.
static bool isHardwareLoop(const MachineInstr *MI) {
  return MI->getOpcode() == Hexagon::LOOP0_r ||
         MI->getOpcode() == Hexagon::LOOP0_i;
}


bool HexagonFixupHwLoops::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = fixupLoopInstrs(MF);
  return Changed;
}


/// \brief For Hexagon, if the loop label is to far from the
/// loop instruction then we need to set the LC0 and SA0 registers
/// explicitly instead of using LOOP(start,count).  This function
/// checks the distance, and generates register assignments if needed.
///
/// This function makes two passes over the basic blocks.  The first
/// pass computes the offset of the basic block from the start.
/// The second pass checks all the loop instructions.
bool HexagonFixupHwLoops::fixupLoopInstrs(MachineFunction &MF) {

  // Offset of the current instruction from the start.
  unsigned InstOffset = 0;
  // Map for each basic block to it's first instruction.
  DenseMap<MachineBasicBlock*, unsigned> BlockToInstOffset;

  // First pass - compute the offset of each basic block.
  for (MachineFunction::iterator MBB = MF.begin(), MBBe = MF.end();
       MBB != MBBe; ++MBB) {
    BlockToInstOffset[MBB] = InstOffset;
    InstOffset += (MBB->size() * 4);
  }

  // Second pass - check each loop instruction to see if it needs to
  // be converted.
  InstOffset = 0;
  bool Changed = false;
  RegScavenger RS;

  // Loop over all the basic blocks.
  for (MachineFunction::iterator MBB = MF.begin(), MBBe = MF.end();
       MBB != MBBe; ++MBB) {
    InstOffset = BlockToInstOffset[MBB];
    RS.enterBasicBlock(MBB);

    // Loop over all the instructions.
    MachineBasicBlock::iterator MIE = MBB->end();
    MachineBasicBlock::iterator MII = MBB->begin();
    while (MII != MIE) {
      if (isHardwareLoop(MII)) {
        RS.forward(MII);
        assert(MII->getOperand(0).isMBB() &&
               "Expect a basic block as loop operand");
        int Sub = InstOffset - BlockToInstOffset[MII->getOperand(0).getMBB()];
        unsigned Dist = Sub > 0 ? Sub : -Sub;
        if (Dist > MAX_LOOP_DISTANCE) {
          // Convert to explicity setting LC0 and SA0.
          convertLoopInstr(MF, MII, RS);
          MII = MBB->erase(MII);
          Changed = true;
        } else {
          ++MII;
        }
      } else {
        ++MII;
      }
      InstOffset += 4;
    }
  }

  return Changed;
}


/// \brief convert a loop instruction to a sequence of instructions that
/// set the LC0 and SA0 register explicitly.
void HexagonFixupHwLoops::convertLoopInstr(MachineFunction &MF,
                                           MachineBasicBlock::iterator &MII,
                                           RegScavenger &RS) {
  const TargetInstrInfo *TII = MF.getTarget().getInstrInfo();
  MachineBasicBlock *MBB = MII->getParent();
  DebugLoc DL = MII->getDebugLoc();
  unsigned Scratch = RS.scavengeRegister(&Hexagon::IntRegsRegClass, MII, 0);

  // First, set the LC0 with the trip count.
  if (MII->getOperand(1).isReg()) {
    // Trip count is a register
    BuildMI(*MBB, MII, DL, TII->get(Hexagon::TFCR), Hexagon::LC0)
      .addReg(MII->getOperand(1).getReg());
  } else {
    // Trip count is an immediate.
    BuildMI(*MBB, MII, DL, TII->get(Hexagon::TFRI), Scratch)
      .addImm(MII->getOperand(1).getImm());
    BuildMI(*MBB, MII, DL, TII->get(Hexagon::TFCR), Hexagon::LC0)
      .addReg(Scratch);
  }
  // Then, set the SA0 with the loop start address.
  BuildMI(*MBB, MII, DL, TII->get(Hexagon::CONST32_Label), Scratch)
    .addMBB(MII->getOperand(0).getMBB());
  BuildMI(*MBB, MII, DL, TII->get(Hexagon::TFCR), Hexagon::SA0)
    .addReg(Scratch);
}
