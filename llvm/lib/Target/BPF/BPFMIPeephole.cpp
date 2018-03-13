//===-------------- BPFMIPeephole.cpp - MI Peephole Cleanups  -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs peephole optimizations to cleanup ugly code sequences at
// MachineInstruction layer.
//
// Currently, the only optimization in this pass is to eliminate type promotion
// sequences, those zero extend 32-bit subregisters to 64-bit registers, if the
// compiler could prove the subregisters is defined by 32-bit operations in
// which case the upper half of the underlying 64-bit registers were zeroed
// implicitly.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-mi-zext-elim"

STATISTIC(ZExtElemNum, "Number of zero extension shifts eliminated");

namespace {

struct BPFMIPeephole : public MachineFunctionPass {

  static char ID;
  const BPFInstrInfo *TII;
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  BPFMIPeephole() : MachineFunctionPass(ID) {
    initializeBPFMIPeepholePass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  bool isMovFrom32Def(MachineInstr *MovMI);
  bool eliminateZExtSeq(void);

public:

  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    initialize(MF);

    return eliminateZExtSeq();
  }
};

// Initialize class variables.
void BPFMIPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  TII = MF->getSubtarget<BPFSubtarget>().getInstrInfo();
  DEBUG(dbgs() << "*** BPF MI peephole pass ***\n\n");
}

bool BPFMIPeephole::isMovFrom32Def(MachineInstr *MovMI)
{
  MachineInstr *DefInsn = MRI->getVRegDef(MovMI->getOperand(1).getReg());

  if (!DefInsn)
    return false;

  if (DefInsn->isPHI()) {
    for (unsigned i = 1, e = DefInsn->getNumOperands(); i < e; i += 2) {
      MachineOperand &opnd = DefInsn->getOperand(i);

      if (!opnd.isReg())
        return false;

      MachineInstr *PhiDef = MRI->getVRegDef(opnd.getReg());
      // quick check on PHI incoming definitions.
      if (!PhiDef || PhiDef->isPHI() || PhiDef->getOpcode() == BPF::COPY)
        return false;
    }
  }

  if (DefInsn->getOpcode() == BPF::COPY) {
    MachineOperand &opnd = DefInsn->getOperand(1);

    if (!opnd.isReg())
      return false;

    unsigned Reg = opnd.getReg();
    if ((TargetRegisterInfo::isVirtualRegister(Reg) &&
         MRI->getRegClass(Reg) == &BPF::GPRRegClass))
       return false;
  }

  return true;
}

bool BPFMIPeephole::eliminateZExtSeq(void) {
  MachineInstr* ToErase = nullptr;
  bool Eliminated = false;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      // If the previous instruction was marked for elimination, remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // Eliminate the 32-bit to 64-bit zero extension sequence when possible.
      //
      //   MOV_32_64 rB, wA
      //   SLL_ri    rB, rB, 32
      //   SRL_ri    rB, rB, 32
      if (MI.getOpcode() == BPF::SRL_ri &&
          MI.getOperand(2).getImm() == 32) {
        unsigned DstReg = MI.getOperand(0).getReg();
        unsigned ShfReg = MI.getOperand(1).getReg();
        MachineInstr *SllMI = MRI->getVRegDef(ShfReg);

        if (!SllMI ||
            SllMI->isPHI() ||
            SllMI->getOpcode() != BPF::SLL_ri ||
            SllMI->getOperand(2).getImm() != 32)
          continue;

        MachineInstr *MovMI = MRI->getVRegDef(SllMI->getOperand(1).getReg());
        if (!MovMI ||
            MovMI->isPHI() ||
            MovMI->getOpcode() != BPF::MOV_32_64)
          continue;

        unsigned SubReg = MovMI->getOperand(1).getReg();
        if (!isMovFrom32Def(MovMI))
          continue;

        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(BPF::SUBREG_TO_REG), DstReg)
          .addImm(0).addReg(SubReg).addImm(BPF::sub_32);

        SllMI->eraseFromParent();
        MovMI->eraseFromParent();
        // MI is the right shift, we can't erase it in it's own iteration.
        // Mark it to ToErase, and erase in the next iteration.
        ToErase = &MI;
        ZExtElemNum++;
        Eliminated = true;
      }
    }
  }

  return Eliminated;
}

} // end default namespace

INITIALIZE_PASS(BPFMIPeephole, DEBUG_TYPE, "BPF MI Peephole Optimization",
                false, false)

char BPFMIPeephole::ID = 0;
FunctionPass* llvm::createBPFMIPeepholePass() { return new BPFMIPeephole(); }
