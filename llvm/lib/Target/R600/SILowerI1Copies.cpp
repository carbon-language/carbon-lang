//===-- SILowerI1Copies.cpp - Lower I1 Copies -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// i1 values are usually inserted by the CFG Structurize pass and they are
/// unique in that they can be copied from VALU to SALU registers.
/// This is not possible for any other value type.  Since there are no
/// MOV instructions for i1, we to use V_CMP_* and V_CNDMASK to move the i1.
///
//===----------------------------------------------------------------------===//
//

#define DEBUG_TYPE "si-i1-copies"
#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

class SILowerI1Copies : public MachineFunctionPass {
public:
  static char ID;

public:
  SILowerI1Copies() : MachineFunctionPass(ID) {
    initializeSILowerI1CopiesPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override;

  virtual const char *getPassName() const override {
    return "SI Lower il Copies";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
  AU.addRequired<MachineDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SILowerI1Copies, DEBUG_TYPE,
                      "SI Lower il Copies", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SILowerI1Copies, DEBUG_TYPE,
                    "SI Lower il Copies", false, false)

char SILowerI1Copies::ID = 0;

char &llvm::SILowerI1CopiesID = SILowerI1Copies::ID;

FunctionPass *llvm::createSILowerI1CopiesPass() {
  return new SILowerI1Copies();
}

bool SILowerI1Copies::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      MF.getTarget().getInstrInfo());
  const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      if (MI.getOpcode() == AMDGPU::V_MOV_I1) {
        MI.setDesc(TII->get(AMDGPU::V_MOV_B32_e32));
        continue;
      }

      if (MI.getOpcode() != AMDGPU::COPY ||
          !TargetRegisterInfo::isVirtualRegister(MI.getOperand(0).getReg()) ||
          !TargetRegisterInfo::isVirtualRegister(MI.getOperand(1).getReg()))
        continue;


      const TargetRegisterClass *DstRC =
          MRI.getRegClass(MI.getOperand(0).getReg());
      const TargetRegisterClass *SrcRC =
          MRI.getRegClass(MI.getOperand(1).getReg());

      if (DstRC == &AMDGPU::VReg_1RegClass &&
          TRI->getCommonSubClass(SrcRC, &AMDGPU::SGPR_64RegClass)) {
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(AMDGPU::V_CNDMASK_B32_e64))
                .addOperand(MI.getOperand(0))
                .addImm(0)
                .addImm(-1)
                .addOperand(MI.getOperand(1))
                .addImm(0)
                .addImm(0)
                .addImm(0)
                .addImm(0);
        MI.eraseFromParent();
      } else if (TRI->getCommonSubClass(DstRC, &AMDGPU::SGPR_64RegClass) &&
                 SrcRC == &AMDGPU::VReg_1RegClass) {
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(AMDGPU::V_CMP_NE_I32_e64))
                .addOperand(MI.getOperand(0))
                .addImm(0)
                .addOperand(MI.getOperand(1))
                .addImm(0)
                .addImm(0)
                .addImm(0)
                .addImm(0);
        MI.eraseFromParent();
      }

    }
  }
  return false;
}
