//===-- SIFoldOperands.cpp - Fold operands --- ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//
//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
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

#define DEBUG_TYPE "si-fold-operands"
using namespace llvm;

namespace {

class SIFoldOperands : public MachineFunctionPass {
public:
  static char ID;

public:
  SIFoldOperands() : MachineFunctionPass(ID) {
    initializeSIFoldOperandsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fold Operands";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFoldOperands, DEBUG_TYPE,
                      "SI Fold Operands", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SIFoldOperands, DEBUG_TYPE,
                    "SI Fold Operands", false, false)

char SIFoldOperands::ID = 0;

char &llvm::SIFoldOperandsID = SIFoldOperands::ID;

FunctionPass *llvm::createSIFoldOperandsPass() {
  return new SIFoldOperands();
}

static bool isSafeToFold(unsigned Opcode) {
  switch(Opcode) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::COPY:
    return true;
  default:
    return false;
  }
}

static bool updateOperand(MachineInstr *MI, unsigned OpNo,
                          const MachineOperand &New,
                          const TargetRegisterInfo &TRI) {
  MachineOperand &Old = MI->getOperand(OpNo);
  assert(Old.isReg());

  if (New.isImm()) {
    Old.ChangeToImmediate(New.getImm());
    return true;
  }

  if (New.isFPImm()) {
    Old.ChangeToFPImmediate(New.getFPImm());
    return true;
  }

  if (New.isReg())  {
    if (TargetRegisterInfo::isVirtualRegister(Old.getReg()) &&
        TargetRegisterInfo::isVirtualRegister(New.getReg())) {
      Old.substVirtReg(New.getReg(), New.getSubReg(), TRI);
      return true;
    }
  }

  // FIXME: Handle physical registers.

  return false;
}

bool SIFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      if (!isSafeToFold(MI.getOpcode()))
        continue;

      MachineOperand &OpToFold = MI.getOperand(1);

      // FIXME: Fold operands with subregs.
      if (OpToFold.isReg() &&
          (!TargetRegisterInfo::isVirtualRegister(OpToFold.getReg()) ||
           OpToFold.getSubReg()))
        continue;

      std::vector<std::pair<MachineInstr *, unsigned>> FoldList;
      for (MachineRegisterInfo::use_iterator
           Use = MRI.use_begin(MI.getOperand(0).getReg()), E = MRI.use_end();
           Use != E; ++Use) {

        MachineInstr *UseMI = Use->getParent();
        const MachineOperand &UseOp = UseMI->getOperand(Use.getOperandNo());

        // FIXME: Fold operands with subregs.
        if (UseOp.isReg() && UseOp.getSubReg()) {
          continue;
        }

        // In order to fold immediates into copies, we need to change the
        // copy to a MOV.
        if ((OpToFold.isImm() || OpToFold.isFPImm()) &&
             UseMI->getOpcode() == AMDGPU::COPY) {
          const TargetRegisterClass *TRC =
              MRI.getRegClass(UseMI->getOperand(0).getReg());

          if (TRC->getSize() == 4) {
            if (TRI.isSGPRClass(TRC))
              UseMI->setDesc(TII->get(AMDGPU::S_MOV_B32));
            else
              UseMI->setDesc(TII->get(AMDGPU::V_MOV_B32_e32));
          } else if (TRC->getSize() == 8 && TRI.isSGPRClass(TRC)) {
            UseMI->setDesc(TII->get(AMDGPU::S_MOV_B64));
          } else {
            continue;
          }
        }

        const MCInstrDesc &UseDesc = UseMI->getDesc();

        // Don't fold into target independent nodes.  Target independent opcodes
        // don't have defined register classes.
        if (UseDesc.isVariadic() ||
            UseDesc.OpInfo[Use.getOperandNo()].RegClass == -1)
          continue;

        // Normal substitution
        if (TII->isOperandLegal(UseMI, Use.getOperandNo(), &OpToFold)) {
          FoldList.push_back(std::make_pair(UseMI, Use.getOperandNo()));
          continue;
        }

        // FIXME: We could commute the instruction to create more opportunites
        // for folding.  This will only be useful if we have 32-bit instructions.

        // FIXME: We could try to change the instruction from 64-bit to 32-bit
        // to enable more folding opportunites.  The shrink operands pass
        // already does this.
      }

      for (std::pair<MachineInstr *, unsigned> Fold : FoldList) {
        if (updateOperand(Fold.first, Fold.second, OpToFold, TRI)) {
          // Clear kill flags.
          if (OpToFold.isReg())
            OpToFold.setIsKill(false);
          DEBUG(dbgs() << "Folded source from " << MI << " into OpNo " <<
                Fold.second << " of " << *Fold.first << '\n');
        }
      }
    }
  }
  return false;
}
