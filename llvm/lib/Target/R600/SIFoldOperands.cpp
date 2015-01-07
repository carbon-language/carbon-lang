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

struct FoldCandidate {
  MachineInstr *UseMI;
  unsigned UseOpNo;
  MachineOperand *OpToFold;
  uint64_t ImmToFold;

  FoldCandidate(MachineInstr *MI, unsigned OpNo, MachineOperand *FoldOp) :
      UseMI(MI), UseOpNo(OpNo), OpToFold(FoldOp), ImmToFold(0) { }

  FoldCandidate(MachineInstr *MI, unsigned OpNo, uint64_t Imm) :
      UseMI(MI), UseOpNo(OpNo), OpToFold(nullptr), ImmToFold(Imm) { }

  bool isImm() const {
    return !OpToFold;
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
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::COPY:
    return true;
  default:
    return false;
  }
}

static bool updateOperand(FoldCandidate &Fold,
                          const TargetRegisterInfo &TRI) {
  MachineInstr *MI = Fold.UseMI;
  MachineOperand &Old = MI->getOperand(Fold.UseOpNo);
  assert(Old.isReg());

  if (Fold.isImm()) {
    Old.ChangeToImmediate(Fold.ImmToFold);
    return true;
  }

  MachineOperand *New = Fold.OpToFold;
  if (TargetRegisterInfo::isVirtualRegister(Old.getReg()) &&
      TargetRegisterInfo::isVirtualRegister(New->getReg())) {
    Old.substVirtReg(New->getReg(), New->getSubReg(), TRI);
    return true;
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
      bool FoldingImm = OpToFold.isImm() || OpToFold.isFPImm();

      // Folding immediates with more than one use will increase program side.
      // FIXME: This will also reduce register usage, which may be better
      // in some cases.  A better heuristic is needed.
      if (FoldingImm && !TII->isInlineConstant(OpToFold) &&
          !MRI.hasOneUse(MI.getOperand(0).getReg()))
        continue;

      // FIXME: Fold operands with subregs.
      if (OpToFold.isReg() &&
          (!TargetRegisterInfo::isVirtualRegister(OpToFold.getReg()) ||
           OpToFold.getSubReg()))
        continue;

      std::vector<FoldCandidate> FoldList;
      for (MachineRegisterInfo::use_iterator
           Use = MRI.use_begin(MI.getOperand(0).getReg()), E = MRI.use_end();
           Use != E; ++Use) {

        MachineInstr *UseMI = Use->getParent();
        const MachineOperand &UseOp = UseMI->getOperand(Use.getOperandNo());

        // FIXME: Fold operands with subregs.
        if (UseOp.isReg() && UseOp.getSubReg() && OpToFold.isReg()) {
          continue;
        }

        APInt Imm;

        if (FoldingImm) {
          const TargetRegisterClass *UseRC = MRI.getRegClass(UseOp.getReg());

          if (OpToFold.isFPImm()) {
            Imm = OpToFold.getFPImm()->getValueAPF().bitcastToAPInt();
          } else {
            Imm = APInt(64, OpToFold.getImm());
          }

          // Split 64-bit constants into 32-bits for folding.
          if (UseOp.getSubReg()) {
            if (UseRC->getSize() != 8)
              continue;

            if (UseOp.getSubReg() == AMDGPU::sub0) {
              Imm = Imm.getLoBits(32);
            } else {
              assert(UseOp.getSubReg() == AMDGPU::sub1);
              Imm = Imm.getHiBits(32);
            }
          }

          // In order to fold immediates into copies, we need to change the
          // copy to a MOV.
          if (UseMI->getOpcode() == AMDGPU::COPY) {
            unsigned MovOp = TII->getMovOpcode(
                MRI.getRegClass(UseMI->getOperand(0).getReg()));
            if (MovOp == AMDGPU::COPY)
              continue;

            UseMI->setDesc(TII->get(MovOp));
          }
        }

        const MCInstrDesc &UseDesc = UseMI->getDesc();

        // Don't fold into target independent nodes.  Target independent opcodes
        // don't have defined register classes.
        if (UseDesc.isVariadic() ||
            UseDesc.OpInfo[Use.getOperandNo()].RegClass == -1)
          continue;


        if (FoldingImm) {
          const MachineOperand ImmOp = MachineOperand::CreateImm(Imm.getSExtValue());
          if (TII->isOperandLegal(UseMI, Use.getOperandNo(), &ImmOp)) {
            FoldList.push_back(FoldCandidate(UseMI, Use.getOperandNo(),
                               Imm.getSExtValue()));
          }
          continue;
        }

        // Normal substitution with registers
        if (TII->isOperandLegal(UseMI, Use.getOperandNo(), &OpToFold)) {
          FoldList.push_back(FoldCandidate(UseMI, Use.getOperandNo(), &OpToFold));
          continue;
        }

        // FIXME: We could commute the instruction to create more opportunites
        // for folding.  This will only be useful if we have 32-bit instructions.

        // FIXME: We could try to change the instruction from 64-bit to 32-bit
        // to enable more folding opportunites.  The shrink operands pass
        // already does this.
      }

      for (FoldCandidate &Fold : FoldList) {
        if (updateOperand(Fold, TRI)) {
          // Clear kill flags.
          if (!Fold.isImm()) {
            assert(Fold.OpToFold && Fold.OpToFold->isReg());
            Fold.OpToFold->setIsKill(false);
          }
          DEBUG(dbgs() << "Folded source from " << MI << " into OpNo " <<
                Fold.UseOpNo << " of " << *Fold.UseMI << '\n');
        }
      }
    }
  }
  return false;
}
