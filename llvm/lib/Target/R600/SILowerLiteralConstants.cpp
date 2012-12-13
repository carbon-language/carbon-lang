//===-- SILowerLiteralConstants.cpp - Lower intrs using literal constants--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass performs the following transformation on instructions with
/// literal constants:
///
/// %VGPR0 = V_MOV_IMM_I32 1
///
/// becomes:
///
/// BUNDLE
///   * %VGPR = V_MOV_B32_32 SI_LITERAL_CONSTANT
///   * SI_LOAD_LITERAL 1
///
/// The resulting sequence matches exactly how the hardware handles immediate
/// operands, so this transformation greatly simplifies the code generator.
///
/// Only the *_MOV_IMM_* support immediate operands at the moment, but when
/// support for immediate operands is added to other instructions, they
/// will be lowered here as well.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"

using namespace llvm;

namespace {

class SILowerLiteralConstantsPass : public MachineFunctionPass {

private:
  static char ID;
  const TargetInstrInfo *TII;

public:
  SILowerLiteralConstantsPass(TargetMachine &tm) :
    MachineFunctionPass(ID), TII(tm.getInstrInfo()) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const {
    return "SI Lower literal constants pass";
  }
};

} // End anonymous namespace

char SILowerLiteralConstantsPass::ID = 0;

FunctionPass *llvm::createSILowerLiteralConstantsPass(TargetMachine &tm) {
  return new SILowerLiteralConstantsPass(tm);
}

bool SILowerLiteralConstantsPass::runOnMachineFunction(MachineFunction &MF) {
  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), Next = llvm::next(I);
                               I != MBB.end(); I = Next) {
      Next = llvm::next(I);
      MachineInstr &MI = *I;
      switch (MI.getOpcode()) {
      default: break;
      case AMDGPU::S_MOV_IMM_I32:
      case AMDGPU::S_MOV_IMM_I64:
      case AMDGPU::V_MOV_IMM_F32:
      case AMDGPU::V_MOV_IMM_I32: {
          unsigned MovOpcode;
          unsigned LoadLiteralOpcode;
          MachineOperand LiteralOp = MI.getOperand(1);
          if (AMDGPU::VReg_32RegClass.contains(MI.getOperand(0).getReg())) {
            MovOpcode = AMDGPU::V_MOV_B32_e32;
          } else {
            MovOpcode = AMDGPU::S_MOV_B32;
          }
          if (LiteralOp.isImm()) {
            LoadLiteralOpcode = AMDGPU::SI_LOAD_LITERAL_I32;
          } else {
            LoadLiteralOpcode = AMDGPU::SI_LOAD_LITERAL_F32;
          }
          MIBundleBuilder Bundle(MBB, I);
          Bundle
            .append(BuildMI(MF, MBB.findDebugLoc(I), TII->get(MovOpcode),
                            MI.getOperand(0).getReg())
                    .addReg(AMDGPU::SI_LITERAL_CONSTANT))
            .append(BuildMI(MF, MBB.findDebugLoc(I),
                            TII->get(LoadLiteralOpcode))
                    .addOperand(MI.getOperand(1)));
          llvm::finalizeBundle(MBB, Bundle.begin());
          MI.eraseFromParent();
          break;
        }
      }
    }
  }
  return false;
}
