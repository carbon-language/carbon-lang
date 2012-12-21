//===-- SILowerControlFlow.cpp - Use predicates for control flow ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass lowers the pseudo control flow instructions to real
/// machine instructions.
///
/// All control flow is handled using predicated instructions and
/// a predicate stack.  Each Scalar ALU controls the operations of 64 Vector
/// ALUs.  The Scalar ALU can update the predicate for any of the Vector ALUs
/// by writting to the 64-bit EXEC register (each bit corresponds to a
/// single vector ALU).  Typically, for predicates, a vector ALU will write
/// to its bit of the VCC register (like EXEC VCC is 64-bits, one for each
/// Vector ALU) and then the ScalarALU will AND the VCC register with the
/// EXEC to update the predicates.
///
/// For example:
/// %VCC = V_CMP_GT_F32 %VGPR1, %VGPR2
/// %SGPR0 = SI_IF %VCC
///   %VGPR0 = V_ADD_F32 %VGPR0, %VGPR0
/// %SGPR0 = SI_ELSE %SGPR0
///   %VGPR0 = V_SUB_F32 %VGPR0, %VGPR0
/// SI_END_CF %SGPR0
///
/// becomes:
///
/// %SGPR0 = S_AND_SAVEEXEC_B64 %VCC  // Save and update the exec mask
/// %SGPR0 = S_XOR_B64 %SGPR0, %EXEC  // Clear live bits from saved exec mask
/// S_CBRANCH_EXECZ label0            // This instruction is an optional
///                                   // optimization which allows us to
///                                   // branch if all the bits of
///                                   // EXEC are zero.
/// %VGPR0 = V_ADD_F32 %VGPR0, %VGPR0 // Do the IF block of the branch
///
/// label0:
/// %SGPR0 = S_OR_SAVEEXEC_B64 %EXEC   // Restore the exec mask for the Then block
/// %EXEC = S_XOR_B64 %SGPR0, %EXEC    // Clear live bits from saved exec mask
/// S_BRANCH_EXECZ label1              // Use our branch optimization
///                                    // instruction again.
/// %VGPR0 = V_SUB_F32 %VGPR0, %VGPR   // Do the THEN block
/// label1:
/// %EXEC = S_OR_B64 %EXEC, %SGPR0     // Re-enable saved exec mask bits
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {

class SILowerControlFlowPass : public MachineFunctionPass {

private:
  static const unsigned SkipThreshold = 12;

  static char ID;
  const TargetInstrInfo *TII;

  void Skip(MachineInstr &MI, MachineOperand &To);

  void If(MachineInstr &MI);
  void Else(MachineInstr &MI);
  void Break(MachineInstr &MI);
  void IfBreak(MachineInstr &MI);
  void ElseBreak(MachineInstr &MI);
  void Loop(MachineInstr &MI);
  void EndCf(MachineInstr &MI);

  void Branch(MachineInstr &MI);

public:
  SILowerControlFlowPass(TargetMachine &tm) :
    MachineFunctionPass(ID), TII(tm.getInstrInfo()) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const {
    return "SI Lower control flow instructions";
  }

};

} // End anonymous namespace

char SILowerControlFlowPass::ID = 0;

FunctionPass *llvm::createSILowerControlFlowPass(TargetMachine &tm) {
  return new SILowerControlFlowPass(tm);
}

void SILowerControlFlowPass::Skip(MachineInstr &From, MachineOperand &To) {
  unsigned NumInstr = 0;

  for (MachineBasicBlock *MBB = *From.getParent()->succ_begin();
       NumInstr < SkipThreshold && MBB != To.getMBB() && !MBB->succ_empty();
       MBB = *MBB->succ_begin()) {

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         NumInstr < SkipThreshold && I != E; ++I) {

      if (I->isBundle() || !I->isBundled())
        ++NumInstr;
    }
  }

  if (NumInstr < SkipThreshold)
    return;

  DebugLoc DL = From.getDebugLoc();
  BuildMI(*From.getParent(), &From, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
          .addOperand(To)
          .addReg(AMDGPU::EXEC);
}

void SILowerControlFlowPass::If(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();
  unsigned Vcc = MI.getOperand(1).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), Reg)
          .addReg(Vcc);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), Reg)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  Skip(MI, MI.getOperand(2));

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Else(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_SAVEEXEC_B64), Dst)
          .addReg(Src); // Saved EXEC

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Dst);

  Skip(MI, MI.getOperand(2));

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Break(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::IfBreak(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Vcc = MI.getOperand(1).getReg();
  unsigned Src = MI.getOperand(2).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(Vcc)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::ElseBreak(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Saved = MI.getOperand(1).getReg();
  unsigned Src = MI.getOperand(2).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(Saved)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Loop(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Src = MI.getOperand(0).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ANDN2_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
          .addOperand(MI.getOperand(1))
          .addReg(AMDGPU::EXEC);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::EndCf(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Branch(MachineInstr &MI) {
  MachineBasicBlock *Next = MI.getParent()->getNextNode();
  MachineBasicBlock *Target = MI.getOperand(0).getMBB();
  if (Target == Next)
    MI.eraseFromParent();
  else
    assert(0);
}

bool SILowerControlFlowPass::runOnMachineFunction(MachineFunction &MF) {
  bool HaveCf = false;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), Next = llvm::next(I);
         I != MBB.end(); I = Next) {

      Next = llvm::next(I);
      MachineInstr &MI = *I;
      switch (MI.getOpcode()) {
        default: break;
        case AMDGPU::SI_IF:
          If(MI);
          break;

        case AMDGPU::SI_ELSE:
          Else(MI);
          break;

        case AMDGPU::SI_BREAK:
          Break(MI);
          break;

        case AMDGPU::SI_IF_BREAK:
          IfBreak(MI);
          break;

        case AMDGPU::SI_ELSE_BREAK:
          ElseBreak(MI);
          break;

        case AMDGPU::SI_LOOP:
          Loop(MI);
          break;

        case AMDGPU::SI_END_CF:
          HaveCf = true;
          EndCf(MI);
          break;

        case AMDGPU::S_BRANCH:
          Branch(MI);
          break;
      }
    }
  }

  // TODO: What is this good for?
  unsigned ShaderType = MF.getInfo<SIMachineFunctionInfo>()->ShaderType;
  if (HaveCf && ShaderType == ShaderType::PIXEL) {
    for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
         BI != BE; ++BI) {

      MachineBasicBlock &MBB = *BI;
      if (MBB.succ_empty()) {

        MachineInstr &MI = *MBB.getFirstNonPHI();
        DebugLoc DL = MI.getDebugLoc();

        // If the exec mask is non-zero, skip the next two instructions
        BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
               .addImm(3)
               .addReg(AMDGPU::EXEC);

        // Exec mask is zero: Export to NULL target...
        BuildMI(MBB, &MI, DL, TII->get(AMDGPU::EXP))
                .addImm(0)
                .addImm(0x09) // V_008DFC_SQ_EXP_NULL
                .addImm(0)
                .addImm(1)
                .addImm(1)
                .addReg(AMDGPU::SREG_LIT_0)
                .addReg(AMDGPU::SREG_LIT_0)
                .addReg(AMDGPU::SREG_LIT_0)
                .addReg(AMDGPU::SREG_LIT_0);

        // ... and terminate wavefront
        BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ENDPGM));
      }
    }
  }

  return true;
}
