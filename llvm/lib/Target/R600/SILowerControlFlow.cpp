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
/// \brief This pass lowers the pseudo control flow instructions (SI_IF_NZ, ELSE, ENDIF)
/// to predicated instructions.
///
/// All control flow (except loops) is handled using predicated instructions and
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
/// SI_IF_NZ %VCC
///   %VGPR0 = V_ADD_F32 %VGPR0, %VGPR0
/// ELSE
///   %VGPR0 = V_SUB_F32 %VGPR0, %VGPR0
/// ENDIF
///
/// becomes:
///
/// %SGPR0 = S_AND_SAVEEXEC_B64 %VCC  // Save and update the exec mask
/// %SGPR0 = S_XOR_B64 %SGPR0, %EXEC  // Clear live bits from saved exec mask
/// S_CBRANCH_EXECZ label0            // This instruction is an
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
/// %EXEC = S_OR_B64 %EXEC, %SGPR2     // Re-enable saved exec mask bits
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
  static char ID;
  const TargetInstrInfo *TII;
  std::vector<unsigned> PredicateStack;
  std::vector<unsigned> UnusedRegisters;

  unsigned allocReg();
  void freeReg(unsigned Reg);

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

bool SILowerControlFlowPass::runOnMachineFunction(MachineFunction &MF) {

  // Find all the unused registers that can be used for the predicate stack.
  for (TargetRegisterClass::iterator I = AMDGPU::SReg_64RegClass.begin(),
                                     S = AMDGPU::SReg_64RegClass.end();
                                     I != S; ++I) {
    unsigned Reg = *I;
    if (!MF.getRegInfo().isPhysRegUsed(Reg)) {
      UnusedRegisters.insert(UnusedRegisters.begin(), Reg);
    }
  }

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    for (MachineBasicBlock::iterator I = MBB.begin(), Next = llvm::next(I);
                               I != MBB.end(); I = Next) {
      Next = llvm::next(I);
      MachineInstr &MI = *I;
      unsigned Reg;
      switch (MI.getOpcode()) {
        default: break;
        case AMDGPU::SI_IF_NZ:
          Reg = allocReg();
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_AND_SAVEEXEC_B64),
                  Reg)
                  .addOperand(MI.getOperand(0)); // VCC
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_XOR_B64),
                  Reg)
                  .addReg(Reg)
                  .addReg(AMDGPU::EXEC);
          MI.eraseFromParent();
          PredicateStack.push_back(Reg);
          break;

        case AMDGPU::ELSE:
          Reg = PredicateStack.back();
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_OR_SAVEEXEC_B64),
                  Reg)
                  .addReg(Reg);
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_XOR_B64),
                  AMDGPU::EXEC)
                  .addReg(Reg)
                  .addReg(AMDGPU::EXEC);
          MI.eraseFromParent();
          break;

        case AMDGPU::ENDIF:
          Reg = PredicateStack.back();
          PredicateStack.pop_back();
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_OR_B64),
                  AMDGPU::EXEC)
                  .addReg(AMDGPU::EXEC)
                  .addReg(Reg);
          freeReg(Reg);

          if (MF.getInfo<SIMachineFunctionInfo>()->ShaderType == ShaderType::PIXEL &&
              PredicateStack.empty()) {
            // If the exec mask is non-zero, skip the next two instructions
            BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_CBRANCH_EXECNZ))
                    .addImm(3)
                    .addReg(AMDGPU::EXEC);

            // Exec mask is zero: Export to NULL target...
            BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::EXP))
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
            BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::S_ENDPGM));
          }
          MI.eraseFromParent();
          break;
      }
    }
  }
  return true;
}

unsigned SILowerControlFlowPass::allocReg() {

  assert(!UnusedRegisters.empty() && "Ran out of registers for predicate stack");
  unsigned Reg = UnusedRegisters.back();
  UnusedRegisters.pop_back();
  return Reg;
}

void SILowerControlFlowPass::freeReg(unsigned Reg) {

  UnusedRegisters.push_back(Reg);
}
