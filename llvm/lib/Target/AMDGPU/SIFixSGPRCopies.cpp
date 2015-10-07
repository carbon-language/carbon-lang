//===-- SIFixSGPRCopies.cpp - Remove potential VGPR => SGPR copies --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copies from VGPR to SGPR registers are illegal and the register coalescer
/// will sometimes generate these illegal copies in situations like this:
///
///  Register Class <vsrc> is the union of <vgpr> and <sgpr>
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///   %vreg1 <vsrc> = COPY %vreg0 <sgpr>
///    ...
///    BRANCH %cond BB1, BB2
///  BB1:
///    %vreg2 <vgpr> = VECTOR_INST
///    %vreg3 <vsrc> = COPY %vreg2 <vgpr>
///  BB2:
///    %vreg4 <vsrc> = PHI %vreg1 <vsrc>, <BB#0>, %vreg3 <vrsc>, <BB#1>
///    %vreg5 <vgpr> = VECTOR_INST %vreg4 <vsrc>
///
///
/// The coalescer will begin at BB0 and eliminate its copy, then the resulting
/// code will look like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <vsrc> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <vsrc>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now that the result of the PHI instruction is an SGPR, the register
/// allocator is now forced to constrain the register class of %vreg3 to
/// <sgpr> so we end up with final code like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <sgpr> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <sgpr>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now this code contains an illegal copy from a VGPR to an SGPR.
///
/// In order to avoid this problem, this pass searches for PHI instructions
/// which define a <vsrc> register and constrains its definition class to
/// <vgpr> if the user of the PHI's definition register is a vector instruction.
/// If the PHI's definition class is constrained to <vgpr> then the coalescer
/// will be unable to perform the COPY removal from the above example  which
/// ultimately led to the creation of an illegal COPY.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "sgpr-copies"

namespace {

class SIFixSGPRCopies : public MachineFunctionPass {

private:
  static char ID;
  bool isVGPRToSGPRCopy(const MachineInstr &Copy, const SIRegisterInfo *TRI,
                        const MachineRegisterInfo &MRI) const;

public:
  SIFixSGPRCopies(TargetMachine &tm) : MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fix SGPR copies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace

char SIFixSGPRCopies::ID = 0;

FunctionPass *llvm::createSIFixSGPRCopiesPass(TargetMachine &tm) {
  return new SIFixSGPRCopies(tm);
}

static bool hasVGPROperands(const MachineInstr &MI, const SIRegisterInfo *TRI) {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    if (!MI.getOperand(i).isReg() ||
        !TargetRegisterInfo::isVirtualRegister(MI.getOperand(i).getReg()))
      continue;

    if (TRI->hasVGPRs(MRI.getRegClass(MI.getOperand(i).getReg())))
      return true;
  }
  return false;
}

bool SIFixSGPRCopies::isVGPRToSGPRCopy(const MachineInstr &Copy,
                                      const SIRegisterInfo *TRI,
                                      const MachineRegisterInfo &MRI) const {

  unsigned DstReg = Copy.getOperand(0).getReg();
  unsigned SrcReg = Copy.getOperand(1).getReg();
  unsigned SrcSubReg = Copy.getOperand(1).getSubReg();

  if (!TargetRegisterInfo::isVirtualRegister(DstReg)) {
    // If the destination register is a physical register there isn't really
    // much we can do to fix this.
    return false;
  }

  if (!TargetRegisterInfo::isVirtualRegister(SrcReg))
    return false;

  const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);
  const TargetRegisterClass *SrcRC
    = TRI->getSubRegClass(MRI.getRegClass(SrcReg), SrcSubReg);
  return TRI->isSGPRClass(DstRC) && TRI->hasVGPRs(SrcRC);
}

bool SIFixSGPRCopies::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E; ++I) {
      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      default:
        continue;
      case AMDGPU::COPY: {
        if (isVGPRToSGPRCopy(MI, TRI, MRI)) {
          DEBUG(dbgs() << "Fixing VGPR -> SGPR copy: " << MI);
          TII->moveToVALU(MI);
        }

        break;
      }
      case AMDGPU::PHI: {
        DEBUG(dbgs() << "Fixing PHI: " << MI);
        unsigned Reg = MI.getOperand(0).getReg();
        if (!TRI->isSGPRClass(MRI.getRegClass(Reg)))
          break;

        // If a PHI node defines an SGPR and any of its operands are VGPRs,
        // then we need to move it to the VALU.
        //
        // Also, if a PHI node defines an SGPR and has all SGPR operands
        // we must move it to the VALU, because the SGPR operands will
        // all end up being assigned the same register, which means
        // there is a potential for a conflict if different threads take
        // different control flow paths.
        //
        // For Example:
        //
        // sgpr0 = def;
        // ...
        // sgpr1 = def;
        // ...
        // sgpr2 = PHI sgpr0, sgpr1
        // use sgpr2;
        //
        // Will Become:
        //
        // sgpr2 = def;
        // ...
        // sgpr2 = def;
        // ...
        // use sgpr2
        //
        // FIXME: This is OK if the branching decision is made based on an
        // SGPR value.
        bool SGPRBranch = false;

        // The one exception to this rule is when one of the operands
        // is defined by a SI_BREAK, SI_IF_BREAK, or SI_ELSE_BREAK
        // instruction.  In this case, there we know the program will
        // never enter the second block (the loop) without entering
        // the first block (where the condition is computed), so there
        // is no chance for values to be over-written.

        bool HasBreakDef = false;
        for (unsigned i = 1; i < MI.getNumOperands(); i+=2) {
          unsigned Reg = MI.getOperand(i).getReg();
          if (TRI->hasVGPRs(MRI.getRegClass(Reg))) {
            TII->moveToVALU(MI);
            break;
          }
          MachineInstr *DefInstr = MRI.getUniqueVRegDef(Reg);
          assert(DefInstr);
          switch(DefInstr->getOpcode()) {

          case AMDGPU::SI_BREAK:
          case AMDGPU::SI_IF_BREAK:
          case AMDGPU::SI_ELSE_BREAK:
          // If we see a PHI instruction that defines an SGPR, then that PHI
          // instruction has already been considered and should have
          // a *_BREAK as an operand.
          case AMDGPU::PHI:
            HasBreakDef = true;
            break;
          }
        }

        if (!SGPRBranch && !HasBreakDef)
          TII->moveToVALU(MI);
        break;
      }
      case AMDGPU::REG_SEQUENCE: {
        if (TRI->hasVGPRs(TII->getOpRegClass(MI, 0)) ||
            !hasVGPROperands(MI, TRI))
          continue;

        DEBUG(dbgs() << "Fixing REG_SEQUENCE: " << MI);

        TII->moveToVALU(MI);
        break;
      }
      case AMDGPU::INSERT_SUBREG: {
        const TargetRegisterClass *DstRC, *Src0RC, *Src1RC;
        DstRC = MRI.getRegClass(MI.getOperand(0).getReg());
        Src0RC = MRI.getRegClass(MI.getOperand(1).getReg());
        Src1RC = MRI.getRegClass(MI.getOperand(2).getReg());
        if (TRI->isSGPRClass(DstRC) &&
            (TRI->hasVGPRs(Src0RC) || TRI->hasVGPRs(Src1RC))) {
          DEBUG(dbgs() << " Fixing INSERT_SUBREG: " << MI);
          TII->moveToVALU(MI);
        }
        break;
      }
      }
    }
  }

  return true;
}
