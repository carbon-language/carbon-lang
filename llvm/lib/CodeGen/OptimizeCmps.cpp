//===-- OptimizeCmps.cpp - Optimize comparison instrs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs optimization of comparison instructions. For instance, in
// this code:
//
//     sub r1, 1
//     cmp r1, 0
//     bz  L1
//
// If the "sub" instruction all ready sets (or could be modified to set) the
// same flag that the "cmp" instruction sets and that "bz" uses, then we can
// eliminate the "cmp" instruction.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "opt-compares"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumEliminated, "Number of compares eliminated");

static cl::opt<bool>
EnableOptCmps("enable-optimize-cmps", cl::init(false), cl::Hidden);

namespace {
  class OptimizeCmps : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineRegisterInfo   *MRI;

    bool OptimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);

  public:
    static char ID; // Pass identification
    OptimizeCmps() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char OptimizeCmps::ID = 0;
INITIALIZE_PASS(OptimizeCmps, "opt-cmps",
                "Optimize comparison instrs", false, false);

FunctionPass *llvm::createOptimizeCmpsPass() { return new OptimizeCmps(); }

/// OptimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool OptimizeCmps::OptimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg;
  int CmpValue;
  if (!TII->isCompareInstr(MI, SrcReg, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg) ||
      CmpValue != 0)
    return false;

  MachineRegisterInfo::def_iterator DI = MRI->def_begin(SrcReg);
  if (llvm::next(DI) != MRI->def_end())
    // Only support one definition.
    return false;

  // Attempt to convert the defining instruction to set the "zero" flag.
  if (TII->convertToSetZeroFlag(&*DI, MI)) {
    ++NumEliminated;
    return true;
  }

  return false;
}

bool OptimizeCmps::runOnMachineFunction(MachineFunction &MF) {
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();

  if (!EnableOptCmps) return false;

  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (MachineBasicBlock::iterator
           MII = MBB->begin(), ME = MBB->end(); MII != ME; ) {
      MachineInstr *MI = &*MII++;
      Changed |= OptimizeCmpInstr(MI, MBB);
    }
  }

  return Changed;
}
