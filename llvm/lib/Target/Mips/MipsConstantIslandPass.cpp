//===-- MipsConstantIslandPass.cpp - Emit Pc Relative loads----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// This pass is used to make Pc relative loads of constants.
// For now, only Mips16 will use this. While it has the same name and
// uses many ideas from the LLVM ARM Constant Island Pass, it's not intended
// to reuse any of the code from the ARM version.
//
// Loading constants inline is expensive on Mips16 and it's in general better
// to place the constant nearby in code space and then it can be loaded with a
// simple 16 bit load instruction.
//
// The constants can be not just numbers but addresses of functions and labels.
// This can be particularly helpful in static relocation mode for embedded
// non linux targets.
//
//

#define DEBUG_TYPE "mips-constant-islands"

#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

namespace {
  typedef MachineBasicBlock::iterator Iter;
  typedef MachineBasicBlock::reverse_iterator ReverseIter;

  class MipsConstantIslands : public MachineFunctionPass {

  public:
    static char ID;
    MipsConstantIslands(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm),
        IsPIC(TM.getRelocationModel() == Reloc::PIC_),
        ABI(TM.getSubtarget<MipsSubtarget>().getTargetABI()) {}

    virtual const char *getPassName() const {
      return "Mips Constant Islands";
    }

    bool runOnMachineFunction(MachineFunction &F);

  private:
    const TargetMachine &TM;
    bool IsPIC;
    unsigned ABI;
  };

  char MipsConstantIslands::ID = 0;
} // end of anonymous namespace

/// createMipsLongBranchPass - Returns a pass that converts branches to long
/// branches.
FunctionPass *llvm::createMipsConstantIslandPass(MipsTargetMachine &tm) {
  return new MipsConstantIslands(tm);
}

bool MipsConstantIslands::runOnMachineFunction(MachineFunction &F) {
  // The intention is for this to be a mips16 only pass for now
  // FIXME:
  // if (!TM.getSubtarget<MipsSubtarget>().inMips16Mode())
  //  return false;
  return false;
}

