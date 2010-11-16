//===-- llvm/CodeGen/ExpandPseudos.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Expand Psuedo-instructions produced by ISel. These are usually to allow
// the expansion to contain control flow, such as a conditional move
// implemented with a conditional branch and a phi, or an atomic operation
// implemented with a loop.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "expand-pseudos"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  class ExpandPseudos : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    ExpandPseudos() : MachineFunctionPass(ID) {}

  private:
    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const {
      return "Expand CodeGen Pseudo-instructions";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
} // end anonymous namespace

char ExpandPseudos::ID = 0;
INITIALIZE_PASS_BEGIN(ExpandPseudos, "expand-pseudos",
                "Expand CodeGen Psueod-instructions", false, false)
INITIALIZE_PASS_END(ExpandPseudos, "expand-pseudos",
                "Expand CodeGen Psueod-instructions", false, false)

FunctionPass *llvm::createExpandPseudosPass() {
  return new ExpandPseudos();
}

bool ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const TargetLowering *TLI = MF.getTarget().getTargetLowering();

  // Iterate through each instruction in the function, looking for pseudos.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE; ) {
      MachineInstr *MI = MBBI++;

      // If MI is a pseudo, expand it.
      const TargetInstrDesc &TID = MI->getDesc();
      if (TID.usesCustomInsertionHook()) {
        Changed = true;
        MachineBasicBlock *NewMBB =
          TLI->EmitInstrWithCustomInserter(MI, MBB);
        // The expansion may involve new basic blocks.
        if (NewMBB != MBB) {
          MBB = NewMBB;
          I = NewMBB;
          MBBI = NewMBB->begin();
          MBBE = NewMBB->end();
        }
      }
    }
  }

  return Changed;
}
