//===-- PTXRegAlloc.cpp - PTX Register Allocator --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a register allocator for PTX code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ptx-reg-alloc"

#include "PTX.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegAllocRegistry.h"

using namespace llvm;

namespace {
  // Special register allocator for PTX.
  class PTXRegAlloc : public MachineFunctionPass {
  public:
    static char ID;
    PTXRegAlloc() : MachineFunctionPass(ID) {
      initializePHIEliminationPass(*PassRegistry::getPassRegistry());
      initializeTwoAddressInstructionPassPass(*PassRegistry::getPassRegistry());
    }

    virtual const char* getPassName() const {
      return "PTX Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(PHIEliminationID);
      AU.addRequiredID(TwoAddressInstructionPassID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      // We do not actually do anything (at least not yet).
      return false;
    }
  };

  char PTXRegAlloc::ID = 0;

  static RegisterRegAlloc
    ptxRegAlloc("ptx", "PTX register allocator", createPTXRegisterAllocator);
}

FunctionPass *llvm::createPTXRegisterAllocator() {
  return new PTXRegAlloc();
}

