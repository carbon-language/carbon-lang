//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> TotalInsts ("instcount", "Number of instructions (of all types)");
  Statistic<> TotalBlocks("instcount", "Number of basic blocks");
  Statistic<> TotalFuncs ("instcount", "Number of non-external functions");
  Statistic<> TotalMemInst("instcount", "Number of memory instructions");

#define HANDLE_INST(N, OPCODE, CLASS) \
    Statistic<> Num##OPCODE##Inst("instcount", "Number of " #OPCODE " insts");

#include "llvm/Instruction.def"

  class InstCount : public FunctionPass, public InstVisitor<InstCount> {
    friend class InstVisitor<InstCount>;

    void visitFunction  (Function &F) { ++TotalFuncs; }
    void visitBasicBlock(BasicBlock &BB) { ++TotalBlocks; }

#define HANDLE_INST(N, OPCODE, CLASS) \
    void visit##OPCODE(CLASS &) { ++Num##OPCODE##Inst; ++TotalInsts; }

#include "llvm/Instruction.def"

    void visitInstruction(Instruction &I) {
      std::cerr << "Instruction Count does not know about " << I;
      abort();
    }
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
    virtual void print(std::ostream &O, const Module *M) const {}

  };

  RegisterPass<InstCount> X("instcount",
                            "Counts the various types of Instructions");
}

FunctionPass *llvm::createInstCountPass() { return new InstCount(); }

// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::runOnFunction(Function &F) {
  unsigned StartMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst + NumMallocInst + NumFreeInst;
  visit(F);
  unsigned EndMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst + NumMallocInst + NumFreeInst;
  TotalMemInst += EndMemInsts-StartMemInsts;
  return false;
}
