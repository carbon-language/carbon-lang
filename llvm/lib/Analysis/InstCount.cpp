//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// This pass collects the count of all instructions and reports them 
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> TotalInsts ("instcount", "Number of instructions (of all types)");
  Statistic<> TotalBlocks("instcount", "Number of basic blocks");
  Statistic<> TotalFuncs ("instcount", "Number of non-external functions");

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

  RegisterAnalysis<InstCount> X("instcount",
                                "Counts the various types of Instructions");
}

// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::runOnFunction(Function &F) {
  visit(F);
  return false;
}
