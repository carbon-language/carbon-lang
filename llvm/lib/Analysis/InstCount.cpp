//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// This pass collects the count of all instructions and reports them 
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"

namespace {
#define HANDLE_INST(N, OPCODE, CLASS) \
    Statistic<> Num##OPCODE##Inst("instcount", "Number of " #OPCODE " insts");

#include "llvm/Instruction.def"

  class InstCount : public Pass, public InstVisitor<InstCount> {
    friend class InstVisitor<InstCount>;

#define HANDLE_INST(N, OPCODE, CLASS) \
    void visit##OPCODE(CLASS &) { Num##OPCODE##Inst++; }

#include "llvm/Instruction.def"

    void visitInstruction(Instruction &I) {
      std::cerr << "Instruction Count does not know about " << I;
      abort();
    }
  public:
    virtual bool run(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
    virtual void print(std::ostream &O, Module *M) const {}

  };

  RegisterAnalysis<InstCount> X("instcount",
                                "Counts the various types of Instructions");
}

// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::run(Module &M) {
  visit(M);
  return false;
}
