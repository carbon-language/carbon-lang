//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Function.h"

namespace {
  // Hello - The first implementation, without getAnalysisUsage.
  struct Hello : public FunctionPass {
    virtual bool runOnFunction(Function &F) {
      std::cerr << "Hello: " << F.getName() << "\n";
      return false;
    }
  }; 
  RegisterOpt<Hello> X("hello", "Hello World Pass");

  // Hello2 - The second implementation with getAnalysisUsage implemented.
  struct Hello2 : public FunctionPass {
    virtual bool runOnFunction(Function &F) {
      std::cerr << "Hello: " << F.getName() << "\n";
      return false;
    }

    // We don't modify the program, so we preserve all analyses
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    };
  }; 
  RegisterOpt<Hello2> Y("hello2", "Hello World Pass (with getAnalysisUsage implemented)");
}
