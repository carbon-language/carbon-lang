//===- AnalysisWrappers.cpp - Wrappers around non-pass analyses -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines pass wrappers around LLVM analyses that don't make sense to
// be passes.  It provides a nice standard pass interface to these classes so
// that they can be printed out by analyze.
//
// These classes are separated out of analyze.cpp so that it is more clear which
// code is the integral part of the analyze tool, and which part of the code is
// just making it so more passes are available.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/InstForest.h"

using namespace llvm;

namespace {
  struct InstForestHelper : public FunctionPass {
    Function *F;
    virtual bool runOnFunction(Function &Func) { F = &Func; return false; }

    void print(std::ostream &OS) const {
      std::cout << InstForest<char>(F);
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<InstForestHelper> P1("instforest", "InstForest Printer");
}
