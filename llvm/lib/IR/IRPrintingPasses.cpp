//===--- IRPrintingPasses.cpp - Module and Function printing passes -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PrintModulePass and PrintFunctionPass implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

class PrintModulePass : public ModulePass {
  raw_ostream &Out;
  std::string Banner;

public:
  static char ID;
  PrintModulePass() : ModulePass(ID), Out(dbgs()) {}
  PrintModulePass(raw_ostream &Out, const std::string &Banner)
      : ModulePass(ID), Out(Out), Banner(Banner) {}

  bool runOnModule(Module &M) {
    Out << Banner << M;
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

class PrintFunctionPass : public FunctionPass {
  raw_ostream &Out;
  std::string Banner;

public:
  static char ID;
  PrintFunctionPass() : FunctionPass(ID), Out(dbgs()) {}
  PrintFunctionPass(raw_ostream &Out, const std::string &Banner)
      : FunctionPass(ID), Out(Out), Banner(Banner) {}

  // This pass just prints a banner followed by the function as it's processed.
  bool runOnFunction(Function &F) {
    Out << Banner << static_cast<Value &>(F);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

class PrintBasicBlockPass : public BasicBlockPass {
  raw_ostream &Out;
  std::string Banner;

public:
  static char ID;
  PrintBasicBlockPass() : BasicBlockPass(ID), Out(dbgs()) {}
  PrintBasicBlockPass(raw_ostream &Out, const std::string &Banner)
      : BasicBlockPass(ID), Out(Out), Banner(Banner) {}

  bool runOnBasicBlock(BasicBlock &BB) {
    Out << Banner << BB;
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

}

char PrintModulePass::ID = 0;
INITIALIZE_PASS(PrintModulePass, "print-module", "Print module to stderr",
                false, false)
char PrintFunctionPass::ID = 0;
INITIALIZE_PASS(PrintFunctionPass, "print-function", "Print function to stderr",
                false, false)
char PrintBasicBlockPass::ID = 0;
INITIALIZE_PASS(PrintBasicBlockPass, "print-bb", "Print BB to stderr", false,
                false)

ModulePass *llvm::createPrintModulePass(llvm::raw_ostream &OS,
                                        const std::string &Banner) {
  return new PrintModulePass(OS, Banner);
}

FunctionPass *llvm::createPrintFunctionPass(llvm::raw_ostream &OS,
                                            const std::string &Banner) {
  return new PrintFunctionPass(OS, Banner);
}

BasicBlockPass *llvm::createPrintBasicBlockPass(llvm::raw_ostream &OS,
                                                const std::string &Banner) {
  return new PrintBasicBlockPass(OS, Banner);
}
