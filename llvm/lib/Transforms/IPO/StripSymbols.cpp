//===- StripSymbols.cpp - Strip symbols and debug info from a module ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements stripping symbols out of symbol tables.
//
// Specifically, this allows you to strip all of the symbols out of:
//   * All functions in a module
//   * All non-essential symbols in a module (all function symbols + all module
//     scope symbols)
//   * Debug information.
//
// Notice that:
//   * This pass makes code much less readable, so it should only be used in
//     situations where the 'strip' utility would be used (such as reducing 
//     code size, and making it harder to reverse engineer code).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  class StripSymbols : public ModulePass {
    bool OnlyDebugInfo;
  public:
    StripSymbols(bool ODI = false) : OnlyDebugInfo(ODI) {}

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  RegisterOpt<StripSymbols> X("strip", "Strip all symbols from a module");
}

ModulePass *llvm::createStripSymbolsPass(bool OnlyDebugInfo) {
  return new StripSymbols(OnlyDebugInfo);
}


bool StripSymbols::runOnModule(Module &M) {
  // If we're not just stripping debug info, strip all symbols from the
  // functions and the names from any internal globals.
  if (!OnlyDebugInfo) {
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      if (I->hasInternalLinkage())
        I->setName("");     // Internal symbols can't participate in linkage

    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (I->hasInternalLinkage())
        I->setName("");     // Internal symbols can't participate in linkage
      I->getSymbolTable().strip();
    }
  }

  // FIXME: implement stripping of debug info.
  return true; 
}
