//===- SymbolStripping.cpp - Strip symbols for functions and modules ------===//
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
//
// Notice that:
//   * This pass makes code much less readable, so it should only be used in
//     situations where the 'strip' utility would be used (such as reducing 
//     code size, and making it harder to reverse engineer code).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  struct SymbolStripping : public FunctionPass {
    virtual bool runOnFunction(Function &F) {
      return F.getSymbolTable().strip();
    }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  RegisterOpt<SymbolStripping> X("strip", "Strip symbols from functions");

  struct FullSymbolStripping : public SymbolStripping {
    virtual bool doInitialization(Module &M) {
      return M.getSymbolTable().strip();
    }
  };
  RegisterOpt<FullSymbolStripping> Y("mstrip",
                                     "Strip symbols from module and functions");
}

Pass *llvm::createSymbolStrippingPass() {
  return new SymbolStripping();
}

Pass *llvm::createFullSymbolStrippingPass() {
  return new FullSymbolStripping();
}
