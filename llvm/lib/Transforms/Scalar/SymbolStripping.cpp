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

static bool StripSymbolTable(SymbolTable &SymTab) {
  bool RemovedSymbol = false;

  for (SymbolTable::iterator I = SymTab.begin(); I != SymTab.end();) {
    // Removing items from the plane can cause the plane itself to get deleted.
    // If this happens, make sure we incremented our plane iterator already!
    std::map<const std::string, Value *> &Plane = (I++)->second;
    
    SymbolTable::type_iterator B = Plane.begin(), Bend = Plane.end();
    while (B != Bend) {   // Found nonempty type plane!
      Value *V = B->second;

      if (isa<Constant>(V) || isa<Type>(V)) {
	SymTab.type_remove(B++);
        RemovedSymbol = true;
      } else {
        ++B;
        if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasInternalLinkage()){
          // Set name to "", removing from symbol table!
          V->setName("", &SymTab);
          RemovedSymbol = true;
        }
      }
    }
  }
 
  return RemovedSymbol;
}

namespace {
  struct SymbolStripping : public FunctionPass {
    virtual bool runOnFunction(Function &F) {
      return StripSymbolTable(F.getSymbolTable());
    }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  RegisterOpt<SymbolStripping> X("strip", "Strip symbols from functions");

  struct FullSymbolStripping : public SymbolStripping {
    virtual bool doInitialization(Module &M) {
      return StripSymbolTable(M.getSymbolTable());
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
