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
//   * A function
//   * All functions in a module
//   * All symbols in a module (all function symbols + all module scope symbols)
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

namespace llvm {

static bool StripSymbolTable(SymbolTable &SymTab) {
  bool RemovedSymbol = false;

  for (SymbolTable::iterator I = SymTab.begin(); I != SymTab.end(); ++I) {
    std::map<const std::string, Value *> &Plane = I->second;
    
    SymbolTable::type_iterator B;
    while ((B = Plane.begin()) != Plane.end()) {   // Found nonempty type plane!
      Value *V = B->second;
      if (isa<Constant>(V) || isa<Type>(V))
	SymTab.type_remove(B);
      else 
	V->setName("", &SymTab);  // Set name to "", removing from symbol table!
      RemovedSymbol = true;
      assert(Plane.begin() != B && "Symbol not removed from table!");
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

Pass *createSymbolStrippingPass() {
  return new SymbolStripping();
}

Pass *createFullSymbolStrippingPass() {
  return new FullSymbolStripping();
}

} // End llvm namespace
