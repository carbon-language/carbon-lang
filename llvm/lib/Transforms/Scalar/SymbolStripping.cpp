//===- SymbolStripping.cpp - Code to string symbols for methods and modules -=//
//
// This file implements stripping symbols out of symbol tables.
//
// Specifically, this allows you to strip all of the symbols out of:
//   * A method
//   * All methods in a module
//   * All symbols in a module (all method symbols + all module scope symbols)
//
// Notice that:
//   * This pass makes code much less readable, so it should only be used in
//     situations where the 'strip' utility would be used (such as reducing 
//     code size, and making it harder to reverse engineer code).
//
//===----------------------------------------------------------------------===//

#include "llvm/Optimizations/AllOpts.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"

static bool StripSymbolTable(SymbolTable *SymTab) {
  if (SymTab == 0) return false;    // No symbol table?  No problem.
  bool RemovedSymbol = false;

  for (SymbolTable::iterator I = SymTab->begin(); I != SymTab->end(); ++I) {
    std::map<const std::string, Value *> &Plane = I->second;
    
    SymbolTable::type_iterator B;
    while ((B = Plane.begin()) != Plane.end()) {   // Found nonempty type plane!
      Value *V = B->second;
      if (isa<Constant>(V) || isa<Type>(V))
	SymTab->type_remove(B);
      else 
	V->setName("", SymTab);   // Set name to "", removing from symbol table!
      RemovedSymbol = true;
      assert(Plane.begin() != B && "Symbol not removed from table!");
    }
  }
 
  return RemovedSymbol;
}


// DoSymbolStripping - Remove all symbolic information from a method
//
bool opt::SymbolStripping::doSymbolStripping(Method *M) {
  return StripSymbolTable(M->getSymbolTable());
}

// doStripGlobalSymbols - Remove all symbolic information from all methods 
// in a module, and all module level symbols. (method names, etc...)
//
bool opt::FullSymbolStripping::doStripGlobalSymbols(Module *M) {
  // Remove all symbols from methods in this module... and then strip all of the
  // symbols in this module...
  //  
  return StripSymbolTable(M->getSymbolTable());
}
