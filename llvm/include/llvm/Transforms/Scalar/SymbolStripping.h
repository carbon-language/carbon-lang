//===-- SymbolStripping.h - Functions that Strip Symbol Tables ---*- C++ -*--=//
//
// This family of functions removes symbols from the symbol tables of methods
// and classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SYMBOL_STRIPPING_H
#define LLVM_OPT_SYMBOL_STRIPPING_H

#include "llvm/Pass.h"

struct SymbolStripping : public MethodPass {
  // doSymbolStripping - Remove all symbolic information from a method
  //
  static bool doSymbolStripping(Method *M);

  virtual bool runOnMethod(Method *M) {
    return doSymbolStripping(M);
  }
};

struct FullSymbolStripping : public MethodPass {
  
  // doStripGlobalSymbols - Remove all symbolic information from all methods 
  // in a module, and all module level symbols. (method names, etc...)
  //
  static bool doStripGlobalSymbols(Module *M);

  virtual bool doInitialization(Module *M) {
    return doStripGlobalSymbols(M);
  }

  virtual bool runOnMethod(Method *M) {
    return SymbolStripping::doSymbolStripping(M);
  }
};

#endif
