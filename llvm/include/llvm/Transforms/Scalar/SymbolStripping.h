//===-- SymbolStripping.h - Functions that Strip Symbol Tables ---*- C++ -*--=//
//
// This family of functions removes symbols from the symbol tables of methods
// and classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SYMBOL_STRIPPING_H
#define LLVM_OPT_SYMBOL_STRIPPING_H

class Method;
class Module;
#include "llvm/Transforms/Pass.h"

namespace opt {

struct SymbolStripping : public Pass {
  // doSymbolStripping - Remove all symbolic information from a method
  //
  static bool doSymbolStripping(Method *M);

  virtual bool doPerMethodWork(Method *M) {
    return doSymbolStripping(M);
  }
};

struct FullSymbolStripping : public Pass {
  
  // doStripGlobalSymbols - Remove all symbolic information from all methods 
  // in a module, and all module level symbols. (method names, etc...)
  //
  static bool doStripGlobalSymbols(Module *M);

  virtual bool doPassInitialization(Module *M) {
    return doStripGlobalSymbols(M);
  }

  virtual bool doPerMethodWork(Method *M) {
    return SymbolStripping::doSymbolStripping(M);
  }
};

} // End namespace opt 
#endif
