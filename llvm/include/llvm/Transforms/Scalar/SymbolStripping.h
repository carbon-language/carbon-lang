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

namespace opt {

// DoSymbolStripping - Remove all symbolic information from a method
//
bool DoSymbolStripping(Method *M);

// DoSymbolStripping - Remove all symbolic information from all methods in a 
// module
//
static inline bool DoSymbolStripping(Module *M) { 
  return M->reduceApply(DoSymbolStripping); 
}

// DoFullSymbolStripping - Remove all symbolic information from all methods 
// in a module, and all module level symbols. (method names, etc...)
//
bool DoFullSymbolStripping(Module *M);

} // End namespace opt 
#endif
