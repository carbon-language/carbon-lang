//===-- SymbolStripping.h - Functions that Strip Symbol Tables ---*- C++ -*--=//
//
// This family of functions removes symbols from the symbol tables of methods
// and classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SYMBOL_STRIPPING_H
#define LLVM_TRANSFORMS_SYMBOL_STRIPPING_H

class Pass;

Pass *createSymbolStrippingPass();
Pass *createFullSymbolStrippingPass();

#endif
