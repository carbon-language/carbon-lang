//===- llvm/Transforms/PrintModulePass.h - Printing Pass ---------*- C++ -*--=//
//
// This file defines a simple pass to print out methods of a module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PRINTMODULE_H
#define LLVM_TRANSFORMS_PRINTMODULE_H

#include "llvm/Transforms/Pass.h"
#include "llvm/Assembly/Writer.h"

class PrintModulePass : public ConcretePass {
  string Banner;          // String to print before each method
  ostream *Out;           // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  inline PrintModulePass(const string &B, ostream *o = &cout, bool DS = false) 
    : Banner(B), Out(o), DeleteStream(DS) {}

  ~PrintModulePass() {
    if (DeleteStream) delete Out;
  }

  // doPerMethodWork - This pass just prints a banner followed by the method as
  // it's processed.
  //
  bool doPerMethodWorkVirt(Method *M) {
    (*Out) << Banner << M;
    return false;
  }
};

#endif
