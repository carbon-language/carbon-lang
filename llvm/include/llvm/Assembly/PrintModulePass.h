//===- llvm/Assembly/PrintModulePass.h - Printing Pass -----------*- C++ -*--=//
//
// This file defines a simple pass to print out methods of a module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include <iostream>

class PrintModulePass : public Pass {
  std::string Banner;     // String to print before each method
  std::ostream *Out;      // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
  bool PrintPerMethod;    // Print one method at a time rather than the whole?
public:
  inline PrintModulePass(const std::string &B, std::ostream *o = &std::cout,
                         bool DS = false,
                         bool printPerMethod = true)
    : Banner(B), Out(o), DeleteStream(DS), PrintPerMethod(printPerMethod) {
  }
  
  inline ~PrintModulePass() {
    if (DeleteStream) delete Out;
  }
  
  // doPerMethodWork - This pass just prints a banner followed by the method as
  // it's processed.
  //
  bool doPerMethodWork(Method *M) {
    if (PrintPerMethod)
      (*Out) << Banner << M;
    return false;
  }

  // doPassFinalization - Virtual method overriden by subclasses to do any post
  // processing needed after all passes have run.
  //
  bool doPassFinalization(Module *M) {
    if (! PrintPerMethod)
      (*Out) << Banner << M;
    return false;
  }
};

#endif
