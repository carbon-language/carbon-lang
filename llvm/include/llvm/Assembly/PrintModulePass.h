//===- llvm/Assembly/PrintModulePass.h - Printing Pass -----------*- C++ -*--=//
//
// This file defines two passes to print out a module.  The PrintModulePass
// pass simply prints out the entire module when it is executed.  The
// PrintMethodPass class is designed to be pipelined with other MethodPass's,
// and prints out the methods of the class as they are processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include "llvm/Pass.h"
#include <iostream>

class PrintModulePass : public Pass {
  std::ostream *Out;      // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  inline PrintModulePass(std::ostream *o = &std::cout, bool DS = false)
    : Out(o), DeleteStream(DS) {
  }
  
  inline ~PrintModulePass() {
    if (DeleteStream) delete Out;
  }
  
  bool run(Module *M) {
    (*Out) << M;
    return false;
  }
};

class PrintFunctionPass : public MethodPass {
  std::string Banner;     // String to print before each method
  std::ostream *Out;      // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  inline PrintFunctionPass(const std::string &B, std::ostream *o = &std::cout,
                           bool DS = false)
    : Banner(B), Out(o), DeleteStream(DS) {
  }
  
  inline ~PrintFunctionPass() {
    if (DeleteStream) delete Out;
  }
  
  // runOnMethod - This pass just prints a banner followed by the method as
  // it's processed.
  //
  bool runOnMethod(Function *F) {
    (*Out) << Banner << F;
    return false;
  }
};

#endif
