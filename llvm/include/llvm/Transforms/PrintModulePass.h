//===- llvm/Transforms/PrintModulePass.h - Printing Pass ---------*- C++ -*--=//
//
// This file defines a simple pass to print out methods of a module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PRINTMODULE_H
#define LLVM_TRANSFORMS_PRINTMODULE_H

#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Bytecode/Writer.h"

class PrintModulePass : public Pass {
  string Banner;          // String to print before each method
  ostream *Out;           // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
  bool PrintPerMethod;    // Print one method at a time rather than the whole?
public:
  inline PrintModulePass(const string &B, ostream *o = &cout,
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

class WriteModuleBytecode : public Pass {
  ostream *Out;           // ostream to print on
  bool DeleteStream;
public:
  inline WriteModuleBytecode(ostream *o = &cout, bool DS = false)
    : Out(o), DeleteStream(DS) {
  }

  inline ~WriteModuleBytecode() {
    if (DeleteStream) delete Out;
  }
  
  bool doPassFinalization(Module *M) {
    WriteBytecodeToFile(M, *Out);    
    return false;
  }
};

#endif
