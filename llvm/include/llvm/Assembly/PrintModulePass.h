//===- llvm/Assembly/PrintModulePass.h - Printing Pass ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two passes to print out a module.  The PrintModulePass pass
// simply prints out the entire module when it is executed.  The
// PrintFunctionPass class is designed to be pipelined with other
// FunctionPass's, and prints out the functions of the module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/Streams.h"

namespace llvm {

class PrintModulePass : public ModulePass {
  OStream *Out;           // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  static char ID;
  PrintModulePass() : ModulePass(intptr_t(&ID)), Out(&cerr), 
                      DeleteStream(false) {}
  PrintModulePass(OStream *o, bool DS = false)
    : ModulePass(intptr_t(&ID)), Out(o), DeleteStream(DS) {}

  ~PrintModulePass() {
    if (DeleteStream) delete Out;
  }

  bool runOnModule(Module &M) {
    (*Out) << M << std::flush;
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

class PrintFunctionPass : public FunctionPass {
  std::string Banner;     // String to print before each function
  OStream *Out;           // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  static char ID;
  PrintFunctionPass() : FunctionPass(intptr_t(&ID)), Banner(""), Out(&cerr), 
                        DeleteStream(false) {}
  PrintFunctionPass(const std::string &B, OStream *o = &cout,
                    bool DS = false)
    : FunctionPass(intptr_t(&ID)), Banner(B), Out(o), DeleteStream(DS) {}

  inline ~PrintFunctionPass() {
    if (DeleteStream) delete Out;
  }

  // runOnFunction - This pass just prints a banner followed by the function as
  // it's processed.
  //
  bool runOnFunction(Function &F) {
    (*Out) << Banner << static_cast<Value&>(F);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

} // End llvm namespace

#endif
