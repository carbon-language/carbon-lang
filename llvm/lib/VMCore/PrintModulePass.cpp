//===--- VMCore/PrintModulePass.cpp - Module/Function Printer -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PrintModulePass and PrintFunctionPass implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/PrintModulePass.h"

#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {

  class VISIBILITY_HIDDEN PrintModulePass : public ModulePass {
    OStream *Out;           // ostream to print on
    bool DeleteStream;      // Delete the ostream in our dtor?
  public:
    static char ID;
    PrintModulePass() : ModulePass(intptr_t(&ID)), Out(&cerr), 
      DeleteStream(false) {}
    PrintModulePass(OStream *o, bool DS)
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
    PrintFunctionPass(const std::string &B, OStream *o, bool DS)
      : FunctionPass(intptr_t(&ID)), Banner(B), Out(o), DeleteStream(DS) {}
    
    inline ~PrintFunctionPass() {
      if (DeleteStream) delete Out;
    }
    
    // runOnFunction - This pass just prints a banner followed by the
    // function as it's processed.
    //
    bool runOnFunction(Function &F) {
      (*Out) << Banner << static_cast<Value&>(F);
      return false;
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char PrintModulePass::ID = 0;
static RegisterPass<PrintModulePass>
X("print-module", "Print module to stderr");
char PrintFunctionPass::ID = 0;
static RegisterPass<PrintFunctionPass>
Y("print-function","Print function to stderr");

/// createPrintModulePass - Create and return a pass that writes the
/// module to the specified OStream.
ModulePass *llvm::createPrintModulePass(llvm::OStream *OS, bool DeleteStream) {
  return new PrintModulePass(OS, DeleteStream);
}

/// createPrintFunctionPass - Create and return a pass that prints
/// functions to the specified OStream as they are processed.
FunctionPass *llvm::createPrintFunctionPass(const std::string &Banner,
                                      OStream *OS, bool DeleteStream) {
  return new PrintFunctionPass(Banner, OS, DeleteStream);
}

