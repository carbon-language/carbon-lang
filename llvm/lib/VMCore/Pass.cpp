//===- Pass.cpp - LLVM Pass Infrastructure Impementation ------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"
#include "PassManagerT.h"         // PassManagerT implementation
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "Support/STLExtras.h"
#include "Support/CommandLine.h"
#include <typeinfo>
#include <iostream>

// Source of unique analysis ID #'s.
unsigned AnalysisID::NextID = 0;

void AnalysisResolver::setAnalysisResolver(Pass *P, AnalysisResolver *AR) {
  assert(P->Resolver == 0 && "Pass already in a PassManager!");
  P->Resolver = AR;
}

//===----------------------------------------------------------------------===//
// PassManager implementation - The PassManager class is a simple Pimpl class
// that wraps the PassManagerT template.
//
PassManager::PassManager() : PM(new PassManagerT<Module>()) {}
PassManager::~PassManager() { delete PM; }
void PassManager::add(Pass *P) { PM->add(P); }
bool PassManager::run(Module *M) { return PM->run(M); }


//===----------------------------------------------------------------------===//
// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//

// Different debug levels that can be enabled...
enum PassDebugLevel {
  None, PassStructure, PassExecutions, PassDetails
};

static cl::Enum<enum PassDebugLevel> PassDebugging("debug-pass", cl::Hidden,
  "Print PassManager debugging information",
  clEnumVal(None          , "disable debug output"),
  clEnumVal(PassStructure , "print pass structure before run()"),
  clEnumVal(PassExecutions, "print pass name before it is executed"),
  clEnumVal(PassDetails   , "print pass details when it is executed"), 0); 

void PMDebug::PrintPassStructure(Pass *P) {
  if (PassDebugging >= PassStructure)
    P->dumpPassStructure();
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Annotable *V) {
  if (PassDebugging >= PassExecutions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '" 
              << typeid(*P).name();
    if (V) {
      std::cerr << "' on ";

      if (dynamic_cast<Module*>(V)) {
        std::cerr << "Module\n"; return;
      } else if (Function *F = dynamic_cast<Function*>(V))
        std::cerr << "Function '" << F->getName();
      else if (BasicBlock *BB = dynamic_cast<BasicBlock*>(V))
        std::cerr << "BasicBlock '" << BB->getName();
      else if (Value *Val = dynamic_cast<Value*>(V))
        std::cerr << typeid(*Val).name() << " '" << Val->getName();
    }
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintAnalysisSetInfo(unsigned Depth, const char *Msg,
                                   Pass *P, const std::vector<AnalysisID> &Set){
  if (PassDebugging >= PassDetails && !Set.empty()) {
    std::cerr << (void*)P << std::string(Depth*2+3, ' ') << Msg << " Analyses:";
    for (unsigned i = 0; i != Set.size(); ++i) {
      Pass *P = Set[i].createPass();   // Good thing this is just debug code...
      std::cerr << "  " << typeid(*P).name();
      delete P;
    }
    std::cerr << "\n";
  }
}

// dumpPassStructure - Implement the -debug-passes=PassStructure option
void Pass::dumpPassStructure(unsigned Offset = 0) {
  std::cerr << std::string(Offset*2, ' ') << typeid(*this).name() << "\n";
}


//===----------------------------------------------------------------------===//
// Pass Implementation
//

void Pass::addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

// run - On a module, we run this pass by initializing, runOnFunction'ing once
// for every function in the module, then by finalizing.
//
bool FunctionPass::run(Module *M) {
  bool Changed = doInitialization(M);
  
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!(*I)->isExternal())      // Passes are not run on external functions!
    Changed |= runOnFunction(*I);
  
  return Changed | doFinalization(M);
}

// run - On a function, we simply initialize, run the function, then finalize.
//
bool FunctionPass::run(Function *F) {
  if (F->isExternal()) return false;// Passes are not run on external functions!

  return doInitialization(F->getParent()) | runOnFunction(F)
       | doFinalization(F->getParent());
}

void FunctionPass::addToPassManager(PassManagerT<Module> *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void FunctionPass::addToPassManager(PassManagerT<Function> *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

// doesNotModifyCFG - This function should be called by our subclasses to
// implement the getAnalysisUsage virtual function, iff they do not:
//
//  1. Add or remove basic blocks from the function
//  2. Modify terminator instructions in any way.
//
// This function annotates the AnalysisUsage info object to say that analyses
// that only depend on the CFG are preserved by this pass.
//
void FunctionPass::doesNotModifyCFG(AnalysisUsage &Info) {

}


//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

// To run this pass on a function, we simply call runOnBasicBlock once for each
// function.
//
bool BasicBlockPass::runOnFunction(Function *F) {
  bool Changed = false;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    Changed |= runOnBasicBlock(*I);
  return Changed;
}

// To run directly on the basic block, we initialize, runOnBasicBlock, then
// finalize.
//
bool BasicBlockPass::run(BasicBlock *BB) {
  Module *M = BB->getParent()->getParent();
  return doInitialization(M) | runOnBasicBlock(BB) | doFinalization(M);
}

void BasicBlockPass::addToPassManager(PassManagerT<Function> *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void BasicBlockPass::addToPassManager(PassManagerT<BasicBlock> *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

