//===- Pass.cpp - LLVM Pass Infrastructure Impementation ------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "Support/STLExtras.h"
#include <algorithm>

// Source of unique analysis ID #'s.
unsigned AnalysisID::NextID = 0;

void AnalysisResolver::setAnalysisResolver(Pass *P, AnalysisResolver *AR) {
  assert(P->Resolver == 0 && "Pass already in a PassManager!");
  P->Resolver = AR;
}


// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//
#include "Support/CommandLine.h"
#include <typeinfo>
#include <iostream>

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
                                   Pass *P, Value *V) {
  if (PassDebugging >= PassExecutions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '" 
              << typeid(*P).name();
    if (V) {
      std::cerr << "' on ";
      switch (V->getValueType()) {
      case Value::ModuleVal:
        std::cerr << "Module\n"; return;
      case Value::FunctionVal:
        std::cerr << "Function '" << V->getName(); break;
      case Value::BasicBlockVal:
        std::cerr << "BasicBlock '" << V->getName(); break;
      default:
        std::cerr << typeid(*V).name() << " '" << V->getName(); break;
      }
    }
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintAnalysisSetInfo(unsigned Depth, const char *Msg,
                                   Pass *P, const Pass::AnalysisSet &Set) {
  if (PassDebugging >= PassDetails && !Set.empty()) {
    std::cerr << (void*)P << std::string(Depth*2+3, ' ') << Msg << " Analyses:";
    for (unsigned i = 0; i < Set.size(); ++i) {
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

void Pass::addToPassManager(PassManagerT<Module> *PM, AnalysisSet &Required,
                            AnalysisSet &Destroyed, AnalysisSet &Provided) {
  PM->addPass(this, Required, Destroyed, Provided);
}

//===----------------------------------------------------------------------===//
// MethodPass Implementation
//

// run - On a module, we run this pass by initializing, ronOnMethod'ing once
// for every method in the module, then by finalizing.
//
bool MethodPass::run(Module *M) {
  bool Changed = doInitialization(M);
  
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!(*I)->isExternal())      // Passes are not run on external methods!
    Changed |= runOnMethod(*I);
  
  return Changed | doFinalization(M);
}

// run - On a method, we simply initialize, run the method, then finalize.
//
bool MethodPass::run(Function *F) {
  if (F->isExternal()) return false;  // Passes are not run on external methods!

  return doInitialization(F->getParent()) | runOnMethod(F)
       | doFinalization(F->getParent());
}

void MethodPass::addToPassManager(PassManagerT<Module> *PM,
                                  AnalysisSet &Required, AnalysisSet &Destroyed,
                                  AnalysisSet &Provided) {
  PM->addPass(this, Required, Destroyed, Provided);
}

void MethodPass::addToPassManager(PassManagerT<Function> *PM,
                                  AnalysisSet &Required, AnalysisSet &Destroyed,
                                  AnalysisSet &Provided) {
  PM->addPass(this, Required, Destroyed, Provided);
}

//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

// To run this pass on a method, we simply call runOnBasicBlock once for each
// method.
//
bool BasicBlockPass::runOnMethod(Function *F) {
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
                                      AnalysisSet &Required,
                                      AnalysisSet &Destroyed,
                                      AnalysisSet &Provided) {
  PM->addPass(this, Required, Destroyed, Provided);
}

void BasicBlockPass::addToPassManager(PassManagerT<BasicBlock> *PM,
                                      AnalysisSet &Required,
                                      AnalysisSet &Destroyed,
                                      AnalysisSet &Provided) {
  PM->addPass(this, Required, Destroyed, Provided);
}

