//===- Pass.cpp - LLVM Pass Infrastructure Impementation ------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "Support/STLExtras.h"
#include <algorithm>

// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//
#ifdef NDEBUG
// If not debugging, remove the option
inline static void PrintPassInformation(const char *, Pass *, Value *) { }
#else

#include "Support/CommandLine.h"
#include <typeinfo>
#include <iostream>

// The option is hidden from --help by default
static cl::Flag PassDebugEnabled("debug-pass",
  "Print pass names as they are executed by the PassManager", cl::Hidden);

static void PrintPassInformation(const char *Action, Pass *P, Value *V) {
  if (PassDebugEnabled)
    std::cerr << Action << " Pass '" << typeid(*P).name() << "' on "
              << typeid(*V).name() << " '" << V->getName() << "'...\n";
}
#endif



PassManager::~PassManager() {
  for_each(Passes.begin(), Passes.end(), deleter<Pass>);
}

class BasicBlockPassBatcher : public MethodPass {
  typedef std::vector<BasicBlockPass*> SubPassesType;
  SubPassesType SubPasses;
public:
  ~BasicBlockPassBatcher() {
    for_each(SubPasses.begin(), SubPasses.end(), deleter<BasicBlockPass>);
  }

  void add(BasicBlockPass *P) { SubPasses.push_back(P); }

  virtual bool doInitialization(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I) {
      PrintPassInformation("Initializing", *I, M);
      Changed |= (*I)->doInitialization(M);
    }
    return Changed;
  }

  virtual bool runOnMethod(Method *M) {
    bool Changed = false;

    for (Method::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI)
      for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
           I != E; ++I) {
        PrintPassInformation("Executing", *I, *MI);
        Changed |= (*I)->runOnBasicBlock(*MI);
      }
    return Changed;
  }

  virtual bool doFinalization(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I) {
      PrintPassInformation("Finalizing", *I, M);
      Changed |= (*I)->doFinalization(M);
    }
    return Changed;
  }
};

class MethodPassBatcher : public Pass {
  typedef std::vector<MethodPass*> SubPassesType;
  SubPassesType SubPasses;
  BasicBlockPassBatcher *BBPBatcher;
public:
  inline MethodPassBatcher() : BBPBatcher(0) {}

  inline ~MethodPassBatcher() {
    for_each(SubPasses.begin(), SubPasses.end(), deleter<MethodPass>);
  }

  void add(BasicBlockPass *BBP) {
    if (BBPBatcher == 0) {
      BBPBatcher = new BasicBlockPassBatcher();
      SubPasses.push_back(BBPBatcher);
    }
    BBPBatcher->add(BBP);
  }

  void add(MethodPass *P) {
    if (BasicBlockPass *BBP = dynamic_cast<BasicBlockPass*>(P)) {
      add(BBP);
    } else {
      BBPBatcher = 0;  // Ensure that passes don't get accidentally reordered
      SubPasses.push_back(P);
    }
  }

  virtual bool run(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I) {
      PrintPassInformation("Initializing", *I, M);
      Changed |= (*I)->doInitialization(M);
    }

    for (Module::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI)
      for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
           I != E; ++I) {
        PrintPassInformation("Executing", *I, M);
        Changed |= (*I)->runOnMethod(*MI);
      }

    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I) {
      PrintPassInformation("Finalizing", *I, M);
      Changed |= (*I)->doFinalization(M);
    }
    return Changed;
  }
};

// add(BasicBlockPass*) - If we know it's a BasicBlockPass, we don't have to do
// any checking...
//
void PassManager::add(BasicBlockPass *BBP) {
  if (Batcher == 0)         // If we don't have a batcher yet, make one now.
    add((MethodPass*)BBP);
  else
    Batcher->add(BBP);
}


// add(MethodPass*) - MethodPass's must be batched together... make sure this
// happens now.
//
void PassManager::add(MethodPass *MP) {
  if (Batcher == 0) { // If we don't have a batcher yet, make one now.
    Batcher = new MethodPassBatcher();
    Passes.push_back(Batcher);
  }
  Batcher->add(MP);   // The Batcher will queue them passes up
}

// add - Add a pass to the PassManager, batching it up as appropriate...
void PassManager::add(Pass *P) {
  if (MethodPass *MP = dynamic_cast<MethodPass*>(P)) {
    add(MP);  // Use the methodpass specific code to do the addition
  } else {
    Batcher = 0;  // Ensure that passes don't get accidentally reordered
    Passes.push_back(P);
  }
}


bool PassManager::run(Module *M) {
  bool MadeChanges = false;
  // Run all of the pass initializers
  for (unsigned i = 0, e = Passes.size(); i < e; ++i) {
    PrintPassInformation("Executing", Passes[i], M);
    MadeChanges |= Passes[i]->run(M);
  }
  return MadeChanges;
}
