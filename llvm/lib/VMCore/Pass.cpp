//===- Pass.cpp - LLVM Pass Infrastructure Impementation ------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "Support/STLExtras.h"

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

  virtual bool doPassInitialization(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I)
      Changed |= (*I)->doInitialization(M);
    return Changed;
  }

  virtual bool runOnMethod(Method *M) {
    bool Changed = false;

    for (Method::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI)
      for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
           I != E; ++I)
        Changed |= (*I)->runOnBasicBlock(*MI);
    return Changed;
  }

  virtual bool doFinalization(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I)
      Changed |= (*I)->doFinalization(M);

    return Changed;
  }
};

class MethodPassBatcher : public Pass {
  typedef std::vector<MethodPass*> SubPassesType;
  SubPassesType SubPasses;
  BasicBlockPassBatcher *BBPBatcher;
public:
  ~MethodPassBatcher() {
    for_each(SubPasses.begin(), SubPasses.end(), deleter<MethodPass>);
  }

  void add(MethodPass *P) {
    if (BasicBlockPass *BBP = dynamic_cast<BasicBlockPass*>(P)) {
      if (BBPBatcher == 0) {
        BBPBatcher = new BasicBlockPassBatcher();
        SubPasses.push_back(BBPBatcher);
      }
      BBPBatcher->add(BBP);
    } else {
      BBPBatcher = 0;  // Ensure that passes don't get accidentally reordered
      SubPasses.push_back(P);
    }
  }

  virtual bool run(Module *M) {
    bool Changed = false;
    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I)
      Changed |= (*I)->doInitialization(M);

    for (Module::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI)
      for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
           I != E; ++I)
        Changed |= (*I)->runOnMethod(*MI);

    for (SubPassesType::iterator I = SubPasses.begin(), E = SubPasses.end();
         I != E; ++I)
      Changed |= (*I)->doFinalization(M);

    return Changed;
  }
};




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
