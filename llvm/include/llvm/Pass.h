//===- llvm/Pass.h - Base class for XForm Passes -----------------*- C++ -*--=//
//
// This file defines a base class that indicates that a specified class is a
// transformation pass implementation.
//
// Pass's are designed this way so that it is possible to run passes in a cache
// and organizationally optimal order without having to specify it at the front
// end.  This allows arbitrary passes to be strung together and have them
// executed as effeciently as possible.
//
// Passes should extend one of the classes below, depending on the guarantees
// that it can make about what will be modified as it is run.  For example, most
// global optimizations should derive from MethodPass, because they do not add
// or delete methods, they operate on the internals of the method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_H
#define LLVM_PASS_H

#include "llvm/Module.h"
#include "llvm/Method.h"

class MethodPassBatcher;

//===----------------------------------------------------------------------===//
// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
// interprocedural optimization or you do not fit into any of the more
// constrained passes described below.
//
struct Pass {
  // Destructor - Virtual so we can be subclassed
  inline virtual ~Pass() {}

  virtual bool run(Module *M) = 0;
};


//===----------------------------------------------------------------------===//
// MethodPass class - This class is used to implement most global optimizations.
// Optimizations should subclass this class if they meet the following
// constraints:
//  1. Optimizations are organized globally, ie a method at a time
//  2. Optimizing a method does not cause the addition or removal of any methods
//     in the module
//
struct MethodPass : public Pass {
  // doInitialization - Virtual method overridden by subclasses to do
  // any neccesary per-module initialization.
  //
  virtual bool doInitialization(Module *M) { return false; }

  // runOnMethod - Virtual method overriden by subclasses to do the per-method
  // processing of the pass.
  //
  virtual bool runOnMethod(Method *M) = 0;

  // doFinalization - Virtual method overriden by subclasses to do any post
  // processing needed after all passes have run.
  //
  virtual bool doFinalization(Module *M) { return false; }


  virtual bool run(Module *M) {
    bool Changed = doInitialization(M);

    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      Changed |= runOnMethod(*I);

    return Changed | doFinalization(M);
  }

 bool run(Method *M) {
   return doInitialization(M->getParent()) | runOnMethod(M)
        | doFinalization(M->getParent());
  }
};



//===----------------------------------------------------------------------===//
// CFGSafeMethodPass class - This class is used to implement global
// optimizations that do not modify the CFG of a method.  Optimizations should
// subclass this class if they meet the following constraints:
//   1. Optimizations are global, operating on a method at a time.
//   2. Optimizations do not modify the CFG of the contained method, by adding,
//      removing, or changing the order of basic blocks in a method.
//   3. Optimizations conform to all of the contstraints of MethodPass's.
//
struct CFGSafeMethodPass : public MethodPass {

  // TODO: Differentiation from MethodPass will come later

};


//===----------------------------------------------------------------------===//
// BasicBlockPass class - This class is used to implement most local
// optimizations.  Optimizations should subclass this class if they
// meet the following constraints:
//   1. Optimizations are local, operating on either a basic block or
//      instruction at a time.
//   2. Optimizations do not modify the CFG of the contained method, or any
//      other basic block in the method.
//   3. Optimizations conform to all of the contstraints of CFGSafeMethodPass's.
//
struct BasicBlockPass : public CFGSafeMethodPass {
  // runOnBasicBlock - Virtual method overriden by subclasses to do the
  // per-basicblock processing of the pass.
  //
  virtual bool runOnBasicBlock(BasicBlock *M) = 0;

  virtual bool runOnMethod(Method *M) {
    bool Changed = false;
    for (Method::iterator I = M->begin(), E = M->end(); I != E; ++I)
      Changed |= runOnBasicBlock(*I);
    return Changed;
  }

  bool run(BasicBlock *BB) {
    Module *M = BB->getParent()->getParent();
    return doInitialization(M) | runOnBasicBlock(BB) | doFinalization(M);
  }
};


//===----------------------------------------------------------------------===//
// PassManager - Container object for passes.  The PassManager destructor
// deletes all passes contained inside of the PassManager, so you shouldn't 
// delete passes manually, and all passes should be dynamically allocated.
//
class PassManager {
  std::vector<Pass*> Passes;
  MethodPassBatcher *Batcher;
public:
  PassManager() : Batcher(0) {}
  ~PassManager();

  // run - Run all of the queued passes on the specified module in an optimal
  // way.
  bool run(Module *M);

  // add - Add a pass to the queue of passes to run.  This passes ownership of
  // the Pass to the PassManager.  When the PassManager is destroyed, the pass
  // will be destroyed as well, so there is no need to delete the pass.  Also,
  // all passes MUST be new'd.
  //
  void add(Pass *P);
  void add(MethodPass *P);
  void add(BasicBlockPass *P);
};

#endif
