//===- llvm/Transforms/Pass.h - Base class for XForm Passes ------*- C++ -*--=//
//
// This file defines a marker class that indicates that a specified class is a
// transformation pass implementation.
//
// Pass's are designed this way so that it is possible to apply N passes to a
// module, by first doing N Pass specific initializations for the module, then
// looping over all of the methods in the module, doing method specific work
// N times for each method.  Like this:
//
// for_each(Passes.begin(), Passes.end(), doPassInitialization(Module));
// for_each(Method *M <- Module->begin(), Module->end())
//   for_each(Passes.begin(), Passes.end(), doPerMethodWork(M));
//
// The other way to do things is like this:
// for_each(Pass *P <- Passes.begin(), Passes.end()) {
//   Passes->doPassInitialization(Module)
//   for_each(Module->begin(), Module->end(), P->doPerMethodWork);
// }
//
// But this can cause thrashing and poor cache performance, so we don't do it
// that way.
//
// Because a transformation does not see all methods consecutively, it should
// be careful about the state that it maintains... another pass may modify a
// method between two invocatations of doPerMethodWork.
//
// Also, implementations of doMethodWork should not remove any methods from the
// module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PASS_H
#define LLVM_TRANSFORMS_PASS_H

#include "llvm/Module.h"
#include "llvm/Method.h"

//===----------------------------------------------------------------------===//
// Pass interface - Implemented by all 'passes'.
//
struct Pass {
  //===--------------------------------------------------------------------===//
  // The externally useful entry points
  //

  // runAllPasses - Run a bunch of passes on the specified module, efficiently.
  static bool runAllPasses(Module *M, vector<Pass*> &Passes) {
    bool MadeChanges = false;
    // Run all of the pass initializers
    for (unsigned i = 0; i < Passes.size(); ++i)
      MadeChanges |= Passes[i]->doPassInitialization(M);
    
    // Loop over all of the methods, applying all of the passes to them
    for (unsigned m = 0; m < M->size(); ++m)
      for (unsigned i = 0; i < Passes.size(); ++i)
        MadeChanges |= Passes[i]->doPerMethodWork(*(M->begin()+m));

    // Run all of the pass finalizers...
    for (unsigned i = 0; i < Passes.size(); ++i)
      MadeChanges |= Passes[i]->doPassFinalization(M);
    return MadeChanges;
  }

  // runAllPassesAndFree - Run a bunch of passes on the specified module,
  // efficiently.  When done, delete all of the passes.
  //
  static bool runAllPassesAndFree(Module *M, vector<Pass*> &Passes) {
    // First run all of the passes
    bool MadeChanges = runAllPasses(M, Passes);

    // Free all of the passes.
    for (unsigned i = 0; i < Passes.size(); ++i)
      delete Passes[i];
    return MadeChanges;
  }


  // run(Module*) - Run this pass on a module and all of the methods contained
  // within it.  Returns true if any of the contained passes returned true.
  //
  bool run(Module *M) {
    bool MadeChanges = doPassInitialization(M);

    // Loop over methods in the module.  doPerMethodWork could add a method to
    // the Module, so we have to keep checking for end of method list condition.
    //
    for (unsigned m = 0; m < M->size(); ++m)
      MadeChanges |= doPerMethodWork(*(M->begin()+m));
    return MadeChanges | doPassFinalization(M);
  }

  // run(Method*) - Run this pass on a module and one specific method.  Returns
  // false on success.
  //
  bool run(Method *M) {
    return doPassInitialization(M->getParent()) | doPerMethodWork(M) |
           doPassFinalization(M->getParent());
  }


  //===--------------------------------------------------------------------===//
  // Functions to be implemented by subclasses
  //

  // Destructor - Virtual so we can be subclassed
  inline virtual ~Pass() {}

  // doPassInitialization - Virtual method overridden by subclasses to do
  // any neccesary per-module initialization.
  //
  virtual bool doPassInitialization(Module *M) { return false; }

  // doPerMethodWork - Virtual method overriden by subclasses to do the
  // per-method processing of the pass.
  //
  virtual bool doPerMethodWork(Method *M) { return false; }

  // doPassFinalization - Virtual method overriden by subclasses to do any post
  // processing needed after all passes have run.
  //
  virtual bool doPassFinalization(Module *M) { return false; }
};

#endif

