//===- llvm/Transforms/ConstantMerge.h - Merge duplicate consts --*- C++ -*--=//
//
// This file defines the interface to a pass that merges duplicate global
// constants together into a single constant that is shared.  This is useful
// because some passes (ie TraceValues) insert a lot of string constants into
// the program, regardless of whether or not they duplicate an existing string.
//
// Algorithm: ConstantMerge is designed to build up a map of available constants
// and elminate duplicates when it is initialized.
//
// The DynamicConstantMerge method is a superset of the ConstantMerge algorithm
// that checks for each method to see if constants have been added to the
// constant pool since it was last run... if so, it processes them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CONSTANTMERGE_H
#define LLVM_TRANSFORMS_CONSTANTMERGE_H

#include "llvm/Pass.h"
#include <map>
class Constant;
class GlobalVariable;

class ConstantMerge : public Pass {
protected:
  std::map<Constant*, GlobalVariable*> Constants;
  unsigned LastConstantSeen;
public:
  inline ConstantMerge() : LastConstantSeen(0) {}

  // mergeDuplicateConstants - Static accessor for clients that don't want to
  // deal with passes.
  //
  static bool mergeDuplicateConstants(Module *M);

  // doPassInitialization - For this pass, process all of the globals in the
  // module, eliminating duplicate constants.
  //
  bool doPassInitialization(Module *M);

  // doPassFinalization - Clean up internal state for this module
  //
  bool doPassFinalization(Module *M) {
    LastConstantSeen = 0;
    Constants.clear();
    return false;
  }
};

struct DynamicConstantMerge : public ConstantMerge {
  // doPerMethodWork - Check to see if any globals have been added to the 
  // global list for the module.  If so, eliminate them.
  //
  bool doPerMethodWork(Method *M);
};

#endif
