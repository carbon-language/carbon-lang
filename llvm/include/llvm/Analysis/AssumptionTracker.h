//===- llvm/Analysis/AssumptionTracker.h - Track @llvm.assume ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that keeps track of @llvm.assume intrinsics in
// the functions of a module (allowing assumptions within any function to be
// found cheaply by other parts of the optimizer).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ASSUMPTIONTRACKER_H
#define LLVM_ANALYSIS_ASSUMPTIONTRACKER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {

/// An immutable pass that tracks @llvm.assume intrinsics in a module.
class AssumptionTracker : public ImmutablePass {
  /// A callback value handle applied to function objects, which we use to
  /// delete our cache of intrinsics for a function when it is deleted.
  class FunctionCallbackVH : public CallbackVH {
    AssumptionTracker *AT;
    void deleted() override;

    public:
      typedef DenseMapInfo<Value *> DMI;

      FunctionCallbackVH(Value *V, AssumptionTracker *AT = nullptr)
        : CallbackVH(V), AT(AT) {}
  };

  /// A callback value handle applied to call instructions, which keeps
  /// track of the call's parent function so that we can remove a
  /// assumption intrinsic call from our cache when the instruction is
  /// deleted.
  class CallCallbackVH : public CallbackVH {
    AssumptionTracker *AT;
    void deleted() override;

    // We store the function here because we need it to lookup the set
    // containing this handle when the underlying CallInst is being deleted.
    Function *F;

    public:
      typedef DenseMapInfo<Instruction *> DMI;

      CallCallbackVH(Instruction *I, AssumptionTracker *AT = nullptr)
        : CallbackVH(I), AT(AT), F(nullptr) {
        if (I != DMI::getEmptyKey() && I != DMI::getTombstoneKey())
          F = I->getParent()->getParent();
      }

      operator CallInst*() const {
        Value *V = getValPtr();
        if (V == DMI::getEmptyKey() || V == DMI::getTombstoneKey())
          return reinterpret_cast<CallInst*>(V);

        return cast<CallInst>(V);
      }

      CallInst *operator->() const { return cast<CallInst>(getValPtr()); }
      CallInst &operator*() const { return *cast<CallInst>(getValPtr()); }
  };

  friend FunctionCallbackVH;
  friend CallCallbackVH;

  // FIXME: SmallSet might be better here, but it currently has no iterators.
  typedef DenseSet<CallCallbackVH, CallCallbackVH::DMI> CallHandleSet;
  typedef DenseMap<FunctionCallbackVH, std::unique_ptr<CallHandleSet>,
                   FunctionCallbackVH::DMI> FunctionCallsMap;
  FunctionCallsMap CachedAssumeCalls;

  /// Scan the provided function for @llvm.assume intrinsic calls. Returns an
  /// iterator to the set for this function in the CachedAssumeCalls map.
  FunctionCallsMap::iterator scanFunction(Function *F);

public:
  /// Remove the cache of @llvm.assume intrinsics for the given function.
  void forgetCachedAssumptions(Function *F);

  /// Add an @llvm.assume intrinsic to the cache for its parent function.
  void registerAssumption(CallInst *CI);

  typedef CallHandleSet::iterator assumption_iterator;
  typedef iterator_range<assumption_iterator> assumption_range;

  inline assumption_range assumptions(Function *F) {
    FunctionCallsMap::iterator I = CachedAssumeCalls.find_as(F);
    if (I == CachedAssumeCalls.end()) {
      I = scanFunction(F);
    }

    return assumption_range(I->second->begin(), I->second->end());
  }

  AssumptionTracker();
  ~AssumptionTracker();

  void releaseMemory() override {
    CachedAssumeCalls.shrink_and_clear();
  }

  void verifyAnalysis() const override;
  bool doFinalization(Module &) override {
    verifyAnalysis();
    return false;
  }

  static char ID; // Pass identification, replacement for typeid
};

} // end namespace llvm

#endif
