//===- AssumptionTracker.cpp - Track @llvm.assume -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that keeps track of @llvm.assume intrinsics in
// the functions of a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionTracker.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
using namespace llvm;
using namespace llvm::PatternMatch;

void AssumptionTracker::FunctionCallbackVH::deleted() {
  AT->forgetCachedAssumptions(cast<Function>(getValPtr()));
  // 'this' now dangles!
}

void AssumptionTracker::forgetCachedAssumptions(Function *F) {
  CachedAssumeCalls.erase(F);
}

void AssumptionTracker::CallCallbackVH::deleted() {
  assert(F && "delete callback called on dummy handle");
  FunctionCallsMap::iterator I = AT->CachedAssumeCalls.find(F);
  assert(I != AT->CachedAssumeCalls.end() &&
         "Function cleared from the map without removing the values?");

  I->second->erase(*this);
  // 'this' now dangles!
}

AssumptionTracker::FunctionCallsMap::iterator
AssumptionTracker::scanFunction(Function *F) {
  auto IP =
    CachedAssumeCalls.insert(std::make_pair(FunctionCallbackVH(F, this),
                                            std::unique_ptr<CallHandleSet>(
                                              new CallHandleSet())));
  assert(IP.second && "Scanning function already in the map?");

  FunctionCallsMap::iterator I = IP.first;

  // Go through all instructions in all blocks, add all calls to @llvm.assume
  // to our cache.
  for (BasicBlock &B : *F)
    for (Instruction &II : B)
      if (match(cast<Value>(&II), m_Intrinsic<Intrinsic::assume>(m_Value())))
        I->second->insert(CallCallbackVH(&II, this));

  return I;
}

void AssumptionTracker::verifyAnalysis() const {
#ifndef NDEBUG
  for (const auto &I : CachedAssumeCalls) {
    for (const BasicBlock &B : cast<Function>(*I.first))
      for (const Instruction &II : B) {
        Instruction *C = const_cast<Instruction*>(&II);
        if (match(C, m_Intrinsic<Intrinsic::assume>(m_Value()))) {
          assert(I.second->count(CallCallbackVH(C,
                   const_cast<AssumptionTracker*>(this))) &&
                 "Assumption in scanned function not in cache");
        }
    }
  }
#endif
}

void AssumptionTracker::registerAssumption(CallInst *CI) {
  assert(match(cast<Value>(CI),
               m_Intrinsic<Intrinsic::assume>(m_Value())) &&
         "Registered call does not call @llvm.assume");
  assert(CI->getParent() &&
         "Cannot register @llvm.assume call not in a basic block");

  Function *F = CI->getParent()->getParent();
  assert(F && "Cannot register @llvm.assume call not in a function");

  FunctionCallsMap::iterator I = CachedAssumeCalls.find(F);
  if (I == CachedAssumeCalls.end()) {
    // If this function has not already been scanned, then don't do anything
    // here. This intrinsic will be found, if it still exists, if the list of
    // assumptions in this function is requested at some later point. This
    // maintains the following invariant: if a function is present in the
    // cache, then its list of assumption intrinsic calls is complete.
    return;
  }

  I->second->insert(CallCallbackVH(CI, this));
}

AssumptionTracker::AssumptionTracker() : ImmutablePass(ID) {
  initializeAssumptionTrackerPass(*PassRegistry::getPassRegistry());
}

AssumptionTracker::~AssumptionTracker() {}

INITIALIZE_PASS(AssumptionTracker, "assumption-tracker", "Assumption Tracker",
                false, true)
char AssumptionTracker::ID = 0;

