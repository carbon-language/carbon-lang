//== ConstraintManager.h - Constraints on symbolic values.-------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the interface to manage constraints on symbolic values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CONSTRAINT_MANAGER_H
#define LLVM_CLANG_ANALYSIS_CONSTRAINT_MANAGER_H

// FIXME: Typedef LiveSymbolsTy/DeadSymbolsTy at a more appropriate place.
#include "clang/Checker/PathSensitive/Store.h"

namespace llvm {
class APSInt;
}

namespace clang {

class GRState;
class GRStateManager;
class GRSubEngine;
class SVal;

class ConstraintManager {
public:
  virtual ~ConstraintManager();
  virtual const GRState *Assume(const GRState *state, DefinedSVal Cond,
                                bool Assumption) = 0;

  virtual const GRState *AssumeInBound(const GRState *state, DefinedSVal Idx,
                                       DefinedSVal UpperBound, bool Assumption) = 0;

  std::pair<const GRState*, const GRState*> AssumeDual(const GRState *state,
                                                       DefinedSVal Cond) {
    return std::make_pair(Assume(state, Cond, true),
                          Assume(state, Cond, false));
  }

  virtual const llvm::APSInt* getSymVal(const GRState *state,
                                        SymbolRef sym) const = 0;

  virtual bool isEqual(const GRState *state, SymbolRef sym,
                       const llvm::APSInt& V) const = 0;

  virtual const GRState *RemoveDeadBindings(const GRState *state,
                                            SymbolReaper& SymReaper) = 0;

  virtual void print(const GRState *state, llvm::raw_ostream& Out,
                     const char* nl, const char *sep) = 0;

  virtual void EndPath(const GRState *state) {}

  /// canReasonAbout - Not all ConstraintManagers can accurately reason about
  ///  all SVal values.  This method returns true if the ConstraintManager can
  ///  reasonably handle a given SVal value.  This is typically queried by
  ///  GRExprEngine to determine if the value should be replaced with a
  ///  conjured symbolic value in order to recover some precision.
  virtual bool canReasonAbout(SVal X) const = 0;
};

ConstraintManager* CreateBasicConstraintManager(GRStateManager& statemgr,
                                                GRSubEngine &subengine);
ConstraintManager* CreateRangeConstraintManager(GRStateManager& statemgr,
                                                GRSubEngine &subengine);

} // end clang namespace

#endif
