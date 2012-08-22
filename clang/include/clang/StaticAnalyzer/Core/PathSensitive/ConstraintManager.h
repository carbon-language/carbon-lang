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

#ifndef LLVM_CLANG_GR_CONSTRAINT_MANAGER_H
#define LLVM_CLANG_GR_CONSTRAINT_MANAGER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"

namespace llvm {
class APSInt;
}

namespace clang {
namespace ento {

class SubEngine;

class ConstraintManager {
public:
  virtual ~ConstraintManager();
  virtual ProgramStateRef assume(ProgramStateRef state,
                                     DefinedSVal Cond,
                                     bool Assumption) = 0;

  std::pair<ProgramStateRef, ProgramStateRef >
    assumeDual(ProgramStateRef state, DefinedSVal Cond)
  {
    std::pair<ProgramStateRef, ProgramStateRef > res =
      std::make_pair(assume(state, Cond, true), assume(state, Cond, false));

    assert(!(!res.first && !res.second) && "System is over constrained.");
    return res;
  }

  virtual const llvm::APSInt* getSymVal(ProgramStateRef state,
                                        SymbolRef sym) const = 0;

  virtual bool isEqual(ProgramStateRef state,
                       SymbolRef sym,
                       const llvm::APSInt& V) const = 0;

  virtual ProgramStateRef removeDeadBindings(ProgramStateRef state,
                                                 SymbolReaper& SymReaper) = 0;

  virtual void print(ProgramStateRef state,
                     raw_ostream &Out,
                     const char* nl,
                     const char *sep) = 0;

  virtual void EndPath(ProgramStateRef state) {}

protected:
  /// canReasonAbout - Not all ConstraintManagers can accurately reason about
  ///  all SVal values.  This method returns true if the ConstraintManager can
  ///  reasonably handle a given SVal value.  This is typically queried by
  ///  ExprEngine to determine if the value should be replaced with a
  ///  conjured symbolic value in order to recover some precision.
  virtual bool canReasonAbout(SVal X) const = 0;
};

ConstraintManager* CreateRangeConstraintManager(ProgramStateManager& statemgr,
                                                SubEngine &subengine);

} // end GR namespace

} // end clang namespace

#endif
