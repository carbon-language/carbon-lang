//== Environment.h - Map from Stmt* to Locations/Values ---------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the Environment and EnvironmentManager classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ENVIRONMENT_H
#define LLVM_CLANG_ANALYSIS_ENVIRONMENT_H

// For using typedefs in StoreManager. Should find a better place for these
// typedefs.
#include "clang/Checker/PathSensitive/Store.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {

class AnalysisContext;
class EnvironmentManager;
class ValueManager;
class LiveVariables;


class Environment {
private:
  friend class EnvironmentManager;

  // Type definitions.
  typedef llvm::ImmutableMap<const Stmt*,SVal> BindingsTy;

  // Data.
  BindingsTy ExprBindings;
  AnalysisContext *ACtx;

  Environment(BindingsTy eb, AnalysisContext *aCtx)
    : ExprBindings(eb), ACtx(aCtx) {}

public:
  typedef BindingsTy::iterator iterator;
  iterator begin() const { return ExprBindings.begin(); }
  iterator end() const { return ExprBindings.end(); }

  SVal LookupExpr(const Stmt* E) const {
    const SVal* X = ExprBindings.lookup(E);
    return X ? *X : UnknownVal();
  }

  SVal GetSVal(const Stmt* Ex, ValueManager& ValMgr) const;

  AnalysisContext &getAnalysisContext() const { return *ACtx; }

  /// Profile - Profile the contents of an Environment object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const Environment* E) {
    E->ExprBindings.Profile(ID);
  }

  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, this);
  }

  bool operator==(const Environment& RHS) const {
    return ExprBindings == RHS.ExprBindings;
  }
};

class EnvironmentManager {
private:
  typedef Environment::BindingsTy::Factory FactoryTy;
  FactoryTy F;

public:
  EnvironmentManager(llvm::BumpPtrAllocator& Allocator) : F(Allocator) {}
  ~EnvironmentManager() {}

  Environment getInitialEnvironment(AnalysisContext *ACtx) {
    return Environment(F.GetEmptyMap(), ACtx);
  }

  Environment BindExpr(Environment Env, const Stmt *S, SVal V,
                       bool Invalidate);

  Environment RemoveDeadBindings(Environment Env, const Stmt *S,
                                 SymbolReaper &SymReaper, const GRState *ST,
                          llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);
};

} // end clang namespace

#endif
