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

#ifndef LLVM_CLANG_GR_ENVIRONMENT_H
#define LLVM_CLANG_GR_ENVIRONMENT_H

#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/ImmutableMap.h"

namespace clang {

class LiveVariables;

namespace ento {

class EnvironmentManager;
class SValBuilder;

/// Environment - An immutable map from Stmts to their current
///  symbolic values (SVals).
///
class Environment {
private:
  friend class EnvironmentManager;

  // Type definitions.
  typedef llvm::ImmutableMap<const Stmt*,SVal> BindingsTy;

  // Data.
  BindingsTy ExprBindings;

  Environment(BindingsTy eb)
    : ExprBindings(eb) {}

  SVal lookupExpr(const Stmt* E) const;

public:
  typedef BindingsTy::iterator iterator;
  iterator begin() const { return ExprBindings.begin(); }
  iterator end() const { return ExprBindings.end(); }


  /// getSVal - Fetches the current binding of the expression in the
  ///  Environment.
  SVal getSVal(const Stmt* Ex, SValBuilder& svalBuilder,
	       bool useOnlyDirectBindings = false) const;

  /// Profile - Profile the contents of an Environment object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const Environment* env) {
    env->ExprBindings.Profile(ID);
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

  Environment getInitialEnvironment() {
    return Environment(F.getEmptyMap());
  }

  /// Bind the value 'V' to the statement 'S'.
  Environment bindExpr(Environment Env, const Stmt *S, SVal V,
                       bool Invalidate);
  
  /// Bind the location 'location' and value 'V' to the statement 'S'.  This
  /// is used when simulating loads/stores.
  Environment bindExprAndLocation(Environment Env, const Stmt *S, SVal location,
                                  SVal V);

  Environment removeDeadBindings(Environment Env,
                                 SymbolReaper &SymReaper, const GRState *ST);
};

} // end GR namespace

} // end clang namespace

#endif
