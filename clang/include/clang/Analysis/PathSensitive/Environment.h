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
#include "clang/Analysis/PathSensitive/Store.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Analysis/PathSensitive/SVals.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {

class EnvironmentManager;
class BasicValueFactory;
class LiveVariables;

class Environment : public llvm::FoldingSetNode {
private:
    
  friend class EnvironmentManager;
  
  // Type definitions.
  typedef llvm::ImmutableMap<Stmt*,SVal> BindingsTy;

  // Data.
  BindingsTy SubExprBindings;
  BindingsTy BlkExprBindings;
  
  Environment(BindingsTy seb, BindingsTy beb)
    : SubExprBindings(seb), BlkExprBindings(beb) {}
  
public:
    
  typedef BindingsTy::iterator seb_iterator;
  seb_iterator seb_begin() const { return SubExprBindings.begin(); }
  seb_iterator seb_end() const { return SubExprBindings.end(); }
  
  typedef BindingsTy::iterator beb_iterator;
  beb_iterator beb_begin() const { return BlkExprBindings.begin(); }
  beb_iterator beb_end() const { return BlkExprBindings.end(); }      
  
  SVal LookupSubExpr(Stmt* E) const {
    const SVal* X = SubExprBindings.lookup(cast<Expr>(E));
    return X ? *X : UnknownVal();
  }
  
  SVal LookupBlkExpr(Stmt* E) const {
    const SVal* X = BlkExprBindings.lookup(E);
    return X ? *X : UnknownVal();
  }
  
  SVal LookupExpr(Stmt* E) const {
    const SVal* X = SubExprBindings.lookup(E);
    if (X) return *X;
    X = BlkExprBindings.lookup(E);
    return X ? *X : UnknownVal();
  }
  
  SVal GetSVal(Stmt* Ex, BasicValueFactory& BasicVals) const;
  SVal GetBlkExprSVal(Stmt* Ex, BasicValueFactory& BasicVals) const; 
  
  /// Profile - Profile the contents of an Environment object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const Environment* E) {
    E->SubExprBindings.Profile(ID);
    E->BlkExprBindings.Profile(ID);
  }
  
  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, this);
  }
  
  bool operator==(const Environment& RHS) const {
    return SubExprBindings == RHS.SubExprBindings &&
           BlkExprBindings == RHS.BlkExprBindings;
  }
};
  
class EnvironmentManager {
private:
  typedef Environment::BindingsTy::Factory FactoryTy;
  FactoryTy F;
  
public:
  
  EnvironmentManager(llvm::BumpPtrAllocator& Allocator) : F(Allocator) {}
  ~EnvironmentManager() {}

  /// RemoveBlkExpr - Return a new environment object with the same bindings as
  ///  the provided environment except with any bindings for the provided Stmt*
  ///  removed.  This method only removes bindings for block-level expressions.
  ///  Using this method on a non-block level expression will return the
  ///  same environment object.
  Environment RemoveBlkExpr(const Environment& Env, Stmt* E) {
    return Environment(Env.SubExprBindings, F.Remove(Env.BlkExprBindings, E));
  }
  
  Environment RemoveSubExpr(const Environment& Env, Stmt* E) {
    return Environment(F.Remove(Env.SubExprBindings, E), Env.BlkExprBindings);
  }
  
  Environment AddBlkExpr(const Environment& Env, Stmt* E, SVal V) {
    return Environment(Env.SubExprBindings, F.Add(Env.BlkExprBindings, E, V));
  }
  
  Environment AddSubExpr(const Environment& Env, Stmt* E, SVal V) {
    return Environment(F.Add(Env.SubExprBindings, E, V), Env.BlkExprBindings);
  }
  
  /// RemoveSubExprBindings - Return a new environment object with
  ///  the same bindings as the provided environment except with all the
  ///  subexpression bindings removed.
  Environment RemoveSubExprBindings(const Environment& Env) {
    return Environment(F.GetEmptyMap(), Env.BlkExprBindings);
  }
  
  Environment getInitialEnvironment() {
    return Environment(F.GetEmptyMap(), F.GetEmptyMap());
  }
  
  Environment BindExpr(const Environment& Env, Stmt* E, SVal V,
                       bool isBlkExpr, bool Invalidate);

  Environment
  RemoveDeadBindings(Environment Env, Stmt* Loc, SymbolReaper& SymReaper,
                     GRStateManager& StateMgr, const GRState *state,
                     llvm::SmallVectorImpl<const MemRegion*>& DRoots);

};
  
} // end clang namespace

#endif
