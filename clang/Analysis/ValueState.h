//== ValueState.h - Path-Sens. "State" for tracking valuues -----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolID, ExprBindKey, and ValueState.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_VALUESTATE_H
#define LLVM_CLANG_ANALYSIS_VALUESTATE_H

// FIXME: Reduce the number of includes.

#include "RValues.h"

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

namespace clang {

//===----------------------------------------------------------------------===//
// ValueState - An ImmutableMap type Stmt*/Decl*/Symbols to RValues.
//===----------------------------------------------------------------------===//

namespace vstate {
  typedef llvm::ImmutableSet<llvm::APSInt*> IntSetTy;
  
  typedef llvm::ImmutableMap<Expr*,RValue>                 ExprBindingsTy;
  typedef llvm::ImmutableMap<VarDecl*,RValue>              VarBindingsTy;  
  typedef llvm::ImmutableMap<SymbolID,IntSetTy>            ConstantNotEqTy;
  typedef llvm::ImmutableMap<SymbolID,const llvm::APSInt*> ConstantEqTy;
}

/// ValueStateImpl - This class encapsulates the actual data values for
///  for a "state" in our symbolic value tracking.  It is intended to be
///  used as a functional object; that is once it is created and made
///  "persistent" in a FoldingSet its values will never change.
class ValueStateImpl : public llvm::FoldingSetNode {
private:
  void operator=(const ValueStateImpl& R) const;

public:
  vstate::ExprBindingsTy     SubExprBindings;
  vstate::ExprBindingsTy     BlockExprBindings;  
  vstate::VarBindingsTy      VarBindings;
  vstate::ConstantNotEqTy    ConstantNotEq;
  vstate::ConstantEqTy       ConstantEq;
  
  /// This ctor is used when creating the first ValueStateImpl object.
  ValueStateImpl(vstate::ExprBindingsTy EB,
                 vstate::VarBindingsTy VB,
                 vstate::ConstantNotEqTy CNE,
                 vstate::ConstantEqTy CE)
    : SubExprBindings(EB), 
      BlockExprBindings(EB),
      VarBindings(VB),
      ConstantNotEq(CNE),
      ConstantEq(CE) {}
  
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  ValueStateImpl(const ValueStateImpl& RHS)
    : llvm::FoldingSetNode(),
      SubExprBindings(RHS.SubExprBindings),
      BlockExprBindings(RHS.BlockExprBindings),
      VarBindings(RHS.VarBindings),
      ConstantNotEq(RHS.ConstantNotEq),
      ConstantEq(RHS.ConstantEq) {} 
  

  
  /// Profile - Profile the contents of a ValueStateImpl object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const ValueStateImpl& V) {
    V.SubExprBindings.Profile(ID);
    V.BlockExprBindings.Profile(ID);
    V.VarBindings.Profile(ID);
    V.ConstantNotEq.Profile(ID);
    V.ConstantEq.Profile(ID);
  }

  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, *this);
  }
  
};
  
/// ValueState - This class represents a "state" in our symbolic value
///  tracking. It is really just a "smart pointer", wrapping a pointer
///  to ValueStateImpl object.  Making this class a smart pointer means that its
///  size is always the size of a pointer, which allows easy conversion to
///  void* when being handled by GREngine.  It also forces us to unique states;
///  consequently, a ValueStateImpl* with a specific address will always refer
///  to the unique state with those values.
class ValueState {
  ValueStateImpl* Data;
public:
  ValueState(ValueStateImpl* D) : Data(D) {}
  ValueState() : Data(0) {}
  
  // Accessors.  
  ValueStateImpl* getImpl() const { return Data; }
  ValueStateImpl& operator*() { return *Data; }
  ValueStateImpl* operator->() { return Data; }

  // Typedefs.
  typedef vstate::IntSetTy                 IntSetTy;
  typedef vstate::ExprBindingsTy           ExprBindingsTy;
  typedef vstate::VarBindingsTy            VarBindingsTy;
  typedef vstate::ConstantNotEqTy          ConstantNotEqTy;
  typedef vstate::ConstantEqTy             ConstantEqTy;

  typedef llvm::SmallVector<ValueState,5>  BufferTy;

  // Queries.
  
  bool isNotEqual(SymbolID sym, const llvm::APSInt& V) const;
  const llvm::APSInt* getSymVal(SymbolID sym) const;
  
  // Iterators.

  typedef VarBindingsTy::iterator vb_iterator;
  vb_iterator vb_begin() const { return Data->VarBindings.begin(); }
  vb_iterator vb_end() const { return Data->VarBindings.end(); }
    
  typedef ExprBindingsTy::iterator seb_iterator;
  seb_iterator seb_begin() const { return Data->SubExprBindings.begin(); }
  seb_iterator seb_end() const { return Data->SubExprBindings.end(); }
  
  typedef ExprBindingsTy::iterator beb_iterator;
  beb_iterator beb_begin() const { return Data->BlockExprBindings.begin(); }
  beb_iterator beb_end() const { return Data->BlockExprBindings.end(); }
  
  // Profiling and equality testing.
  
  bool operator==(const ValueState& RHS) const {
    return Data == RHS.Data;
  }
  
  static void Profile(llvm::FoldingSetNodeID& ID, const ValueState& V) {
    ID.AddPointer(V.getImpl());
  }
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, *this);
  }
  
  void printDOT(std::ostream& Out) const;
  void print(std::ostream& Out) const;
  void print() const { print(*llvm::cerr); }
  
};  
  
template<> struct GRTrait<ValueState> {
  static inline void* toPtr(ValueState St) {
    return reinterpret_cast<void*>(St.getImpl());
  }  
  static inline ValueState toState(void* P) {    
    return ValueState(static_cast<ValueStateImpl*>(P));
  }
};
    
  
class ValueStateManager {
public:
  typedef ValueState StateTy;

private:
  ValueState::IntSetTy::Factory           ISetFactory;
  ValueState::ExprBindingsTy::Factory     EXFactory;
  ValueState::VarBindingsTy::Factory      VBFactory;
  ValueState::ConstantNotEqTy::Factory    CNEFactory;
  ValueState::ConstantEqTy::Factory       CEFactory;
  
  /// StateSet - FoldingSet containing all the states created for analyzing
  ///  a particular function.  This is used to unique states.
  llvm::FoldingSet<ValueStateImpl> StateSet;

  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager ValMgr;

  /// SymMgr - Object that manages the symbol information.
  SymbolManager SymMgr;

  /// Alloc - A BumpPtrAllocator to allocate states.
  llvm::BumpPtrAllocator& Alloc;
  
private:
  
  ValueState::ExprBindingsTy Remove(ValueState::ExprBindingsTy B, Expr* E) {
    return EXFactory.Remove(B, E);
  }    
    
  ValueState::VarBindingsTy  Remove(ValueState::VarBindingsTy B, VarDecl* V) {
    return VBFactory.Remove(B, V);
  }

  inline ValueState::ExprBindingsTy Remove(const ValueStateImpl& V, Expr* E) {
    return Remove(V.BlockExprBindings, E);
  }
  
  inline ValueState::VarBindingsTy Remove(const ValueStateImpl& V, VarDecl* D) {
    return Remove(V.VarBindings, D);
  }
                  
  ValueState BindVar(ValueState St, VarDecl* D, const RValue& V);
  ValueState UnbindVar(ValueState St, VarDecl* D);  
  
public:  
  ValueStateManager(ASTContext& Ctx, llvm::BumpPtrAllocator& alloc) 
    : ValMgr(Ctx, alloc), Alloc(alloc) {}
  
  ValueState getInitialState();
        
  ValueManager& getValueManager() { return ValMgr; }
  SymbolManager& getSymbolManager() { return SymMgr; }
  
  ValueState RemoveDeadBindings(ValueState St, Stmt* Loc, 
                                const LiveVariables& Liveness);
  
  ValueState RemoveSubExprBindings(ValueState St) {
    ValueStateImpl NewSt = *St;
    NewSt.SubExprBindings = EXFactory.GetEmptyMap();
    return getPersistentState(NewSt);    
  }
    
  
  ValueState SetValue(ValueState St, Expr* S, bool isBlkExpr, const RValue& V);
  ValueState SetValue(ValueState St, const LValue& LV, const RValue& V);

  RValue GetValue(ValueState St, Expr* S, bool* hasVal = NULL);
  RValue GetValue(ValueState St, const LValue& LV, QualType* T = NULL);    
  LValue GetLValue(ValueState St, Expr* S);
  
  ValueState getPersistentState(const ValueStateImpl& Impl);
  
  ValueState AddEQ(ValueState St, SymbolID sym, const llvm::APSInt& V);
  ValueState AddNE(ValueState St, SymbolID sym, const llvm::APSInt& V);
};
  
} // end clang namespace

#endif
