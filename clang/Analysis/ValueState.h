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

class ExprBindKey {
  uintptr_t Raw;  
  void operator=(const ExprBindKey& RHS); // Do not implement.
  
  inline void* getPtr() const { 
    return reinterpret_cast<void*>(Raw & ~Mask);
  }
  
public:
  enum  Kind { IsSubExpr=0x0, IsBlkExpr=0x1, Mask=0x1 };
  
  inline Kind getKind() const {
    return (Kind) (Raw & Mask);
  }
    
  inline Expr* getExpr() const {
    return (Expr*) getPtr();
  }
    
  ExprBindKey(Expr* E, bool isBlkExpr = false) 
  : Raw(reinterpret_cast<uintptr_t>(E) | (isBlkExpr ? IsBlkExpr : IsSubExpr)){
    assert(E && "Tracked statement cannot be NULL.");
  }
  
  bool isSubExpr() const { return getKind() == IsSubExpr; }
  bool isBlkExpr() const { return getKind() == IsBlkExpr; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getPtr());
  }
  
  inline bool operator==(const ExprBindKey& X) const {
    return getPtr() == X.getPtr();
  }
  
  inline bool operator!=(const ExprBindKey& X) const {
    return !operator==(X);
  }
  
  inline bool operator<(const ExprBindKey& X) const { 
    return getPtr() < X.getPtr();
  }
};

//===----------------------------------------------------------------------===//
// ValueState - An ImmutableMap type Stmt*/Decl*/Symbols to RValues.
//===----------------------------------------------------------------------===//

namespace vstate {
  typedef llvm::ImmutableSet<llvm::APSInt*> IntSetTy;
  
  typedef llvm::ImmutableMap<ExprBindKey,RValue>           ExprBindingsTy;  
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
  vstate::ExprBindingsTy     ExprBindings;
  vstate::VarBindingsTy      VarBindings;
  vstate::ConstantNotEqTy    ConstantNotEq;
  vstate::ConstantEqTy       ConstantEq;
  
  /// This ctor is used when creating the first ValueStateImpl object.
  ValueStateImpl(vstate::ExprBindingsTy EB,
                 vstate::VarBindingsTy VB,
                 vstate::ConstantNotEqTy CNE,
                 vstate::ConstantEqTy CE)
    : ExprBindings(EB), VarBindings(VB), ConstantNotEq(CNE), ConstantEq(CE) {}
  
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  ValueStateImpl(const ValueStateImpl& RHS)
    : llvm::FoldingSetNode(),
      ExprBindings(RHS.ExprBindings),
      VarBindings(RHS.VarBindings),
      ConstantNotEq(RHS.ConstantNotEq),
      ConstantEq(RHS.ConstantEq) {} 
  

  
  /// Profile - Profile the contents of a ValueStateImpl object for use
  ///  in a FoldingSet.
  static void Profile(llvm::FoldingSetNodeID& ID, const ValueStateImpl& V) {
    V.ExprBindings.Profile(ID);
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
  vb_iterator vb_begin() { return Data->VarBindings.begin(); }
  vb_iterator vb_end() { return Data->VarBindings.end(); }
  
  typedef ExprBindingsTy::iterator eb_iterator;
  eb_iterator eb_begin() { return Data->ExprBindings.begin(); }
  eb_iterator eb_end() { return Data->ExprBindings.end(); }
  
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

  StateTy getPersistentState(const ValueState& St);
  
public:  
  ValueStateManager(ASTContext& Ctx, llvm::BumpPtrAllocator& alloc) 
    : ValMgr(Ctx, alloc), Alloc(alloc) {}
  
  StateTy getInitialState();
        
  ValueManager& getValueManager() { return ValMgr; }
  SymbolManager& getSymbolManager() { return SymMgr; }
  
  StateTy RemoveDeadBindings(StateTy St, Stmt* Loc, 
                             const LiveVariables& Liveness);
  
  StateTy SetValue(StateTy St, Expr* S, bool isBlkExpr, const RValue& V);
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);

  RValue GetValue(const StateTy& St, Expr* S, bool* hasVal = NULL);
  RValue GetValue(const StateTy& St, const LValue& LV, QualType* T = NULL);    
  LValue GetLValue(const StateTy& St, Expr* S);
  
  StateTy Add(StateTy St, ExprBindKey K, const RValue& V);
  StateTy Remove(StateTy St, ExprBindKey K);
  
  StateTy Add(StateTy St, VarDecl* D, const RValue& V);
  StateTy Remove(StateTy St, VarDecl* D);
  
  StateTy getPersistentState(const ValueStateImpl& Impl);
  
  StateTy AddEQ(StateTy St, SymbolID sym, const llvm::APSInt& V);
  StateTy AddNE(StateTy St, SymbolID sym, const llvm::APSInt& V);
};
  
} // end clang namespace

#endif
