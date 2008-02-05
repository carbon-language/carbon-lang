//== ValueState.h - Path-Sens. "State" for tracking valuues -----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolID, ValueKey, and ValueState.
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

/// ValueKey - A variant smart pointer that wraps either a ValueDecl* or a
///  Stmt*.  Use cast<> or dyn_cast<> to get actual pointer type
class ValueKey {
  uintptr_t Raw;  
  void operator=(const ValueKey& RHS); // Do not implement.
  
public:
  enum  Kind { IsSubExpr=0x0, IsBlkExpr=0x1, IsDecl=0x2, // L-Value Bindings.
               IsSymbol=0x3, // Symbol Bindings.
               Mask=0x3 };
  
  inline Kind getKind() const {
    return (Kind) (Raw & Mask);
  }
  
  inline void* getPtr() const { 
    assert (getKind() != IsSymbol);
    return reinterpret_cast<void*>(Raw & ~Mask);
  }
  
  inline SymbolID getSymbolID() const {
    assert (getKind() == IsSymbol);
    return Raw >> 2;
  }
  
  ValueKey(const ValueDecl* VD)
  : Raw(reinterpret_cast<uintptr_t>(VD) | IsDecl) {
    assert(VD && "ValueDecl cannot be NULL.");
  }
  
  ValueKey(Stmt* S, bool isBlkExpr = false) 
  : Raw(reinterpret_cast<uintptr_t>(S) | (isBlkExpr ? IsBlkExpr : IsSubExpr)){
    assert(S && "Tracked statement cannot be NULL.");
  }
  
  ValueKey(SymbolID V)
  : Raw((V << 2) | IsSymbol) {}  
  
  bool isSymbol()  const { return getKind() == IsSymbol; }
  bool isSubExpr() const { return getKind() == IsSubExpr; }
  bool isBlkExpr() const { return getKind() == IsBlkExpr; }
  bool isDecl()    const { return getKind() == IsDecl; }
  bool isStmt()    const { return getKind() <= IsBlkExpr; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger(isSymbol() ? 1 : 0);
    
    if (isSymbol())
      ID.AddInteger(getSymbolID());
    else    
      ID.AddPointer(getPtr());
  }
  
  inline bool operator==(const ValueKey& X) const {
    return isSymbol() ? getSymbolID() == X.getSymbolID()
    : getPtr() == X.getPtr();
  }
  
  inline bool operator!=(const ValueKey& X) const {
    return !operator==(X);
  }
  
  inline bool operator<(const ValueKey& X) const { 
    if (isSymbol())
      return X.isSymbol() ? getSymbolID() < X.getSymbolID() : false;
    
    return getPtr() < X.getPtr();
  }
};

//===----------------------------------------------------------------------===//
// ValueState - An ImmutableMap type Stmt*/Decl*/Symbols to RValues.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<ValueKey,RValue> ValueState;

template<>
struct GRTrait<ValueState> {
  static inline void* toPtr(ValueState M) {
    return reinterpret_cast<void*>(M.getRoot());
  }  
  static inline ValueState toState(void* P) {
    return ValueState(static_cast<ValueState::TreeTy*>(P));
  }
};
  
  
class ValueStateManager {
public:
  typedef ValueState StateTy;

private:
  typedef ValueState::Factory FactoryTy;
  FactoryTy Factory;

  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager ValMgr;
  
  /// SymMgr - Object that manages the symbol information.
  SymbolManager SymMgr;
  
public:  
  ValueStateManager(ASTContext& Ctx) : ValMgr(Ctx) {}
  
  StateTy getInitialState() {
    return Factory.GetEmptyMap();
  }
        
  ValueManager& getValueManager() { return ValMgr; }
  SymbolManager& getSymbolManager() { return SymMgr; }
  
  StateTy SetValue(StateTy St, Stmt* S, bool isBlkExpr, const RValue& V);
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);

  RValue GetValue(const StateTy& St, Stmt* S, bool* hasVal = NULL);
  RValue GetValue(const StateTy& St, const LValue& LV);
    
  LValue GetLValue(const StateTy& St, Stmt* S);
  
  StateTy Remove(StateTy St, ValueKey K);
  
};
  
} // end clang namespace

//==------------------------------------------------------------------------==//
// Casting machinery to get cast<> and dyn_cast<> working with ValueKey.
//==------------------------------------------------------------------------==//

namespace llvm {
  
  template<> inline bool
  isa<clang::ValueDecl,clang::ValueKey>(const clang::ValueKey& V) {
    return V.getKind() == clang::ValueKey::IsDecl;
  }
  
  template<> inline bool
  isa<clang::Stmt,clang::ValueKey>(const clang::ValueKey& V) {
    return ((unsigned) V.getKind()) < clang::ValueKey::IsDecl;
  }
  
  template<> struct cast_retty_impl<clang::ValueDecl,clang::ValueKey> {
    typedef const clang::ValueDecl* ret_type;
  };
  
  template<> struct cast_retty_impl<clang::Stmt,clang::ValueKey> {
    typedef const clang::Stmt* ret_type;
  };
  
  template<> struct simplify_type<clang::ValueKey> {
    typedef void* SimpleType;
    static inline SimpleType getSimplifiedValue(const clang::ValueKey &V) {
      return V.getPtr();
    }
  };
} // end llvm namespace

#endif
