//== ValueManager.h - Aggregate manager of symbols and SVals ----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ValueManager, a class that manages symbolic values
//  and SVals created for use by GRExprEngine and related classes.  It
//  wraps SymbolManager, MemRegionManager, and BasicValueFactory.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_AGGREGATE_VALUE_MANAGER_H
#define LLVM_CLANG_ANALYSIS_AGGREGATE_VALUE_MANAGER_H

#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "clang/Analysis/PathSensitive/SVals.h"
#include "clang/Analysis/PathSensitive/BasicValueFactory.h"
#include "clang/Analysis/PathSensitive/SymbolManager.h"

namespace llvm { class BumpPtrAllocator; }
namespace clang { class GRStateManager; }

namespace clang {  
class ValueManager {
  friend class GRStateManager;

  ASTContext &Context;  
  BasicValueFactory BasicVals;
  
  /// SymMgr - Object that manages the symbol information.
  SymbolManager SymMgr;

  // FIXME: Eventually ValueManager will own this object.
  MemRegionManager *MemMgr;

  void setRegionManager(MemRegionManager& mm) { MemMgr = &mm; }
  
public:
  ValueManager(llvm::BumpPtrAllocator &alloc, ASTContext &context)
               : Context(context), BasicVals(Context, alloc),
                 SymMgr(Context, BasicVals, alloc),
                 MemMgr(0) {}

  // Accessors to submanagers.
  
  ASTContext &getContext() { return Context; }
  const ASTContext &getContext() const { return Context; }
  
  BasicValueFactory &getBasicValueFactory() { return BasicVals; }
  const BasicValueFactory &getBasicValueFactory() const { return BasicVals; }
  
  SymbolManager &getSymbolManager() { return SymMgr; }
  const SymbolManager &getSymbolManager() const { return SymMgr; }

  MemRegionManager &getRegionManager() { return *MemMgr; }
  const MemRegionManager &getRegionManager() const { return *MemMgr; }
  
  // Forwarding methods to SymbolManager.
  
  const SymbolConjured* getConjuredSymbol(const Stmt* E, QualType T,
                                          unsigned VisitCount,
                                          const void* SymbolTag = 0) {
    return SymMgr.getConjuredSymbol(E, T, VisitCount, SymbolTag);
  }
  
  const SymbolConjured* getConjuredSymbol(const Expr* E, unsigned VisitCount,
                                          const void* SymbolTag = 0) {    
    return SymMgr.getConjuredSymbol(E, VisitCount, SymbolTag);
  }
  
  // Aggregation methods that use multiple submanagers.
  
  Loc makeRegionVal(SymbolRef Sym) {
    return Loc::MakeVal(MemMgr->getSymbolicRegion(Sym));
  }
  
  /// makeZeroVal - Construct an SVal representing '0' for the specified type.
  SVal makeZeroVal(QualType T);
};
} // end clang namespace
#endif

