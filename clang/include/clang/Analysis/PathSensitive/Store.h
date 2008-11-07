//== Store.h - Interface for maps from Locations to Values ------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the types Store and StoreManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_STORE_H
#define LLVM_CLANG_ANALYSIS_STORE_H

#include "clang/Analysis/PathSensitive/SVals.h"
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <iosfwd>

namespace clang {
  
typedef const void* Store;

class GRState;  
class GRStateManager;
class LiveVariables;
class Stmt;
class Expr;
class ObjCIvarDecl;

class StoreManager {
public:
  typedef llvm::SmallSet<SymbolID, 20>      LiveSymbolsTy;
  typedef llvm::DenseSet<SymbolID>          DeadSymbolsTy;
  
  virtual ~StoreManager() {}
  virtual SVal Retrieve(Store St, Loc LV, QualType T = QualType()) = 0;  

  virtual SVal GetRegionSVal(Store St, const MemRegion* R) {
    return Retrieve(St, loc::MemRegionVal(R));
  }
  
  virtual Store Bind(Store St, Loc LV, SVal V) = 0;
  virtual Store Remove(Store St, Loc LV) = 0;
  
  /// BindCompoundLiteral - Return the store that has the bindings currently
  ///  in 'store' plus the bindings for the CompoundLiteral.  'R' is the region
  ///  for the compound literal and 'BegInit' and 'EndInit' represent an
  ///  array of initializer values.
  virtual Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* CL,
                                    SVal V) = 0;
  
  virtual Store getInitialStore() = 0;
  virtual MemRegionManager& getRegionManager() = 0;

  virtual SVal getLValueVar(const GRState* St, const VarDecl* VD) = 0;

  virtual SVal getLValueString(const GRState* St, const StringLiteral* S) = 0;

  virtual SVal getLValueCompoundLiteral(const GRState* St, 
                                        const CompoundLiteralExpr* CL) = 0;
  
  virtual SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                             SVal Base) = 0;
  
  virtual SVal getLValueField(const GRState* St, SVal Base, 
                              const FieldDecl* D) = 0;
  
  virtual SVal getLValueElement(const GRState* St, 
                                   SVal Base, SVal Offset) = 0;
  
  
  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  virtual SVal ArrayToPointer(SVal Array) = 0;
  
  /// getSelfRegion - Returns the region for the 'self' (Objective-C) or
  ///  'this' object (C++).  When used when analyzing a normal function this
  ///  method returns NULL.
  virtual const MemRegion* getSelfRegion(Store store) = 0;

  virtual Store
  RemoveDeadBindings(Store store, Stmt* Loc, const LiveVariables& Live,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                     LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) = 0;

  virtual Store BindDecl(Store store,
                         const VarDecl* VD, Expr* Ex, 
                         SVal InitVal = UndefinedVal(), 
                         unsigned Count = 0) = 0;

  virtual void print(Store store, std::ostream& Out,
                     const char* nl, const char *sep) = 0;
      
  class BindingsHandler {
  public:    
    virtual ~BindingsHandler();
    virtual bool HandleBinding(StoreManager& SMgr, Store store,
                               MemRegion* R, SVal val) = 0;
  };
  
  /// iterBindings - Iterate over the bindings in the Store.
  virtual void iterBindings(Store store, BindingsHandler& f) = 0;  
};
  
StoreManager* CreateBasicStoreManager(GRStateManager& StMgr);
StoreManager* CreateRegionStoreManager(GRStateManager& StMgr);
  
} // end clang namespace

#endif
