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
  typedef llvm::SmallSet<SymbolRef, 20>      LiveSymbolsTy;
  typedef llvm::DenseSet<SymbolRef>          DeadSymbolsTy;

protected:
  /// MRMgr - Manages region objects associated with this StoreManager.
  MemRegionManager MRMgr;

  StoreManager(llvm::BumpPtrAllocator& Alloc) : MRMgr(Alloc) {}

public:  
  virtual ~StoreManager() {}
  
  /// Return the value bound to specified location in a given state.
  /// \param[in] state The analysis state.
  /// \param[in] loc The symbolic memory location.
  /// \param[in] T An optional type that provides a hint indicating the 
  ///   expected type of the returned value.  This is used if the value is
  ///   lazily computed.
  /// \return The value bound to the location \c loc.
  virtual SVal Retrieve(const GRState* state, Loc loc,
                        QualType T = QualType()) = 0;  

//  /// Retrieves the value bound to the specified region.
//  SVal GetRegionSVal(const GRState* state, const MemRegion* R) {
//    return Retrieve(state, loc::MemRegionVal(R));
//  }

  /// Return a state with the specified value bound to the given location.
  /// \param[in] state The analysis state.
  /// \param[in] loc The symbolic memory location.
  /// \param[in] val The value to bind to location \c loc.
  /// \return A pointer to a GRState object that contains the same bindings as 
  ///   \c state with the addition of having the value specified by \c val bound
  ///   to the location given for \c loc.
  virtual const GRState* Bind(const GRState* state, Loc loc, SVal val) = 0;

  virtual Store Remove(Store St, Loc L) = 0;
  
  /// BindCompoundLiteral - Return the store that has the bindings currently
  ///  in 'store' plus the bindings for the CompoundLiteral.  'R' is the region
  ///  for the compound literal and 'BegInit' and 'EndInit' represent an
  ///  array of initializer values.
  virtual const GRState* BindCompoundLiteral(const GRState* St, 
                                             const CompoundLiteralExpr* CL,
                                             SVal V) = 0;
  
  virtual Store getInitialStore() = 0;
  MemRegionManager& getRegionManager() { return MRMgr; }

  virtual SVal getLValueVar(const GRState* St, const VarDecl* VD) = 0;

  virtual SVal getLValueString(const GRState* St, const StringLiteral* S) = 0;

  virtual SVal getLValueCompoundLiteral(const GRState* St, 
                                        const CompoundLiteralExpr* CL) = 0;
  
  virtual SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                             SVal Base) = 0;
  
  virtual SVal getLValueField(const GRState* St, SVal Base, 
                              const FieldDecl* D) = 0;
  
  virtual SVal getLValueElement(const GRState* St, SVal Base, SVal Offset) = 0;

  virtual SVal getSizeInElements(const GRState* St, const MemRegion* R) {
    return UnknownVal();
  }

  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  virtual SVal ArrayToPointer(SVal Array) = 0;

  
  class CastResult {
    const GRState* State;
    const MemRegion* R;
  public:
    const GRState* getState() const { return State; }
    const MemRegion* getRegion() const { return R; }
    CastResult(const GRState* s, const MemRegion* r = 0) : State(s), R(r) {}
  };
  
  /// CastRegion - Used by GRExprEngine::VisitCast to handle casts from
  ///  a MemRegion* to a specific location type.  'R' is the region being
  ///  casted and 'CastToTy' the result type of the cast.
  virtual CastResult CastRegion(const GRState* state, const MemRegion* R,
                                QualType CastToTy) = 0;
  
  /// getSelfRegion - Returns the region for the 'self' (Objective-C) or
  ///  'this' object (C++).  When used when analyzing a normal function this
  ///  method returns NULL.
  virtual const MemRegion* getSelfRegion(Store store) = 0;

  virtual Store
  RemoveDeadBindings(const GRState* state, Stmt* Loc, const LiveVariables& Live,
                     llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                     LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) = 0;

  virtual const GRState* BindDecl(const GRState* St, const VarDecl* VD, 
                                  SVal InitVal) = 0;

  virtual const GRState* BindDeclWithNoInit(const GRState* St, 
                                            const VarDecl* VD) = 0;

  virtual const GRState* setExtent(const GRState* St,
                                   const MemRegion* R, SVal Extent) {
    return St;
  }

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
