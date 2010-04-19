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

#include "clang/Checker/PathSensitive/MemRegion.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/Checker/PathSensitive/ValueManager.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {

typedef const void* Store;

class GRState;
class GRStateManager;
class Stmt;
class Expr;
class ObjCIvarDecl;
class SubRegionMap;
class StackFrameContext;

class StoreManager {
protected:
  ValueManager &ValMgr;
  GRStateManager &StateMgr;

  /// MRMgr - Manages region objects associated with this StoreManager.
  MemRegionManager &MRMgr;
  ASTContext &Ctx;

  StoreManager(GRStateManager &stateMgr);

public:
  virtual ~StoreManager() {}

  /// Return the value bound to specified location in a given state.
  /// \param[in] state The analysis state.
  /// \param[in] loc The symbolic memory location.
  /// \param[in] T An optional type that provides a hint indicating the
  ///   expected type of the returned value.  This is used if the value is
  ///   lazily computed.
  /// \return The value bound to the location \c loc.
  virtual SVal Retrieve(Store store, Loc loc, QualType T = QualType()) = 0;

  /// Return a state with the specified value bound to the given location.
  /// \param[in] state The analysis state.
  /// \param[in] loc The symbolic memory location.
  /// \param[in] val The value to bind to location \c loc.
  /// \return A pointer to a GRState object that contains the same bindings as
  ///   \c state with the addition of having the value specified by \c val bound
  ///   to the location given for \c loc.
  virtual Store Bind(Store store, Loc loc, SVal val) = 0;

  virtual Store Remove(Store St, Loc L) = 0;

  /// BindCompoundLiteral - Return the store that has the bindings currently
  ///  in 'store' plus the bindings for the CompoundLiteral.  'R' is the region
  ///  for the compound literal and 'BegInit' and 'EndInit' represent an
  ///  array of initializer values.
  virtual Store BindCompoundLiteral(Store store,
                                    const CompoundLiteralExpr* cl,
                                    const LocationContext *LC, SVal v) = 0;

  /// getInitialStore - Returns the initial "empty" store representing the
  ///  value bindings upon entry to an analyzed function.
  virtual Store getInitialStore(const LocationContext *InitLoc) = 0;

  /// getRegionManager - Returns the internal RegionManager object that is
  ///  used to query and manipulate MemRegion objects.
  MemRegionManager& getRegionManager() { return MRMgr; }

  /// getSubRegionMap - Returns an opaque map object that clients can query
  ///  to get the subregions of a given MemRegion object.  It is the
  //   caller's responsibility to 'delete' the returned map.
  virtual SubRegionMap *getSubRegionMap(Store store) = 0;

  virtual SVal getLValueVar(const VarDecl *VD, const LocationContext *LC) {
    return ValMgr.makeLoc(MRMgr.getVarRegion(VD, LC));
  }

  virtual SVal getLValueString(const StringLiteral* S) {
    return ValMgr.makeLoc(MRMgr.getStringRegion(S));
  }

  SVal getLValueCompoundLiteral(const CompoundLiteralExpr* CL,
                                const LocationContext *LC) {
    return loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL, LC));
  }

  virtual SVal getLValueIvar(const ObjCIvarDecl* decl, SVal base) {
    return getLValueFieldOrIvar(decl, base);
  }

  virtual SVal getLValueField(const FieldDecl* D, SVal Base) {
    return getLValueFieldOrIvar(D, Base);
  }

  virtual SVal getLValueElement(QualType elementType, SVal offset, SVal Base);

  // FIXME: Make out-of-line.
  virtual DefinedOrUnknownSVal getSizeInElements(const GRState *state, 
                                                 const MemRegion *region,
                                                 QualType EleTy) {
    return UnknownVal();
  }

  /// ArrayToPointer - Used by GRExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  virtual SVal ArrayToPointer(Loc Array) = 0;

  class CastResult {
    const GRState *state;
    const MemRegion *region;
  public:
    const GRState *getState() const { return state; }
    const MemRegion* getRegion() const { return region; }
    CastResult(const GRState *s, const MemRegion* r = 0) : state(s), region(r){}
  };

  const ElementRegion *GetElementZeroRegion(const MemRegion *R, QualType T);

  /// CastRegion - Used by GRExprEngine::VisitCast to handle casts from
  ///  a MemRegion* to a specific location type.  'R' is the region being
  ///  casted and 'CastToTy' the result type of the cast.
  const MemRegion *CastRegion(const MemRegion *region, QualType CastToTy);


  /// EvalBinOp - Perform pointer arithmetic.
  virtual SVal EvalBinOp(BinaryOperator::Opcode Op,
                         Loc lhs, NonLoc rhs, QualType resultTy) {
    return UnknownVal();
  }

  virtual Store RemoveDeadBindings(Store store, Stmt* Loc,
                                   const StackFrameContext *LCtx,
                                   SymbolReaper& SymReaper,
                      llvm::SmallVectorImpl<const MemRegion*>& RegionRoots) = 0;

  virtual Store BindDecl(Store store, const VarRegion *VR, SVal initVal) = 0;

  virtual Store BindDeclWithNoInit(Store store, const VarRegion *VR) = 0;

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  
  virtual Store InvalidateRegion(Store store,
                                 const MemRegion *R,
                                 const Expr *E, unsigned Count,
                                 InvalidatedSymbols *IS) = 0;
  
  virtual Store InvalidateRegions(Store store,
                                  const MemRegion * const *Begin,
                                  const MemRegion * const *End,
                                  const Expr *E, unsigned Count,
                                  InvalidatedSymbols *IS);  

  // FIXME: Make out-of-line.
  virtual const GRState *setExtent(const GRState *state,
                                    const MemRegion *region, SVal extent) {
    return state;
  }

  /// EnterStackFrame - Let the StoreManager to do something when execution
  /// engine is about to execute into a callee.
  virtual const GRState *EnterStackFrame(const GRState *state,
                                         const StackFrameContext *frame) {
    return state;
  }

  virtual void print(Store store, llvm::raw_ostream& Out,
                     const char* nl, const char *sep) = 0;

  class BindingsHandler {
  public:
    virtual ~BindingsHandler();
    virtual bool HandleBinding(StoreManager& SMgr, Store store,
                               const MemRegion *region, SVal val) = 0;
  };

  /// iterBindings - Iterate over the bindings in the Store.
  virtual void iterBindings(Store store, BindingsHandler& f) = 0;

protected:
  const MemRegion *MakeElementRegion(const MemRegion *Base,
                                     QualType pointeeTy, uint64_t index = 0);

  /// CastRetrievedVal - Used by subclasses of StoreManager to implement
  ///  implicit casts that arise from loads from regions that are reinterpreted
  ///  as another region.
  SVal CastRetrievedVal(SVal val, const TypedRegion *R, QualType castTy,
                        bool performTestOnly = true);

private:
  SVal getLValueFieldOrIvar(const Decl* D, SVal Base);
};

// FIXME: Do we still need this?
/// SubRegionMap - An abstract interface that represents a queryable map
///  between MemRegion objects and their subregions.
class SubRegionMap {
public:
  virtual ~SubRegionMap() {}

  class Visitor {
  public:
    virtual ~Visitor() {}
    virtual bool Visit(const MemRegion* Parent, const MemRegion* SubRegion) = 0;
  };

  virtual bool iterSubRegions(const MemRegion *region, Visitor& V) const = 0;
};

// FIXME: Do we need to pass GRStateManager anymore?
StoreManager *CreateBasicStoreManager(GRStateManager& StMgr);
StoreManager *CreateRegionStoreManager(GRStateManager& StMgr);
StoreManager *CreateFieldsOnlyRegionStoreManager(GRStateManager& StMgr);
StoreManager *CreateFlatStoreManager(GRStateManager &StMgr);
} // end clang namespace

#endif
