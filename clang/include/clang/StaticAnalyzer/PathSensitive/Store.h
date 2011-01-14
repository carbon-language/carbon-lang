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

#ifndef LLVM_CLANG_GR_STORE_H
#define LLVM_CLANG_GR_STORE_H

#include "clang/StaticAnalyzer/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/PathSensitive/SValBuilder.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"

namespace clang {

class Stmt;
class Expr;
class ObjCIvarDecl;
class StackFrameContext;

namespace ento {

/// Store - This opaque type encapsulates an immutable mapping from
///  locations to values.  At a high-level, it represents the symbolic
///  memory model.  Different subclasses of StoreManager may choose
///  different types to represent the locations and values.
typedef const void* Store;

class GRState;
class GRStateManager;
class SubRegionMap;

class StoreManager {
protected:
  SValBuilder &svalBuilder;
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

  virtual Store BindDefault(Store store, const MemRegion *R, SVal V) {
    return store;
  }

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

  virtual Loc getLValueVar(const VarDecl *VD, const LocationContext *LC) {
    return svalBuilder.makeLoc(MRMgr.getVarRegion(VD, LC));
  }

  virtual Loc getLValueString(const StringLiteral* S) {
    return svalBuilder.makeLoc(MRMgr.getStringRegion(S));
  }

  Loc getLValueCompoundLiteral(const CompoundLiteralExpr* CL,
                               const LocationContext *LC) {
    return loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL, LC));
  }

  virtual SVal getLValueIvar(const ObjCIvarDecl* decl, SVal base) {
    return getLValueFieldOrIvar(decl, base);
  }

  virtual SVal getLValueField(const FieldDecl* D, SVal Base) {
    return getLValueFieldOrIvar(D, Base);
  }

  virtual SVal getLValueElement(QualType elementType, NonLoc offset, SVal Base);

  // FIXME: This should soon be eliminated altogether; clients should deal with
  // region extents directly.
  virtual DefinedOrUnknownSVal getSizeInElements(const GRState *state, 
                                                 const MemRegion *region,
                                                 QualType EleTy) {
    return UnknownVal();
  }

  /// ArrayToPointer - Used by ExprEngine::VistCast to handle implicit
  ///  conversions between arrays and pointers.
  virtual SVal ArrayToPointer(Loc Array) = 0;

  /// Evaluates DerivedToBase casts.
  virtual SVal evalDerivedToBase(SVal derived, QualType basePtrType) {
    return UnknownVal();
  }

  class CastResult {
    const GRState *state;
    const MemRegion *region;
  public:
    const GRState *getState() const { return state; }
    const MemRegion* getRegion() const { return region; }
    CastResult(const GRState *s, const MemRegion* r = 0) : state(s), region(r){}
  };

  const ElementRegion *GetElementZeroRegion(const MemRegion *R, QualType T);

  /// CastRegion - Used by ExprEngine::VisitCast to handle casts from
  ///  a MemRegion* to a specific location type.  'R' is the region being
  ///  casted and 'CastToTy' the result type of the cast.
  const MemRegion *CastRegion(const MemRegion *region, QualType CastToTy);

  virtual Store RemoveDeadBindings(Store store, const StackFrameContext *LCtx,
                                   SymbolReaper& SymReaper,
                      llvm::SmallVectorImpl<const MemRegion*>& RegionRoots) = 0;

  virtual Store BindDecl(Store store, const VarRegion *VR, SVal initVal) = 0;

  virtual Store BindDeclWithNoInit(Store store, const VarRegion *VR) = 0;

  typedef llvm::DenseSet<SymbolRef> InvalidatedSymbols;
  typedef llvm::SmallVector<const MemRegion *, 8> InvalidatedRegions;

  /// InvalidateRegions - Clears out the specified regions from the store,
  ///  marking their values as unknown. Depending on the store, this may also
  ///  invalidate additional regions that may have changed based on accessing
  ///  the given regions. Optionally, invalidates non-static globals as well.
  /// \param[in] store The initial store
  /// \param[in] Begin A pointer to the first region to invalidate.
  /// \param[in] End A pointer just past the last region to invalidate.
  /// \param[in] E The current statement being evaluated. Used to conjure
  ///   symbols to mark the values of invalidated regions.
  /// \param[in] Count The current block count. Used to conjure
  ///   symbols to mark the values of invalidated regions.
  /// \param[in,out] IS A set to fill with any symbols that are no longer
  ///   accessible. Pass \c NULL if this information will not be used.
  /// \param[in] invalidateGlobals If \c true, any non-static global regions
  ///   are invalidated as well.
  /// \param[in,out] Regions A vector to fill with any regions being
  ///   invalidated. This should include any regions explicitly invalidated
  ///   even if they do not currently have bindings. Pass \c NULL if this
  ///   information will not be used.
  virtual Store InvalidateRegions(Store store,
                                  const MemRegion * const *Begin,
                                  const MemRegion * const *End,
                                  const Expr *E, unsigned Count,
                                  InvalidatedSymbols *IS,
                                  bool invalidateGlobals,
                                  InvalidatedRegions *Regions) = 0;

  /// enterStackFrame - Let the StoreManager to do something when execution
  /// engine is about to execute into a callee.
  virtual Store enterStackFrame(const GRState *state,
                                const StackFrameContext *frame);

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
  const MemRegion *MakeElementRegion(const MemRegion *baseRegion,
                                     QualType pointeeTy, uint64_t index = 0);

  /// CastRetrievedVal - Used by subclasses of StoreManager to implement
  ///  implicit casts that arise from loads from regions that are reinterpreted
  ///  as another region.
  SVal CastRetrievedVal(SVal val, const TypedRegion *region, QualType castTy,
                        bool performTestOnly = true);

private:
  SVal getLValueFieldOrIvar(const Decl* decl, SVal base);
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

} // end GR namespace

} // end clang namespace

#endif
