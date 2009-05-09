//== RegionStore.cpp - Field-sensitive store model --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a basic region store model. In this model, we do have field
// sensitivity. But we assume nothing about the heap shape. So recursive data
// structures are largely ignored. Basically we do 1-limiting analysis.
// Parameter pointers are assumed with no aliasing. Pointee objects of
// parameters are created lazily.
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/PathSensitive/MemRegion.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

// Actual Store type.
typedef llvm::ImmutableMap<const MemRegion*, SVal> RegionBindingsTy;

//===----------------------------------------------------------------------===//
// Region "Views"
//===----------------------------------------------------------------------===//
//
//  MemRegions can be layered on top of each other.  This GDM entry tracks
//  what are the MemRegions that layer a given MemRegion.
//
typedef llvm::ImmutableSet<const MemRegion*> RegionViews;
namespace { class VISIBILITY_HIDDEN RegionViewMap {}; }
static int RegionViewMapIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionViewMap> 
    : public GRStatePartialTrait<llvm::ImmutableMap<const MemRegion*,
                                                    RegionViews> > {
                                                      
    static void* GDMIndex() { return &RegionViewMapIndex; }
  };
}

// RegionCasts records the current cast type of a region.
namespace { class VISIBILITY_HIDDEN RegionCasts {}; }
static int RegionCastsIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionCasts>
    : public GRStatePartialTrait<llvm::ImmutableMap<const MemRegion*, 
                                                    QualType> > {
    static void* GDMIndex() { return &RegionCastsIndex; }
  };
}

//===----------------------------------------------------------------------===//
// Region "Extents"
//===----------------------------------------------------------------------===//
//
//  MemRegions represent chunks of memory with a size (their "extent").  This
//  GDM entry tracks the extents for regions.  Extents are in bytes.
//
namespace { class VISIBILITY_HIDDEN RegionExtents {}; }
static int RegionExtentsIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionExtents>
    : public GRStatePartialTrait<llvm::ImmutableMap<const MemRegion*, SVal> > {
    static void* GDMIndex() { return &RegionExtentsIndex; }
  };
}

//===----------------------------------------------------------------------===//
// Region "killsets".
//===----------------------------------------------------------------------===//
//
// RegionStore lazily adds value bindings to regions when the analyzer handles
//  assignment statements.  Killsets track which default values have been
//  killed, thus distinguishing between "unknown" values and default
//  values. Regions are added to killset only when they are assigned "unknown"
//  directly, otherwise we should have their value in the region bindings.
//
namespace { class VISIBILITY_HIDDEN RegionKills {}; }
static int RegionKillsIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionKills>
  : public GRStatePartialTrait< llvm::ImmutableSet<const MemRegion*> > {
    static void* GDMIndex() { return &RegionKillsIndex; }
  };
}

//===----------------------------------------------------------------------===//
// Regions with default values.
//===----------------------------------------------------------------------===//
//
// This GDM entry tracks what regions have a default value if they have no bound
// value and have not been killed.
//
namespace { class VISIBILITY_HIDDEN RegionDefaultValue {}; }
static int RegionDefaultValueIndex = 0;
namespace clang {
 template<> struct GRStateTrait<RegionDefaultValue>
   : public GRStatePartialTrait<llvm::ImmutableMap<const MemRegion*, SVal> > {
   static void* GDMIndex() { return &RegionDefaultValueIndex; }
 };
}

//===----------------------------------------------------------------------===//
// Main RegionStore logic.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN RegionStoreSubRegionMap : public SubRegionMap {
  typedef llvm::DenseMap<const MemRegion*,
                         llvm::ImmutableSet<const MemRegion*> > Map;
  
  llvm::ImmutableSet<const MemRegion*>::Factory F;
  Map M;

public:
  void add(const MemRegion* Parent, const MemRegion* SubRegion) {
    Map::iterator I = M.find(Parent);
    M.insert(std::make_pair(Parent, 
             F.Add(I == M.end() ? F.GetEmptySet() : I->second, SubRegion)));
  }
    
  ~RegionStoreSubRegionMap() {}
  
  bool iterSubRegions(const MemRegion* Parent, Visitor& V) const {
    Map::iterator I = M.find(Parent);

    if (I == M.end())
      return true;
    
    llvm::ImmutableSet<const MemRegion*> S = I->second;
    for (llvm::ImmutableSet<const MemRegion*>::iterator SI=S.begin(),SE=S.end();
         SI != SE; ++SI) {
      if (!V.Visit(Parent, *SI))
        return false;
    }
    
    return true;
  }
};  

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  RegionViews::Factory RVFactory;

  const MemRegion* SelfRegion;
  const ImplicitParamDecl *SelfDecl;

public:
  RegionStoreManager(GRStateManager& mgr) 
    : StoreManager(mgr),
      RBFactory(mgr.getAllocator()),
      RVFactory(mgr.getAllocator()),
      SelfRegion(0), SelfDecl(0) {
    if (const ObjCMethodDecl* MD =
          dyn_cast<ObjCMethodDecl>(&StateMgr.getCodeDecl()))
      SelfDecl = MD->getSelfDecl();
  }

  virtual ~RegionStoreManager() {}

  SubRegionMap* getSubRegionMap(const GRState *state);
  
  const GRState* BindCompoundLiteral(const GRState* St, 
                                     const CompoundLiteralExpr* CL, SVal V);

  /// getLValueString - Returns an SVal representing the lvalue of a
  ///  StringLiteral.  Within RegionStore a StringLiteral has an
  ///  associated StringRegion, and the lvalue of a StringLiteral is
  ///  the lvalue of that region.
  SVal getLValueString(const GRState* St, const StringLiteral* S);

  /// getLValueCompoundLiteral - Returns an SVal representing the
  ///   lvalue of a compound literal.  Within RegionStore a compound
  ///   literal has an associated region, and the lvalue of the
  ///   compound literal is the lvalue of that region.
  SVal getLValueCompoundLiteral(const GRState* St, const CompoundLiteralExpr*);

  /// getLValueVar - Returns an SVal that represents the lvalue of a
  ///  variable.  Within RegionStore a variable has an associated
  ///  VarRegion, and the lvalue of the variable is the lvalue of that region.
  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);

  SVal getLValueField(const GRState* St, SVal Base, const FieldDecl* D);
  
  SVal getLValueFieldOrIvar(const GRState* St, SVal Base, const Decl* D);

  SVal getLValueElement(const GRState* St, QualType elementType,
                        SVal Base, SVal Offset);

  SVal getSizeInElements(const GRState* St, const MemRegion* R);

  /// ArrayToPointer - Emulates the "decay" of an array to a pointer
  ///  type.  'Array' represents the lvalue of the array being decayed
  ///  to a pointer, and the returned SVal represents the decayed
  ///  version of that lvalue (i.e., a pointer to the first element of
  ///  the array).  This is called by GRExprEngine when evaluating
  ///  casts from arrays to pointers.
  SVal ArrayToPointer(Loc Array);

  CastResult CastRegion(const GRState* state, const MemRegion* R,
                        QualType CastToTy);

  SVal EvalBinOp(BinaryOperator::Opcode Op, Loc L, NonLoc R);

  /// The high level logic for this method is this:
  /// Retrieve (L)
  ///   if L has binding
  ///     return L's binding
  ///   else if L is in killset
  ///     return unknown
  ///   else
  ///     if L is on stack or heap
  ///       return undefined
  ///     else
  ///       return symbolic
  SVal Retrieve(const GRState* state, Loc L, QualType T = QualType());

  const GRState* Bind(const GRState* St, Loc LV, SVal V);

  Store Remove(Store store, Loc LV);

  Store getInitialStore() { return RBFactory.GetEmptyMap().getRoot(); }
  
  /// getSelfRegion - Returns the region for the 'self' (Objective-C) or
  ///  'this' object (C++).  When used when analyzing a normal function this
  ///  method returns NULL.
  const MemRegion* getSelfRegion(Store) {
    if (!SelfDecl)
      return 0;
    
    if (!SelfRegion) {
      const ObjCMethodDecl *MD = cast<ObjCMethodDecl>(&StateMgr.getCodeDecl());
      SelfRegion = MRMgr.getObjCObjectRegion(MD->getClassInterface(),
                                             MRMgr.getHeapRegion());
    }
    
    return SelfRegion;
  }
  
  /// RemoveDeadBindings - Scans the RegionStore of 'state' for dead values.
  ///  It returns a new Store with these values removed, and populates LSymbols
  //   and DSymbols with the known set of live and dead symbols respectively.
  Store RemoveDeadBindings(const GRState* state, Stmt* Loc,
                           SymbolReaper& SymReaper,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots);

  const GRState* BindDecl(const GRState* St, const VarDecl* VD, SVal InitVal);

  const GRState* BindDeclWithNoInit(const GRState* St, const VarDecl* VD) {
    return St;
  }

  const GRState* setExtent(const GRState* St, const MemRegion* R, SVal Extent);
  const GRState* setCastType(const GRState* St, const MemRegion* R, QualType T);

  static inline RegionBindingsTy GetRegionBindings(Store store) {
   return RegionBindingsTy(static_cast<const RegionBindingsTy::TreeTy*>(store));
  }

  void print(Store store, std::ostream& Out, const char* nl, const char *sep);

  void iterBindings(Store store, BindingsHandler& f) {
    // FIXME: Implement.
  }

private:
  const GRState* BindArray(const GRState* St, const TypedRegion* R, SVal V);

  /// Retrieve the values in a struct and return a CompoundVal, used when doing
  /// struct copy: 
  /// struct s x, y; 
  /// x = y;
  /// y's value is retrieved by this method.
  SVal RetrieveStruct(const GRState* St, const TypedRegion* R);

  SVal RetrieveArray(const GRState* St, const TypedRegion* R);

  const GRState* BindStruct(const GRState* St, const TypedRegion* R, SVal V);

  /// KillStruct - Set the entire struct to unknown. 
  const GRState* KillStruct(const GRState* St, const TypedRegion* R);

  // Utility methods.
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
  ASTContext& getContext() { return StateMgr.getContext(); }

  SymbolManager& getSymbolManager() { return StateMgr.getSymbolManager(); }

  const GRState* AddRegionView(const GRState* St,
                               const MemRegion* View, const MemRegion* Base);
  const GRState* RemoveRegionView(const GRState* St,
                                  const MemRegion* View, const MemRegion* Base);
};

} // end anonymous namespace

StoreManager* clang::CreateRegionStoreManager(GRStateManager& StMgr) {
  return new RegionStoreManager(StMgr);
}

SubRegionMap* RegionStoreManager::getSubRegionMap(const GRState *state) {
  RegionBindingsTy B = GetRegionBindings(state->getStore());
  RegionStoreSubRegionMap *M = new RegionStoreSubRegionMap();
  
  for (RegionBindingsTy::iterator I=B.begin(), E=B.end(); I!=E; ++I) {
    if (const SubRegion* R = dyn_cast<SubRegion>(I.getKey()))
      M->add(R->getSuperRegion(), R);
  }
  
  return M;
}

/// getLValueString - Returns an SVal representing the lvalue of a
///  StringLiteral.  Within RegionStore a StringLiteral has an
///  associated StringRegion, and the lvalue of a StringLiteral is the
///  lvalue of that region.
SVal RegionStoreManager::getLValueString(const GRState* St, 
                                         const StringLiteral* S) {
  return loc::MemRegionVal(MRMgr.getStringRegion(S));
}

/// getLValueVar - Returns an SVal that represents the lvalue of a
///  variable.  Within RegionStore a variable has an associated
///  VarRegion, and the lvalue of the variable is the lvalue of that region.
SVal RegionStoreManager::getLValueVar(const GRState* St, const VarDecl* VD) {
  return loc::MemRegionVal(MRMgr.getVarRegion(VD));
}

/// getLValueCompoundLiteral - Returns an SVal representing the lvalue
///   of a compound literal.  Within RegionStore a compound literal
///   has an associated region, and the lvalue of the compound literal
///   is the lvalue of that region.
SVal
RegionStoreManager::getLValueCompoundLiteral(const GRState* St,
					     const CompoundLiteralExpr* CL) {
  return loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL));
}

SVal RegionStoreManager::getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                                       SVal Base) {
  return getLValueFieldOrIvar(St, Base, D);
}

SVal RegionStoreManager::getLValueField(const GRState* St, SVal Base,
                                        const FieldDecl* D) {
  return getLValueFieldOrIvar(St, Base, D);
}

SVal RegionStoreManager::getLValueFieldOrIvar(const GRState* St, SVal Base,
                                              const Decl* D) {
  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);
  const MemRegion* BaseR = 0;

  switch (BaseL.getSubKind()) {
  case loc::MemRegionKind:
    BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();
    if (const SymbolicRegion* SR = dyn_cast<SymbolicRegion>(BaseR)) {
      SymbolRef Sym = SR->getSymbol();
      BaseR = MRMgr.getTypedViewRegion(Sym->getType(getContext()), SR);
    }
    break;

  case loc::GotoLabelKind:
    // These are anormal cases. Flag an undefined value.
    return UndefinedVal();

  case loc::ConcreteIntKind:
    // While these seem funny, this can happen through casts.
    // FIXME: What we should return is the field offset.  For example,
    //  add the field offset to the integer value.  That way funny things
    //  like this work properly:  &(((struct foo *) 0xa)->f)
    return Base;

  default:
    assert(0 && "Unhandled Base.");
    return Base;
  }
  
  // NOTE: We must have this check first because ObjCIvarDecl is a subclass
  // of FieldDecl.
  if (const ObjCIvarDecl *ID = dyn_cast<ObjCIvarDecl>(D))
    return loc::MemRegionVal(MRMgr.getObjCIvarRegion(ID, BaseR));

  return loc::MemRegionVal(MRMgr.getFieldRegion(cast<FieldDecl>(D), BaseR));
}

SVal RegionStoreManager::getLValueElement(const GRState* St,
                                          QualType elementType,
                                          SVal Base, SVal Offset) {

  // If the base is an unknown or undefined value, just return it back.
  // FIXME: For absolute pointer addresses, we just return that value back as
  //  well, although in reality we should return the offset added to that
  //  value.
  if (Base.isUnknownOrUndef() || isa<loc::ConcreteInt>(Base))
    return Base;

  // Only handle integer offsets... for now.
  if (!isa<nonloc::ConcreteInt>(Offset))
    return UnknownVal();

  const MemRegion* BaseRegion = cast<loc::MemRegionVal>(Base).getRegion();

  // Pointer of any type can be cast and used as array base.
  const ElementRegion *ElemR = dyn_cast<ElementRegion>(BaseRegion);
  
  if (!ElemR) {
    //
    // If the base region is not an ElementRegion, create one.
    // This can happen in the following example:
    //
    //   char *p = __builtin_alloc(10);
    //   p[1] = 8;
    //
    //  Observe that 'p' binds to an AllocaRegion.
    //

    // Offset might be unsigned. We have to convert it to signed ConcreteInt.
    if (nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&Offset)) {
      const llvm::APSInt& OffI = CI->getValue();
      if (OffI.isUnsigned()) {
        llvm::APSInt Tmp = OffI;
        Tmp.setIsSigned(true);
        Offset = NonLoc::MakeVal(getBasicVals(), Tmp);
      }
    }
    return loc::MemRegionVal(MRMgr.getElementRegion(elementType, Offset,
                                                    BaseRegion));
  }
  
  SVal BaseIdx = ElemR->getIndex();
  
  if (!isa<nonloc::ConcreteInt>(BaseIdx))
    return UnknownVal();
  
  const llvm::APSInt& BaseIdxI = cast<nonloc::ConcreteInt>(BaseIdx).getValue();
  const llvm::APSInt& OffI = cast<nonloc::ConcreteInt>(Offset).getValue();
  assert(BaseIdxI.isSigned());
  
  // FIXME: This appears to be the assumption of this code.  We should review
  // whether or not BaseIdxI.getBitWidth() < OffI.getBitWidth().  If it
  // can't we need to put a comment here.  If it can, we should handle it.
  assert(BaseIdxI.getBitWidth() >= OffI.getBitWidth());

  const MemRegion *ArrayR = ElemR->getSuperRegion();
  SVal NewIdx;
  
  if (OffI.isUnsigned() || OffI.getBitWidth() < BaseIdxI.getBitWidth()) {
    // 'Offset' might be unsigned.  We have to convert it to signed and
    // possibly extend it.
    llvm::APSInt Tmp = OffI;
    
    if (OffI.getBitWidth() < BaseIdxI.getBitWidth())
        Tmp.extend(BaseIdxI.getBitWidth());
    
    Tmp.setIsSigned(true);
    Tmp += BaseIdxI; // Compute the new offset.    
    NewIdx = NonLoc::MakeVal(getBasicVals(), Tmp);    
  }
  else
    NewIdx = nonloc::ConcreteInt(getBasicVals().getValue(BaseIdxI + OffI));

  return loc::MemRegionVal(MRMgr.getElementRegion(elementType, NewIdx, ArrayR));
}

SVal RegionStoreManager::getSizeInElements(const GRState* St,
                                           const MemRegion* R) {
  if (const VarRegion* VR = dyn_cast<VarRegion>(R)) {
    // Get the type of the variable.
    QualType T = VR->getDesugaredValueType(getContext());

    // FIXME: Handle variable-length arrays.
    if (isa<VariableArrayType>(T))
      return UnknownVal();
    
    if (const ConstantArrayType* CAT = dyn_cast<ConstantArrayType>(T)) {
      // return the size as signed integer.
      return NonLoc::MakeVal(getBasicVals(), CAT->getSize(), false);
    }

    GRStateRef state(St, StateMgr);
    const QualType* CastTy = state.get<RegionCasts>(VR);

    // If the VarRegion is cast to other type, compute the size with respect to
    // that type.
    if (CastTy) {
      QualType EleTy =cast<PointerType>(CastTy->getTypePtr())->getPointeeType();
      QualType VarTy = VR->getValueType(getContext());
      uint64_t EleSize = getContext().getTypeSize(EleTy);
      uint64_t VarSize = getContext().getTypeSize(VarTy);
      return NonLoc::MakeIntVal(getBasicVals(), VarSize / EleSize, false);
    }

    // Clients can use ordinary variables as if they were arrays.  These
    // essentially are arrays of size 1.
    return NonLoc::MakeIntVal(getBasicVals(), 1, false);
  }

  if (const StringRegion* SR = dyn_cast<StringRegion>(R)) {
    const StringLiteral* Str = SR->getStringLiteral();
    // We intentionally made the size value signed because it participates in 
    // operations with signed indices.
    return NonLoc::MakeIntVal(getBasicVals(), Str->getByteLength()+1, false);
  }

  if (const TypedViewRegion* ATR = dyn_cast<TypedViewRegion>(R)) {
#if 0
    // FIXME: This logic doesn't really work, as we can have all sorts of
    // weird cases.  For example, this crashes on test case 'rdar-6442306-1.m'.
    // The weird cases come in when arbitrary casting comes into play, violating
    // any type-safe programming.
    
    GRStateRef state(St, StateMgr);

    // Get the size of the super region in bytes.
    const SVal* Extent = state.get<RegionExtents>(ATR->getSuperRegion());
    assert(Extent && "region extent not exist");

    // Assume it's ConcreteInt for now.
    llvm::APSInt SSize = cast<nonloc::ConcreteInt>(*Extent).getValue();

    // Get the size of the element in bits.
    QualType LvT = ATR->getLocationType(getContext());
    QualType ElemTy = cast<PointerType>(LvT.getTypePtr())->getPointeeType();

    uint64_t X = getContext().getTypeSize(ElemTy);

    const llvm::APSInt& ESize = getBasicVals().getValue(X, SSize.getBitWidth(),
                                                        false);

    // Calculate the number of elements. 

    // FIXME: What do we do with signed-ness problem? Shall we make all APSInts
    // signed?
    if (SSize.isUnsigned())
      SSize.setIsSigned(true);

    // FIXME: move this operation into BasicVals.
    const llvm::APSInt S = 
      (SSize * getBasicVals().getValue(8, SSize.getBitWidth(), false)) / ESize;

    return NonLoc::MakeVal(getBasicVals(), S);
#else
    ATR = ATR;
    return UnknownVal();
#endif
  }

  if (const FieldRegion* FR = dyn_cast<FieldRegion>(R)) {
    // FIXME: Unsupported yet.
    FR = 0;
    return UnknownVal();
  }

  if (isa<SymbolicRegion>(R)) {
    return UnknownVal();
  }

  if (isa<AllocaRegion>(R)) {
    return UnknownVal();
  }

  if (isa<ElementRegion>(R)) {
    return UnknownVal();
  }

  assert(0 && "Other regions are not supported yet.");
  return UnknownVal();
}

/// ArrayToPointer - Emulates the "decay" of an array to a pointer
///  type.  'Array' represents the lvalue of the array being decayed
///  to a pointer, and the returned SVal represents the decayed
///  version of that lvalue (i.e., a pointer to the first element of
///  the array).  This is called by GRExprEngine when evaluating casts
///  from arrays to pointers.
SVal RegionStoreManager::ArrayToPointer(Loc Array) {
  if (!isa<loc::MemRegionVal>(Array))
    return UnknownVal();
  
  const MemRegion* R = cast<loc::MemRegionVal>(&Array)->getRegion();
  const TypedRegion* ArrayR = dyn_cast<TypedRegion>(R);
  
  if (!ArrayR)
    return UnknownVal();
  
  // Strip off typedefs from the ArrayRegion's ValueType.
  QualType T = ArrayR->getValueType(getContext())->getDesugaredType();
  ArrayType *AT = cast<ArrayType>(T);
  T = AT->getElementType();
  
  nonloc::ConcreteInt Idx(getBasicVals().getZeroWithPtrWidth(false));
  ElementRegion* ER = MRMgr.getElementRegion(T, Idx, ArrayR);
  
  return loc::MemRegionVal(ER);                    
}

RegionStoreManager::CastResult
RegionStoreManager::CastRegion(const GRState* state, const MemRegion* R,
                               QualType CastToTy) {
  
  ASTContext& Ctx = StateMgr.getContext();

  // We need to know the real type of CastToTy.
  QualType ToTy = Ctx.getCanonicalType(CastToTy);

  // Check cast to ObjCQualifiedID type.
  if (isa<ObjCQualifiedIdType>(ToTy)) {
    // FIXME: Record the type information aside.
    return CastResult(state, R);
  }

  // CodeTextRegion should be cast to only function pointer type.
  if (isa<CodeTextRegion>(R)) {
    assert(CastToTy->isFunctionPointerType() || CastToTy->isBlockPointerType());
    return CastResult(state, R);
  }

  // Now assume we are casting from pointer to pointer. Other cases should
  // already be handled.
  QualType PointeeTy = cast<PointerType>(ToTy.getTypePtr())->getPointeeType();

  // Process region cast according to the kind of the region being cast.
  

  // FIXME: Need to handle arbitrary downcasts.
  // FIXME: Handle the case where a TypedViewRegion (layering a SymbolicRegion
  //         or an AllocaRegion is cast to another view, thus causing the memory
  //         to be re-used for a different purpose.

  if (isa<SymbolicRegion>(R) || isa<AllocaRegion>(R)) {
    state = setCastType(state, R, ToTy);
    return CastResult(state, R);
  }

  // VarRegion, ElementRegion, and FieldRegion has an inherent type. Normally
  // they should not be cast. We only layer an ElementRegion when the cast-to
  // pointee type is of smaller size. In other cases, we return the original
  // VarRegion.
  if (isa<VarRegion>(R) || isa<ElementRegion>(R) || isa<FieldRegion>(R)
      || isa<ObjCIvarRegion>(R) || isa<CompoundLiteralRegion>(R)) {
    // If the pointee type is incomplete, do not compute its size, and return
    // the original region.
    if (const RecordType *RT = dyn_cast<RecordType>(PointeeTy.getTypePtr())) {
      const RecordDecl *D = RT->getDecl();
      if (!D->getDefinition(getContext()))
        return CastResult(state, R);
    }

    QualType ObjTy = cast<TypedRegion>(R)->getValueType(getContext());
    uint64_t PointeeTySize = getContext().getTypeSize(PointeeTy);
    uint64_t ObjTySize = getContext().getTypeSize(ObjTy);

    if (PointeeTySize > 0 && PointeeTySize < ObjTySize) {
      // Record the cast type of the region.
      state = setCastType(state, R, ToTy);

      SVal Idx = ValMgr.makeZeroArrayIndex();
      ElementRegion* ER = MRMgr.getElementRegion(PointeeTy, Idx, R);
      return CastResult(state, ER);
    } else
      return CastResult(state, R);
  }

  if (isa<TypedViewRegion>(R)) {
    const MemRegion* ViewR = MRMgr.getTypedViewRegion(CastToTy, R);  
    return CastResult(state, ViewR);
  }

  if (isa<ObjCObjectRegion>(R)) {
    return CastResult(state, R);
  }

  assert(0 && "Unprocessed region.");
}

SVal RegionStoreManager::EvalBinOp(BinaryOperator::Opcode Op, Loc L, NonLoc R) {
  // Assume the base location is MemRegionVal(ElementRegion).
  if (!isa<loc::MemRegionVal>(L))
    return UnknownVal();

  const MemRegion* MR = cast<loc::MemRegionVal>(L).getRegion();
  if (isa<SymbolicRegion>(MR))
    return UnknownVal();

  const TypedRegion* TR = cast<TypedRegion>(MR);
  const ElementRegion* ER = dyn_cast<ElementRegion>(TR);
  
  if (!ER) {
    // If the region is not element region, create one with index 0. This can
    // happen in the following example:
    // char *p = foo();
    // p += 3;
    // Note that p binds to a TypedViewRegion(SymbolicRegion).
    nonloc::ConcreteInt Idx(getBasicVals().getZeroWithPtrWidth(false));
    ER = MRMgr.getElementRegion(TR->getValueType(getContext()), Idx, TR);
  }

  SVal Idx = ER->getIndex();

  nonloc::ConcreteInt* Base = dyn_cast<nonloc::ConcreteInt>(&Idx);
  nonloc::ConcreteInt* Offset = dyn_cast<nonloc::ConcreteInt>(&R);

  // Only support concrete integer indexes for now.
  if (Base && Offset) {
    // FIXME: For now, convert the signedness and bitwidth of offset in case
    //  they don't match.  This can result from pointer arithmetic.  In reality,
    //  we should figure out what are the proper semantics and implement them.
    // 
    //  This addresses the test case test/Analysis/ptr-arith.c
    //
    nonloc::ConcreteInt OffConverted(getBasicVals().Convert(Base->getValue(),
                                                           Offset->getValue()));
    SVal NewIdx = Base->EvalBinOp(getBasicVals(), Op, OffConverted);
    const MemRegion* NewER =
      MRMgr.getElementRegion(ER->getElementType(), NewIdx, 
                             cast<TypedRegion>(ER->getSuperRegion()));
    return Loc::MakeVal(NewER);

  }
  
  return UnknownVal();
}

SVal RegionStoreManager::Retrieve(const GRState* St, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");

  // FIXME: Is this even possible?  Shouldn't this be treated as a null
  //  dereference at a higher level?
  if (isa<loc::ConcreteInt>(L))
    return UndefinedVal();

  const MemRegion* MR = cast<loc::MemRegionVal>(L).getRegion();

  // We return unknown for symbolic region for now. This might be improved.
  // Example:
  // void f(int* p) { int x = *p; }
  if (isa<SymbolicRegion>(MR))
    return UnknownVal();

  // FIXME: Perhaps this method should just take a 'const MemRegion*' argument
  //  instead of 'Loc', and have the other Loc cases handled at a higher level.
  const TypedRegion* R = cast<TypedRegion>(MR);
  assert(R && "bad region");

  // FIXME: We should eventually handle funny addressing.  e.g.:
  //
  //   int x = ...;
  //   int *p = &x;
  //   char *q = (char*) p;
  //   char c = *q;  // returns the first byte of 'x'.
  //
  // Such funny addressing will occur due to layering of regions.

  QualType RTy = R->getValueType(getContext());

  if (RTy->isStructureType())
    return RetrieveStruct(St, R);

  if (RTy->isArrayType())
    return RetrieveArray(St, R);

  // FIXME: handle Vector types.
  if (RTy->isVectorType())
      return UnknownVal();
  
  RegionBindingsTy B = GetRegionBindings(St->getStore());
  RegionBindingsTy::data_type* V = B.lookup(R);

  // Check if the region has a binding.
  if (V)
    return *V;

  GRStateRef state(St, StateMgr);
  
  // Check if the region is in killset.
  if (state.contains<RegionKills>(R))
    return UnknownVal();

  // If the region is an element or field, it may have a default value.
  if (isa<ElementRegion>(R) || isa<FieldRegion>(R)) {
    const MemRegion* SuperR = cast<SubRegion>(R)->getSuperRegion();
    GRStateTrait<RegionDefaultValue>::lookup_type D = 
      state.get<RegionDefaultValue>(SuperR);
    if (D)
      return *D;
  }
  
  if (const ObjCIvarRegion *IVR = dyn_cast<ObjCIvarRegion>(R)) {
    const MemRegion *SR = IVR->getSuperRegion();

    // If the super region is 'self' then return the symbol representing
    // the value of the ivar upon entry to the method.
    if (SR == SelfRegion) {
      // FIXME: Do we need to handle the case where the super region
      // has a view?  We want to canonicalize the bindings.
      return ValMgr.getRegionValueSymbolVal(R);
    }
    
    // Otherwise, we need a new symbol.  For now return Unknown.
    return UnknownVal();
  }

  // The location does not have a bound value.  This means that it has
  // the value it had upon its creation and/or entry to the analyzed
  // function/method.  These are either symbolic values or 'undefined'.

  // We treat function parameters as symbolic values.
  if (const VarRegion* VR = dyn_cast<VarRegion>(R)) {
    const VarDecl *VD = VR->getDecl();
    
    if (VD == SelfDecl)
      return loc::MemRegionVal(getSelfRegion(0));
    
    if (isa<ParmVarDecl>(VD) || isa<ImplicitParamDecl>(VD) ||
        VD->hasGlobalStorage()) {
      QualType VTy = VD->getType();
      if (Loc::IsLocType(VTy) || VTy->isIntegerType())
        return ValMgr.getRegionValueSymbolVal(VR);
      else
        return UnknownVal();
    }
  }  

  if (MRMgr.onStack(R) || MRMgr.onHeap(R)) {
    // All stack variables are considered to have undefined values
    // upon creation.  All heap allocated blocks are considered to
    // have undefined values as well unless they are explicitly bound
    // to specific values.
    return UndefinedVal();
  }

  // All other integer values are symbolic.
  if (Loc::IsLocType(RTy) || RTy->isIntegerType())
    return ValMgr.getRegionValueSymbolVal(R);
  else
    return UnknownVal();
}

SVal RegionStoreManager::RetrieveStruct(const GRState* St,const TypedRegion* R){
  QualType T = R->getValueType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  llvm::ImmutableList<SVal> StructVal = getBasicVals().getEmptySValList();

  std::vector<FieldDecl *> Fields(RD->field_begin(getContext()), 
                                  RD->field_end(getContext()));

  for (std::vector<FieldDecl *>::reverse_iterator Field = Fields.rbegin(),
                                               FieldEnd = Fields.rend();
       Field != FieldEnd; ++Field) {
    FieldRegion* FR = MRMgr.getFieldRegion(*Field, R);
    QualType FTy = (*Field)->getType();
    SVal FieldValue = Retrieve(St, loc::MemRegionVal(FR), FTy);
    StructVal = getBasicVals().consVals(FieldValue, StructVal);
  }

  return NonLoc::MakeCompoundVal(T, StructVal, getBasicVals());
}

SVal RegionStoreManager::RetrieveArray(const GRState* St, const TypedRegion* R){
  QualType T = R->getValueType(getContext());
  ConstantArrayType* CAT = cast<ConstantArrayType>(T.getTypePtr());

  llvm::ImmutableList<SVal> ArrayVal = getBasicVals().getEmptySValList();
  llvm::APSInt Size(CAT->getSize(), false);
  llvm::APSInt i = getBasicVals().getValue(0, Size.getBitWidth(), 
					   Size.isUnsigned());

  for (; i < Size; ++i) {
    SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
    ElementRegion* ER = MRMgr.getElementRegion(R->getValueType(getContext()),
                                               Idx, R);
    QualType ETy = ER->getElementType();
    SVal ElementVal = Retrieve(St, loc::MemRegionVal(ER), ETy);
    ArrayVal = getBasicVals().consVals(ElementVal, ArrayVal);
  }

  return NonLoc::MakeCompoundVal(T, ArrayVal, getBasicVals());
}

const GRState* RegionStoreManager::Bind(const GRState* St, Loc L, SVal V) {
  // If we get here, the location should be a region.
  const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
  assert(R);

  // Check if the region is a struct region.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    if (TR->getValueType(getContext())->isStructureType())
      return BindStruct(St, TR, V);

  Store store = St->getStore();
  RegionBindingsTy B = GetRegionBindings(store);

  if (V.isUnknown()) {
    // Remove the binding.
    store = RBFactory.Remove(B, R).getRoot();

    // Add the region to the killset.
    GRStateRef state(St, StateMgr);
    St = state.add<RegionKills>(R);
  } 
  else
    store = RBFactory.Add(B, R, V).getRoot();

  return StateMgr.MakeStateWithStore(St, store);
}

Store RegionStoreManager::Remove(Store store, Loc L) {
  const MemRegion* R = 0;
  
  if (isa<loc::MemRegionVal>(L))
    R = cast<loc::MemRegionVal>(L).getRegion();
  
  if (R) {
    RegionBindingsTy B = GetRegionBindings(store);  
    return RBFactory.Remove(B, R).getRoot();
  }
  
  return store;
}

const GRState* RegionStoreManager::BindDecl(const GRState* St, 
                                            const VarDecl* VD, SVal InitVal) {

  QualType T = VD->getType();
  VarRegion* VR = MRMgr.getVarRegion(VD);

  if (T->isArrayType())
    return BindArray(St, VR, InitVal);
  if (T->isStructureType())
    return BindStruct(St, VR, InitVal);

  return Bind(St, Loc::MakeVal(VR), InitVal);
}

// FIXME: this method should be merged into Bind().
const GRState* 
RegionStoreManager::BindCompoundLiteral(const GRState* St,
                                        const CompoundLiteralExpr* CL, SVal V) {
  CompoundLiteralRegion* R = MRMgr.getCompoundLiteralRegion(CL);
  return Bind(St, loc::MemRegionVal(R), V);
}

const GRState* RegionStoreManager::setExtent(const GRState* St,
                                             const MemRegion* R, SVal Extent) {
  GRStateRef state(St, StateMgr);
  return state.set<RegionExtents>(R, Extent);
}


static void UpdateLiveSymbols(SVal X, SymbolReaper& SymReaper) {
  if (loc::MemRegionVal *XR = dyn_cast<loc::MemRegionVal>(&X)) {
    const MemRegion *R = XR->getRegion();
    
    while (R) {
      if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R)) {
        SymReaper.markLive(SR->getSymbol());
        return;
      }
      
      if (const SubRegion *SR = dyn_cast<SubRegion>(R)) {
        R = SR->getSuperRegion();
        continue;
      }
      
      break;
    }
    
    return;
  }
  
  for (SVal::symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end();SI!=SE;++SI)
    SymReaper.markLive(*SI);
}

Store RegionStoreManager::RemoveDeadBindings(const GRState* state, Stmt* Loc, 
                                             SymbolReaper& SymReaper,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots)
{

  Store store = state->getStore();
  RegionBindingsTy B = GetRegionBindings(store);
  
  // Lazily constructed backmap from MemRegions to SubRegions.
  typedef llvm::ImmutableSet<const MemRegion*> SubRegionsTy;
  typedef llvm::ImmutableMap<const MemRegion*, SubRegionsTy> SubRegionsMapTy;
  
  // FIXME: As a future optimization we can modifiy BumpPtrAllocator to have
  // the ability to reuse memory.  This way we can keep TmpAlloc around as
  // an instance variable of RegionStoreManager (avoiding repeated malloc
  // overhead).
  llvm::BumpPtrAllocator TmpAlloc;
  
  // Factory objects.
  SubRegionsMapTy::Factory SubRegMapF(TmpAlloc);
  SubRegionsTy::Factory SubRegF(TmpAlloc);
  
  // The backmap from regions to subregions.
  SubRegionsMapTy SubRegMap = SubRegMapF.GetEmptyMap();
  
  // Do a pass over the regions in the store.  For VarRegions we check if
  // the variable is still live and if so add it to the list of live roots.
  // For other regions we populate our region backmap.

  llvm::SmallVector<const MemRegion*, 10> IntermediateRoots;

  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    IntermediateRoots.push_back(I.getKey());
  }

  while (!IntermediateRoots.empty()) {
    const MemRegion* R = IntermediateRoots.back();
    IntermediateRoots.pop_back();

    if (const VarRegion* VR = dyn_cast<VarRegion>(R)) {
      if (SymReaper.isLive(Loc, VR->getDecl()))
        RegionRoots.push_back(VR); // This is a live "root".
    } 
    else if (const SymbolicRegion* SR = dyn_cast<SymbolicRegion>(R)) {
      if (SymReaper.isLive(SR->getSymbol()))
        RegionRoots.push_back(SR);
    }
    else {
      // Get the super region for R.
      const MemRegion* SuperR = cast<SubRegion>(R)->getSuperRegion();

      // Get the current set of subregions for SuperR.
      const SubRegionsTy* SRptr = SubRegMap.lookup(SuperR);
      SubRegionsTy SRs = SRptr ? *SRptr : SubRegF.GetEmptySet();

      // Add R to the subregions of SuperR.
      SubRegMap = SubRegMapF.Add(SubRegMap, SuperR, SubRegF.Add(SRs, R));

      // Super region may be VarRegion or subregion of another VarRegion. Add it
      // to the work list.
      if (isa<SubRegion>(SuperR))
        IntermediateRoots.push_back(SuperR);
    }
  }
  
  // Process the worklist of RegionRoots.  This performs a "mark-and-sweep"
  // of the store.  We want to find all live symbols and dead regions.  
  llvm::SmallPtrSet<const MemRegion*, 10> Marked;
  
  while (!RegionRoots.empty()) {
    // Dequeue the next region on the worklist.
    const MemRegion* R = RegionRoots.back();
    RegionRoots.pop_back();

    // Check if we have already processed this region.
    if (Marked.count(R)) continue;

    // Mark this region as processed.  This is needed for termination in case
    // a region is referenced more than once.
    Marked.insert(R);
    
    // Mark the symbol for any live SymbolicRegion as "live".  This means we
    // should continue to track that symbol.
    if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(R))
      SymReaper.markLive(SymR->getSymbol());

    // Get the data binding for R (if any).
    RegionBindingsTy::data_type* Xptr = B.lookup(R);
    if (Xptr) {
      SVal X = *Xptr;
      UpdateLiveSymbols(X, SymReaper); // Update the set of live symbols.
    
      // If X is a region, then add it the RegionRoots.
      if (loc::MemRegionVal* RegionX = dyn_cast<loc::MemRegionVal>(&X))
        RegionRoots.push_back(RegionX->getRegion());
    }
    
    // Get the subregions of R.  These are RegionRoots as well since they
    // represent values that are also bound to R.
    const SubRegionsTy* SRptr = SubRegMap.lookup(R);      
    if (!SRptr) continue;
    SubRegionsTy SR = *SRptr;
    
    for (SubRegionsTy::iterator I=SR.begin(), E=SR.end(); I!=E; ++I)
      RegionRoots.push_back(*I);
  }
  
  // We have now scanned the store, marking reachable regions and symbols
  // as live.  We now remove all the regions that are dead from the store
  // as well as update DSymbols with the set symbols that are now dead.  
  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const MemRegion* R = I.getKey();
    
    // If this region live?  Is so, none of its symbols are dead.
    if (Marked.count(R))
      continue;
    
    // Remove this dead region from the store.
    store = Remove(store, Loc::MakeVal(R));

    // Mark all non-live symbols that this region references as dead.
    if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(R))
      SymReaper.maybeDead(SymR->getSymbol());

    SVal X = I.getData();
    SVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
    for (; SI != SE; ++SI) SymReaper.maybeDead(*SI);
  }
  
  return store;
}

void RegionStoreManager::print(Store store, std::ostream& Out, 
                               const char* nl, const char *sep) {
  llvm::raw_os_ostream OS(Out);
  RegionBindingsTy B = GetRegionBindings(store);
  OS << "Store:" << nl;

  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    OS << ' '; I.getKey()->print(OS); OS << " : ";
    I.getData().print(OS); OS << nl;
  }
}

const GRState* RegionStoreManager::BindArray(const GRState* St, 
                                             const TypedRegion* R, SVal Init) {
  QualType T = R->getValueType(getContext());
  assert(T->isArrayType());

  // When we are binding the whole array, it always has default value 0.
  GRStateRef state(St, StateMgr);
  St = state.set<RegionDefaultValue>(R, NonLoc::MakeIntVal(getBasicVals(), 0, 
                                                           false));

  ConstantArrayType* CAT = cast<ConstantArrayType>(T.getTypePtr());

  llvm::APSInt Size(CAT->getSize(), false);
  llvm::APSInt i = getBasicVals().getValue(0, Size.getBitWidth(),
                                           Size.isUnsigned());

  // Check if the init expr is a StringLiteral.
  if (isa<loc::MemRegionVal>(Init)) {
    const MemRegion* InitR = cast<loc::MemRegionVal>(Init).getRegion();
    const StringLiteral* S = cast<StringRegion>(InitR)->getStringLiteral();
    const char* str = S->getStrData();
    unsigned len = S->getByteLength();
    unsigned j = 0;

    // Copy bytes from the string literal into the target array. Trailing bytes
    // in the array that are not covered by the string literal are initialized
    // to zero.
    for (; i < Size; ++i, ++j) {
      if (j >= len)
        break;

      SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
      ElementRegion* ER =
        MRMgr.getElementRegion(cast<ArrayType>(T)->getElementType(),
                               Idx, R);

      SVal V = NonLoc::MakeVal(getBasicVals(), str[j], sizeof(char)*8, true);
      St = Bind(St, loc::MemRegionVal(ER), V);
    }

    return St;
  }

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  for (; i < Size; ++i, ++VI) {
    // The init list might be shorter than the array decl.
    if (VI == VE)
      break;

    SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
    ElementRegion* ER =
      MRMgr.getElementRegion(cast<ArrayType>(T)->getElementType(),
                             Idx, R);

    if (CAT->getElementType()->isStructureType())
      St = BindStruct(St, ER, *VI);
    else
      St = Bind(St, Loc::MakeVal(ER), *VI);
  }

  return St;
}

const GRState*
RegionStoreManager::BindStruct(const GRState* St, const TypedRegion* R, SVal V){
  QualType T = R->getValueType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = T->getAsRecordType();
  RecordDecl* RD = RT->getDecl();

  if (!RD->isDefinition())
    return St;

  if (V.isUnknown())
    return KillStruct(St, R);

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  RecordDecl::field_iterator FI = RD->field_begin(getContext()), 
                             FE = RD->field_end(getContext());

  for (; FI != FE; ++FI, ++VI) {

    // There may be fewer values than fields only when we are initializing a
    // struct decl. In this case, mark the region as having default value.
    if (VI == VE) {
      GRStateRef state(St, StateMgr);
      const NonLoc& Idx = NonLoc::MakeIntVal(getBasicVals(), 0, false);
      St = state.set<RegionDefaultValue>(R, Idx);
      break;
    }

    QualType FTy = (*FI)->getType();
    FieldRegion* FR = MRMgr.getFieldRegion(*FI, R);

    if (Loc::IsLocType(FTy) || FTy->isIntegerType())
      St = Bind(St, Loc::MakeVal(FR), *VI);
    
    else if (FTy->isArrayType())
      St = BindArray(St, FR, *VI);

    else if (FTy->isStructureType())
      St = BindStruct(St, FR, *VI);
  }

  return St;
}

const GRState* RegionStoreManager::KillStruct(const GRState* St,
                                              const TypedRegion* R){
  GRStateRef state(St, StateMgr);

  // Kill the struct region because it is assigned "unknown".
  St = state.add<RegionKills>(R);
  
  // Set the default value of the struct region to "unknown".
  St = state.set<RegionDefaultValue>(R, UnknownVal());

  Store store = St->getStore();
  RegionBindingsTy B = GetRegionBindings(store);

  // Remove all bindings for the subregions of the struct.
  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const MemRegion* r = I.getKey();
    if (const SubRegion* sr = dyn_cast<SubRegion>(r))
      if (sr->isSubRegionOf(R))
        store = Remove(store, Loc::MakeVal(sr));
    // FIXME: Maybe we should also remove the bindings for the "views" of the
    // subregions.
  }

  return StateMgr.MakeStateWithStore(St, store);
}

const GRState* RegionStoreManager::AddRegionView(const GRState* St,
                                                 const MemRegion* View,
                                                 const MemRegion* Base) {
  GRStateRef state(St, StateMgr);

  // First, retrieve the region view of the base region.
  const RegionViews* d = state.get<RegionViewMap>(Base);
  RegionViews L = d ? *d : RVFactory.GetEmptySet();

  // Now add View to the region view.
  L = RVFactory.Add(L, View);

  // Create a new state with the new region view.
  return state.set<RegionViewMap>(Base, L);
}

const GRState* RegionStoreManager::RemoveRegionView(const GRState* St,
                                                    const MemRegion* View,
                                                    const MemRegion* Base) {
  GRStateRef state(St, StateMgr);

  // Retrieve the region view of the base region.
  const RegionViews* d = state.get<RegionViewMap>(Base);

  // If the base region has no view, return.
  if (!d)
    return St;

  // Remove the view.
  RegionViews V = *d;
  V = RVFactory.Remove(V, View);

  return state.set<RegionViewMap>(Base, V);
}

const GRState* RegionStoreManager::setCastType(const GRState* St,
                                               const MemRegion* R, QualType T) {
  GRStateRef state(St, StateMgr);
  return state.set<RegionCasts>(R, T);
}
