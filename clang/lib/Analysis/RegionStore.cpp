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
typedef llvm::ImmutableList<const MemRegion*> RegionViews;
namespace { class VISIBILITY_HIDDEN RegionViewMap {}; }
static int RegionViewMapIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionViewMap> 
    : public GRStatePartialTrait<llvm::ImmutableMap<const MemRegion*,
                                                    RegionViews> > {
                                                      
    static void* GDMIndex() { return &RegionViewMapIndex; }
  };
}

//===----------------------------------------------------------------------===//
// Region "Extents"
//===----------------------------------------------------------------------===//
//
//  MemRegions represent chunks of memory with a size (their "extent").  This
//  GDM entry tracks the extents for regions.  Extents are in bytes.
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
// RegionStore lazily adds value bindings to regions when the analyzer
//  handles assignment statements.  Killsets track which default values have
//  been killed, thus distinguishing between "unknown" values and default
//  values.
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
// Regions with default values of '0'.
//===----------------------------------------------------------------------===//
//
// This GDM entry tracks what regions have a default value of 0 if they
// have no bound value and have not been killed.
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

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  RegionViews::Factory RVFactory;

  GRStateManager& StateMgr;
  MemRegionManager MRMgr;

public:
  RegionStoreManager(GRStateManager& mgr) 
    : RBFactory(mgr.getAllocator()),
      RVFactory(mgr.getAllocator()),
      StateMgr(mgr), 
      MRMgr(StateMgr.getAllocator()) {}

  virtual ~RegionStoreManager() {}

  MemRegionManager& getRegionManager() { return MRMgr; }
  
  const GRState* BindCompoundLiteral(const GRState* St, 
                                     const CompoundLiteralExpr* CL, SVal V);

  SVal getLValueString(const GRState* St, const StringLiteral* S);

  SVal getLValueCompoundLiteral(const GRState* St, const CompoundLiteralExpr*);

  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);

  SVal getLValueField(const GRState* St, SVal Base, const FieldDecl* D);

  SVal getLValueElement(const GRState* St, SVal Base, SVal Offset);

  SVal getSizeInElements(const GRState* St, const MemRegion* R);

  SVal ArrayToPointer(SVal Array);

  /// CastRegion - Used by GRExprEngine::VisitCast to handle casts from
  ///  a MemRegion* to a specific location type.  'R' is the region being
  ///  casted and 'CastToTy' the result type of the cast.  
  CastResult CastRegion(const GRState* state, const MemRegion* R,
                        QualType CastToTy);

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
    assert (false && "Not implemented.");
    return 0;
  }
  
  /// RemoveDeadBindings - Scans the RegionStore of 'state' for dead values.
  ///  It returns a new Store with these values removed, and populates LSymbols
  //   and DSymbols with the known set of live and dead symbols respectively.
  Store RemoveDeadBindings(const GRState* state, Stmt* Loc,
                           const LiveVariables& Live,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                           LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols);

  void UpdateLiveSymbols(SVal X, LiveSymbolsTy& LSymbols);

  const GRState* BindDecl(const GRState* St, const VarDecl* VD, SVal InitVal);

  const GRState* BindDeclWithNoInit(const GRState* St, const VarDecl* VD) {
    return St;
  }

  const GRState* setExtent(const GRState* St, const MemRegion* R, SVal Extent);

  static inline RegionBindingsTy GetRegionBindings(Store store) {
   return RegionBindingsTy(static_cast<const RegionBindingsTy::TreeTy*>(store));
  }

  void print(Store store, std::ostream& Out, const char* nl, const char *sep);

  void iterBindings(Store store, BindingsHandler& f) {
    // FIXME: Implement.
  }

private:
  Loc getVarLoc(const VarDecl* VD) {
    return loc::MemRegionVal(MRMgr.getVarRegion(VD));
  }

  const GRState* BindArray(const GRState* St, const TypedRegion* R, SVal V);

  /// Retrieve the values in a struct and return a CompoundVal, used when doing
  /// struct copy: 
  /// struct s x, y; 
  /// x = y;
  /// y's value is retrieved by this method.
  SVal RetrieveStruct(const GRState* St, const TypedRegion* R);

  const GRState* BindStruct(const GRState* St, const TypedRegion* R, SVal V);

  // Utility methods.
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
  ASTContext& getContext() { return StateMgr.getContext(); }
  SymbolManager& getSymbolManager() { return StateMgr.getSymbolManager(); }

  const GRState* AddRegionView(const GRState* St,
                               const MemRegion* View, const MemRegion* Base);
};

} // end anonymous namespace

StoreManager* clang::CreateRegionStoreManager(GRStateManager& StMgr) {
  return new RegionStoreManager(StMgr);
}

SVal RegionStoreManager::getLValueString(const GRState* St, 
                                         const StringLiteral* S) {
  return loc::MemRegionVal(MRMgr.getStringRegion(S));
}

SVal RegionStoreManager::getLValueVar(const GRState* St, const VarDecl* VD) {
  return loc::MemRegionVal(MRMgr.getVarRegion(VD));
}

SVal RegionStoreManager::getLValueCompoundLiteral(const GRState* St,
                                                const CompoundLiteralExpr* CL) {
  return loc::MemRegionVal(MRMgr.getCompoundLiteralRegion(CL));
}

SVal RegionStoreManager::getLValueIvar(const GRState* St, const ObjCIvarDecl* D,
                                       SVal Base) {
  return UnknownVal();
}

SVal RegionStoreManager::getLValueField(const GRState* St, SVal Base,
                                        const FieldDecl* D) {
  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);
  const MemRegion* BaseR = 0;

  switch (BaseL.getSubKind()) {
  case loc::MemRegionKind:
    BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();
    break;

  case loc::SymbolValKind:
    BaseR = MRMgr.getSymbolicRegion(cast<loc::SymbolVal>(&BaseL)->getSymbol());
    break;
  
  case loc::GotoLabelKind:
  case loc::FuncValKind:
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

  return loc::MemRegionVal(MRMgr.getFieldRegion(D, BaseR));
}

SVal RegionStoreManager::getLValueElement(const GRState* St, 
                                          SVal Base, SVal Offset) {
  if (Base.isUnknownOrUndef())
    return Base;

  if (isa<loc::SymbolVal>(Base))
    return Base;

  loc::MemRegionVal& BaseL = cast<loc::MemRegionVal>(Base);

  // Pointer of any type can be cast and used as array base. We do not support
  // that case yet.
  if (!isa<ElementRegion>(BaseL.getRegion())) {
    // Record what we have seen in real code.
    assert(isa<FieldRegion>(BaseL.getRegion()));
    return UnknownVal();
  }

  // We expect BaseR is an ElementRegion, not a base VarRegion.

  const ElementRegion* ElemR = cast<ElementRegion>(BaseL.getRegion());

  SVal Idx = ElemR->getIndex();

  nonloc::ConcreteInt *CI1, *CI2;

  // Only handle integer indices for now.
  if ((CI1 = dyn_cast<nonloc::ConcreteInt>(&Idx)) &&
      (CI2 = dyn_cast<nonloc::ConcreteInt>(&Offset))) {

    // Temporary SVal to hold a potential signed and extended APSInt.
    SVal SignedInt;

    // Index might be unsigned. We have to convert it to signed. It might also
    // be less wide than the size. We have to extend it.
    if (CI2->getValue().isUnsigned() ||
        CI2->getValue().getBitWidth() < CI1->getValue().getBitWidth()) {
      llvm::APSInt SI = CI2->getValue();
      if (CI2->getValue().getBitWidth() < CI1->getValue().getBitWidth())
        SI.extend(CI1->getValue().getBitWidth());
      SI.setIsSigned(true);
      SignedInt = nonloc::ConcreteInt(getBasicVals().getValue(SI));
      CI2 = cast<nonloc::ConcreteInt>(&SignedInt);
    }

    SVal NewIdx = CI1->EvalBinOp(getBasicVals(), BinaryOperator::Add, *CI2);
    return loc::MemRegionVal(MRMgr.getElementRegion(NewIdx, 
                                                    ElemR->getArrayRegion()));
  }

  return UnknownVal();
}

SVal RegionStoreManager::getSizeInElements(const GRState* St,
                                           const MemRegion* R) {
  if (const VarRegion* VR = dyn_cast<VarRegion>(R)) {
    // Get the type of the variable.
    QualType T = VR->getRValueType(getContext());

    // It must be of array type. 
    const ConstantArrayType* CAT = cast<ConstantArrayType>(T.getTypePtr());

    // return the size as signed integer.
    return NonLoc::MakeVal(getBasicVals(), CAT->getSize(), false);
  }

  if (const StringRegion* SR = dyn_cast<StringRegion>(R)) {
    const StringLiteral* Str = SR->getStringLiteral();
    // We intentionally made the size value signed because it participates in 
    // operations with signed indices.
    return NonLoc::MakeVal(getBasicVals(), Str->getByteLength() + 1, false);
  }

  if (const AnonTypedRegion* ATR = dyn_cast<AnonTypedRegion>(R)) {
    GRStateRef state(St, StateMgr);

    // Get the size of the super region in bytes.
    const SVal* Extent = state.get<RegionExtents>(ATR->getSuperRegion());
    assert(Extent && "region extent not exist");

    // Assume it's ConcreteInt for now.
    llvm::APSInt SSize = cast<nonloc::ConcreteInt>(*Extent).getValue();

    // Get the size of the element in bits.
    QualType LvT = ATR->getLValueType(getContext());
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
  }

  if (const FieldRegion* FR = dyn_cast<FieldRegion>(R)) {
    // FIXME: Unsupported yet.
    FR = 0;
    return UnknownVal();
  }

  assert(0 && "Other regions are not supported yet.");
}

// Cast 'pointer to array' to 'pointer to the first element of array'.

SVal RegionStoreManager::ArrayToPointer(SVal Array) {
  if (Array.isUnknownOrUndef())
    return Array;
  
  if (!isa<loc::MemRegionVal>(Array))
    return UnknownVal();
  
  const MemRegion* R = cast<loc::MemRegionVal>(&Array)->getRegion();
  const TypedRegion* ArrayR = dyn_cast<TypedRegion>(R);
  
  if (ArrayR)
    return UnknownVal();
  
  nonloc::ConcreteInt Idx(getBasicVals().getZeroWithPtrWidth(false));
  ElementRegion* ER = MRMgr.getElementRegion(Idx, ArrayR);
  
  return loc::MemRegionVal(ER);                    
}

StoreManager::CastResult
RegionStoreManager::CastRegion(const GRState* state, const MemRegion* R,
                               QualType CastToTy) {
  
  // Return the same region if the region types are compatible.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    ASTContext& Ctx = StateMgr.getContext();
    QualType Ta = Ctx.getCanonicalType(TR->getLValueType(Ctx));
    QualType Tb = Ctx.getCanonicalType(CastToTy);
    
    if (Ta == Tb)
      return CastResult(state, R);
  }
  
  const MemRegion* ViewR = MRMgr.getAnonTypedRegion(CastToTy, R);  
  return CastResult(AddRegionView(state, ViewR, R), ViewR);
}

SVal RegionStoreManager::Retrieve(const GRState* St, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");

  if (isa<loc::SymbolVal>(L))
    return UnknownVal();

  if (isa<loc::ConcreteInt>(L))
    return UndefinedVal();

  if (isa<loc::FuncVal>(L))
    return L;

  const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
  assert(R && "bad region");

  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    if (TR->getRValueType(getContext())->isStructureType())
      return RetrieveStruct(St, TR);
  
  RegionBindingsTy B = GetRegionBindings(St->getStore());
  RegionBindingsTy::data_type* V = B.lookup(R);

  // Check if the region has a binding.
  if (V)
    return *V;
  
  // Check if the region is in killset.
  GRStateRef state(St, StateMgr);
  if (state.contains<RegionKills>(R))
    return UnknownVal();

  // The location is not initialized.
  
  // We treat parameters as symbolic values.
  if (const VarRegion* VR = dyn_cast<VarRegion>(R))
    if (isa<ParmVarDecl>(VR->getDecl()))
      return SVal::MakeSymbolValue(getSymbolManager(), VR,
                                   VR->getRValueType(getContext()));
  
  if (MRMgr.onStack(R) || MRMgr.onHeap(R))
    return UndefinedVal();
  else
    return SVal::MakeSymbolValue(getSymbolManager(), R, 
                             cast<TypedRegion>(R)->getRValueType(getContext()));

  // FIXME: consider default values for elements and fields.
}

SVal RegionStoreManager::RetrieveStruct(const GRState* St,const TypedRegion* R){

  Store store = St->getStore();
  GRStateRef state(St, StateMgr);

  // FIXME: Verify we want getRValueType instead of getLValueType.
  QualType T = R->getRValueType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  llvm::ImmutableList<SVal> StructVal = getBasicVals().getEmptySValList();

  std::vector<FieldDecl *> Fields(RD->field_begin(), RD->field_end());

  for (std::vector<FieldDecl *>::reverse_iterator Field = Fields.rbegin(),
                                               FieldEnd = Fields.rend();
       Field != FieldEnd; ++Field) {
    FieldRegion* FR = MRMgr.getFieldRegion(*Field, R);
    RegionBindingsTy B = GetRegionBindings(store);
    RegionBindingsTy::data_type* data = B.lookup(FR);

    SVal FieldValue;
    if (data)
      FieldValue = *data;
    else if (state.contains<RegionKills>(FR))
      FieldValue = UnknownVal();
    else {
      if (MRMgr.onStack(FR) || MRMgr.onHeap(FR))
        FieldValue = UndefinedVal();
      else
        FieldValue = SVal::MakeSymbolValue(getSymbolManager(), FR,
                                           FR->getRValueType(getContext()));
    }

    StructVal = getBasicVals().consVals(FieldValue, StructVal);
  }

  return NonLoc::MakeCompoundVal(T, StructVal, getBasicVals());
}

const GRState* RegionStoreManager::Bind(const GRState* St, Loc L, SVal V) {
  // Currently we don't bind value to symbolic location. But if the logic is
  // made clear, we might change this decision.
  if (isa<loc::SymbolVal>(L))
    return St;

  // If we get here, the location should be a region.
  const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
  assert(R);

  // Check if the region is a struct region.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    // FIXME: Verify we want getRValueType().
    if (TR->getRValueType(getContext())->isStructureType())
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
  RegionBindingsTy B = GetRegionBindings(store);

  const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
  assert(R);

  return RBFactory.Remove(B, R).getRoot();
}

const GRState* RegionStoreManager::BindDecl(const GRState* St, 
                                            const VarDecl* VD, SVal InitVal) {
  // All static variables are treated as symbolic values.
  if (VD->hasGlobalStorage())
    return St;

  // Process local variables.

  QualType T = VD->getType();
  
  VarRegion* VR = MRMgr.getVarRegion(VD);
  
  if (Loc::IsLocType(T) || T->isIntegerType())
    return Bind(St, Loc::MakeVal(VR), InitVal);

  else if (T->isArrayType())
    return BindArray(St, VR, InitVal);

  else if (T->isStructureType())
    return BindStruct(St, VR, InitVal);

  // Other types of variable are not supported yet.
  return St;
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


void RegionStoreManager::UpdateLiveSymbols(SVal X, LiveSymbolsTy& LSymbols) {
  for (SVal::symbol_iterator SI=X.symbol_begin(),SE=X.symbol_end();SI!=SE;++SI)
    LSymbols.insert(*SI);
}

Store RegionStoreManager::RemoveDeadBindings(const GRState* state, Stmt* Loc, 
                                             const LiveVariables& Live,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                           LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) {

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
  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const MemRegion* R = I.getKey();
    if (const VarRegion* VR = dyn_cast<VarRegion>(R)) {
      if (Live.isLive(Loc, VR->getDecl()))
        RegionRoots.push_back(VR); // This is a live "root".
    }
    else {
      // Get the super region for R.
      const MemRegion* SuperR = cast<SubRegion>(R)->getSuperRegion();
      // Get the current set of subregions for SuperR.
      const SubRegionsTy* SRptr = SubRegMap.lookup(SuperR);
      SubRegionsTy SR = SRptr ? *SRptr : SubRegF.GetEmptySet();
      // Add R to the subregions of SuperR.
      SubRegMap = SubRegMapF.Add(SubRegMap, SuperR, SubRegF.Add(SR, R));
      
      // Finally, check if SuperR is a VarRegion.  We need to do this
      // to also mark SuperR as a root (as it may not have a value directly
      // bound to it in the store).
      if (const VarRegion* VR = dyn_cast<VarRegion>(SuperR)) {
        if (Live.isLive(Loc, VR->getDecl()))
          RegionRoots.push_back(VR); // This is a live "root".
      }
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
      LSymbols.insert(SymR->getSymbol());

    // Get the data binding for R (if any).
    RegionBindingsTy::data_type* Xptr = B.lookup(R);
    if (Xptr) {
      SVal X = *Xptr;
      UpdateLiveSymbols(X, LSymbols); // Update the set of live symbols.
    
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
    if (const SymbolicRegion* SymR = dyn_cast<SymbolicRegion>(R)) {
      SymbolRef Sym = SymR->getSymbol();
      if (!LSymbols.count(Sym)) DSymbols.insert(Sym);
    }

    SVal X = I.getData();
    SVal::symbol_iterator SI = X.symbol_begin(), SE = X.symbol_end();
    for (; SI != SE; ++SI) { if (!LSymbols.count(*SI)) DSymbols.insert(*SI); }
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
  
  // FIXME: Verify we should use getLValueType or getRValueType.
  QualType T = R->getRValueType(getContext());
  assert(T->isArrayType());

  // When we are binding the whole array, it always has default value 0.
  GRStateRef state(St, StateMgr);
  //  St = state.set<RegionDefaultValue>(R, NonLoc::MakeVal(getBasicVals(), 0, 
  //                                                        false));

  Store store = St->getStore();

  ConstantArrayType* CAT = cast<ConstantArrayType>(T.getTypePtr());

  llvm::APSInt Size(CAT->getSize(), false);
  llvm::APSInt i = getBasicVals().getZeroWithPtrWidth(false);

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
      ElementRegion* ER = MRMgr.getElementRegion(Idx, R);

      SVal V = NonLoc::MakeVal(getBasicVals(), str[j], sizeof(char)*8, true);
      St = Bind(St, loc::MemRegionVal(ER), V);
    }

    return StateMgr.MakeStateWithStore(St, store);
  }


  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);

  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  for (; i < Size; ++i, ++VI) {
    // The init list might be shorter than the array decl.
    if (VI == VE)
      break;

    SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
    ElementRegion* ER = MRMgr.getElementRegion(Idx, R);

    if (CAT->getElementType()->isStructureType())
      St = BindStruct(St, ER, *VI);
    else
      St = Bind(St, Loc::MakeVal(ER), *VI);
  }

  return StateMgr.MakeStateWithStore(St, store);
}

const GRState*
RegionStoreManager::BindStruct(const GRState* St, const TypedRegion* R, SVal V){
  // FIXME: Verify that we should use getRValueType or getLValueType.
  QualType T = R->getRValueType(getContext());
  assert(T->isStructureType());

  RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  RecordDecl::field_iterator FI = RD->field_begin(), FE = RD->field_end();

  for (; FI != FE; ++FI, ++VI) {

    // There may be fewer values than fields only when we are initializing a
    // struct decl. In this case, mark the region as having default value.
    if (VI == VE) {
      // GRStateRef state(St, StateMgr);
    //St = state.set<RegionDefaultValue>(R, NonLoc::MakeVal(getBasicVals(), 0, 
      //                                                   false));
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

const GRState* RegionStoreManager::AddRegionView(const GRState* St,
                                                 const MemRegion* View,
                                                 const MemRegion* Base) {
  GRStateRef state(St, StateMgr);

  // First, retrieve the region view of the base region.
  const RegionViews* d = state.get<RegionViewMap>(Base);
  RegionViews L = d ? *d : RVFactory.GetEmptyList();

  // Now add View to the region view.
  L = RVFactory.Add(View, L);

  // Create a new state with the new region view.
  return state.set<RegionViewMap>(Base, L);
}
