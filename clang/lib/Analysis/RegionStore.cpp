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

// RegionView GDM stuff.
typedef llvm::ImmutableList<const MemRegion*> RegionViewTy;
typedef llvm::ImmutableMap<const MemRegion*, RegionViewTy> RegionViewMapTy;
static int RegionViewMapTyIndex = 0;
namespace clang {
template<> struct GRStateTrait<RegionViewMapTy> 
  : public GRStatePartialTrait<RegionViewMapTy> {
  static void* GDMIndex() { return &RegionViewMapTyIndex; }
};
}

// RegionExtents GDM stuff.
// Currently RegionExtents are in bytes. We can change this representation when
// there are real requirements.
typedef llvm::ImmutableMap<const MemRegion*, SVal> RegionExtentsTy;
static int RegionExtentsTyIndex = 0;
namespace clang {
template<> struct GRStateTrait<RegionExtentsTy>
  : public GRStatePartialTrait<RegionExtentsTy> {
  static void* GDMIndex() { return &RegionExtentsTyIndex; }
};
}

// KillSet GDM stuff.
typedef llvm::ImmutableSet<const MemRegion*> RegionKills;
static int RegionKillsIndex = 0;
namespace clang {
  template<> struct GRStateTrait<RegionKills>
  : public GRStatePartialTrait<RegionKills> {
    static void* GDMIndex() { return &RegionKillsIndex; }
  };
}


namespace {

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  RegionViewTy::Factory RVFactory;

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
  
  Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* CL, SVal V);

  SVal getLValueString(const GRState* St, const StringLiteral* S);

  SVal getLValueCompoundLiteral(const GRState* St, const CompoundLiteralExpr*);

  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);

  SVal getLValueField(const GRState* St, SVal Base, const FieldDecl* D);

  SVal getLValueElement(const GRState* St, SVal Base, SVal Offset);

  SVal getSizeInElements(const GRState* St, const MemRegion* R);

  SVal ArrayToPointer(SVal Array);

  std::pair<const GRState*, SVal>
  CastRegion(const GRState* St, SVal VoidPtr, QualType CastToTy, Stmt* CastE);

  SVal Retrieve(const GRState* state, Loc L, QualType T = QualType());

  Store Bind(Store St, Loc LV, SVal V);

  Store Remove(Store store, Loc LV) {
    // FIXME: Implement.
    return store;
  }

  Store getInitialStore();
  
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

  Store BindDecl(Store store, const VarDecl* VD, SVal* InitVal, unsigned Count);

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

  Store InitializeArray(Store store, const TypedRegion* R, SVal Init);
  Store BindArrayToVal(Store store, const TypedRegion* BaseR, SVal V);
  Store BindArrayToSymVal(Store store, const TypedRegion* BaseR);

  Store InitializeStruct(Store store, const TypedRegion* R, SVal Init);
  Store BindStructToVal(Store store, const TypedRegion* BaseR, SVal V);
  Store BindStructToSymVal(Store store, const TypedRegion* BaseR);

  /// Retrieve the values in a struct and return a CompoundVal, used when doing
  /// struct copy: 
  /// struct s x, y; 
  /// x = y;
  /// y's value is retrieved by this method.
  SVal RetrieveStruct(Store store, const TypedRegion* R);

  Store BindStruct(Store store, const TypedRegion* R, SVal V);

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
    QualType T = VR->getType(getContext());

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
    RegionExtentsTy::data_type* T 
      = state.get<RegionExtentsTy>(ATR->getSuperRegion());

    assert(T && "region extent not exist");

    // Assume it's ConcreteInt for now.
    llvm::APSInt SSize = cast<nonloc::ConcreteInt>(*T).getValue();

    // Get the size of the element in bits.
    QualType ElemTy = cast<PointerType>(ATR->getType(getContext()).getTypePtr())
                      ->getPointeeType();

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

std::pair<const GRState*, SVal>
RegionStoreManager::CastRegion(const GRState* St, SVal VoidPtr, 
                               QualType CastToTy, Stmt* CastE) {
  if (const AllocaRegion* AR =
      dyn_cast<AllocaRegion>(cast<loc::MemRegionVal>(VoidPtr).getRegion())) {

    // Create a new region to attach type information to it.
    const AnonTypedRegion* TR = MRMgr.getAnonTypedRegion(CastToTy, AR);

    // Get the pointer to the first element.
    nonloc::ConcreteInt Idx(getBasicVals().getZeroWithPtrWidth(false));
    const ElementRegion* ER = MRMgr.getElementRegion(Idx, TR);

    // Add a RegionView to base region.
    return std::make_pair(AddRegionView(St, TR, AR), loc::MemRegionVal(ER));
  }

  // Default case.
  return std::make_pair(St, UnknownVal());
}

SVal RegionStoreManager::Retrieve(const GRState* state, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");
  Store S = state->getStore();

  switch (L.getSubKind()) {
  case loc::MemRegionKind: {
    const MemRegion* R = cast<loc::MemRegionVal>(L).getRegion();
    assert(R && "bad region");

    if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
      if (TR->getType(getContext())->isStructureType())
        return RetrieveStruct(S, TR);

    RegionBindingsTy B(static_cast<const RegionBindingsTy::TreeTy*>(S));
    RegionBindingsTy::data_type* V = B.lookup(R);
    return V ? *V : UnknownVal();
  }

  case loc::SymbolValKind:
    return UnknownVal();

  case loc::ConcreteIntKind:
    return UndefinedVal(); // As in BasicStoreManager.

  case loc::FuncValKind:
    return L;

  default:
    assert(false && "Invalid Location");
    return L;
  }
}

SVal RegionStoreManager::RetrieveStruct(Store store, const TypedRegion* R) {
  QualType T = R->getType(getContext());
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
    RegionBindingsTy B(static_cast<const RegionBindingsTy::TreeTy*>(store));
    RegionBindingsTy::data_type* data = B.lookup(FR);

    SVal FieldValue = data ? *data : UnknownVal();

    StructVal = getBasicVals().consVals(FieldValue, StructVal);
  }

  return NonLoc::MakeCompoundVal(T, StructVal, getBasicVals());
}

Store RegionStoreManager::Bind(Store store, Loc LV, SVal V) {
  if (LV.getSubKind() == loc::SymbolValKind)
    return store;

  assert(LV.getSubKind() == loc::MemRegionKind);

  const MemRegion* R = cast<loc::MemRegionVal>(LV).getRegion();
  
  assert(R);

  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R))
    if (TR->getType(getContext())->isStructureType())
      return BindStruct(store, TR, V);

  RegionBindingsTy B = GetRegionBindings(store);
  return V.isUnknown()
         ? RBFactory.Remove(B, R).getRoot()
         : RBFactory.Add(B, R, V).getRoot();
}

Store RegionStoreManager::BindStruct(Store store, const TypedRegion* R, SVal V){
  QualType T = R->getType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();

  if (!RD->isDefinition()) {
    // This can only occur when a pointer of incomplete struct type is used as a
    // function argument.
    assert(V.isUnknown());
    return store;
  }

  RegionBindingsTy B = GetRegionBindings(store);

  if (isa<UnknownVal>(V))
    return BindStructToVal(store, R, UnknownVal());

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(V);

  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  RecordDecl::field_iterator FI = RD->field_begin(), FE = RD->field_end();

  for (; FI != FE; ++FI, ++VI) {
    assert(VI != VE);

    FieldRegion* FR = MRMgr.getFieldRegion(*FI, R);

    B = RBFactory.Add(B, FR, *VI);
  }

  return B.getRoot();
}

Store RegionStoreManager::getInitialStore() {
  typedef LiveVariables::AnalysisDataTy LVDataTy;
  LVDataTy& D = StateMgr.getLiveVariables().getAnalysisData();

  Store St = RBFactory.GetEmptyMap().getRoot();

  for (LVDataTy::decl_iterator I=D.begin_decl(), E=D.end_decl(); I != E; ++I) {
    NamedDecl* ND = const_cast<NamedDecl*>(I->first);

    if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
      // Punt on static variables for now.
      if (VD->getStorageClass() == VarDecl::Static)
        continue;

      VarRegion* VR = MRMgr.getVarRegion(VD);

      QualType T = VD->getType();
      // Only handle pointers and integers for now.
      if (Loc::IsLocType(T) || T->isIntegerType()) {
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        SVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? SVal::GetSymbolValue(getSymbolManager(), VD)
                 : UndefinedVal();

        St = Bind(St, getVarLoc(VD), X);
      } 
      else if (T->isArrayType()) {
        if (VD->hasGlobalStorage()) // Params cannot have array type.
          St = BindArrayToSymVal(St, VR);
        else
          St = BindArrayToVal(St, VR, UndefinedVal());
      }
      else if (T->isStructureType()) {
        if (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
            isa<ImplicitParamDecl>(VD))
          St = BindStructToSymVal(St, VR);
        else
          St = BindStructToVal(St, VR, UndefinedVal());
      }
    }
  }
  return St;
}

Store RegionStoreManager::BindDecl(Store store, const VarDecl* VD,
                                   SVal* InitVal, unsigned Count) {
  
  if (VD->hasGlobalStorage()) {
    // Static global variables should not be visited here.
    assert(!(VD->getStorageClass() == VarDecl::Static &&
             VD->isFileVarDecl()));
    // Process static variables.
    if (VD->getStorageClass() == VarDecl::Static) {
      if (!InitVal) {
        // Only handle pointer and integer static variables.

        QualType T = VD->getType();

        if (Loc::IsLocType(T))
          store = Bind(store, getVarLoc(VD),
                       loc::ConcreteInt(getBasicVals().getValue(0, T)));

        else if (T->isIntegerType())
          store = Bind(store, getVarLoc(VD),
                       loc::ConcreteInt(getBasicVals().getValue(0, T)));

        // Other types of static local variables are not handled yet.
      } else {
        store = Bind(store, getVarLoc(VD), *InitVal);
      }
    }
  } else {
    // Process local variables.

    QualType T = VD->getType();

    VarRegion* VR = MRMgr.getVarRegion(VD);

    if (Loc::IsLocType(T) || T->isIntegerType()) {
      SVal V = InitVal ? *InitVal : UndefinedVal();
      store = Bind(store, loc::MemRegionVal(VR), V);
    }
    else if (T->isArrayType()) {
      if (!InitVal)
        store = BindArrayToVal(store, VR, UndefinedVal());
      else
        store = InitializeArray(store, VR, *InitVal);
    }
    else if (T->isStructureType()) {
      if (!InitVal)
        store = BindStructToVal(store, VR, UndefinedVal());
      else
        store = InitializeStruct(store, VR, *InitVal);
    }

    // Other types of local variables are not handled yet.
  }
  return store;
}

Store RegionStoreManager::BindCompoundLiteral(Store store, 
                                              const CompoundLiteralExpr* CL, 
                                              SVal V) {
  CompoundLiteralRegion* R = MRMgr.getCompoundLiteralRegion(CL);
  store = Bind(store, loc::MemRegionVal(R), V);
  return store;
}

const GRState* RegionStoreManager::setExtent(const GRState* St,
                                             const MemRegion* R, SVal Extent) {
  GRStateRef state(St, StateMgr);
  return state.set<RegionExtentsTy>(R, Extent);
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
    store = Remove(store, loc::MemRegionVal(R));

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

Store RegionStoreManager::InitializeArray(Store store, const TypedRegion* R, 
                                          SVal Init) {
  QualType T = R->getType(getContext());
  assert(T->isArrayType());

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

    for (; i < Size; ++i, ++j) {
      SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
      ElementRegion* ER = MRMgr.getElementRegion(Idx, R);

      // Copy bytes from the string literal into the target array. Trailing
      // bytes in the array that are not covered by the string literal are
      // initialized to zero.
      SVal V = (j < len) 
        ? NonLoc::MakeVal(getBasicVals(), str[j], sizeof(char)*8, true)
        : NonLoc::MakeVal(getBasicVals(), 0, sizeof(char)*8, true);

      store = Bind(store, loc::MemRegionVal(ER), V);
    }

    return store;
  }


  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);

  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  for (; i < Size; ++i) {
    SVal Idx = NonLoc::MakeVal(getBasicVals(), i);
    ElementRegion* ER = MRMgr.getElementRegion(Idx, R);
    
    store = Bind(store, loc::MemRegionVal(ER), (VI!=VE) ? *VI : UndefinedVal());
    // The init list might be shorter than the array decl.
    if (VI != VE) ++VI;
  }

  return store;
}

// Bind all elements of the array to some value.
Store RegionStoreManager::BindArrayToVal(Store store, const TypedRegion* BaseR,
                                         SVal V){
  QualType T = BaseR->getType(getContext());
  assert(T->isArrayType());

  // Only handle constant size array for now.
  if (ConstantArrayType* CAT=dyn_cast<ConstantArrayType>(T.getTypePtr())) {

    llvm::APInt Size = CAT->getSize();
    llvm::APInt i = llvm::APInt::getNullValue(Size.getBitWidth());

    for (; i != Size; ++i) {
      nonloc::ConcreteInt Idx(getBasicVals().getValue(llvm::APSInt(i, false)));

      ElementRegion* ER = MRMgr.getElementRegion(Idx, BaseR);

      if (CAT->getElementType()->isStructureType())
        store = BindStructToVal(store, ER, V);
      else
        store = Bind(store, loc::MemRegionVal(ER), V);
    }
  }

  return store;
}

Store RegionStoreManager::BindArrayToSymVal(Store store, 
                                            const TypedRegion* BaseR) {
  QualType T = BaseR->getType(getContext());
  assert(T->isArrayType());

  if (ConstantArrayType* CAT = dyn_cast<ConstantArrayType>(T.getTypePtr())) {
    llvm::APInt Size = CAT->getSize();
    llvm::APInt i = llvm::APInt::getNullValue(Size.getBitWidth());
    for (; i != Size; ++i) {
      nonloc::ConcreteInt Idx(getBasicVals().getValue(llvm::APSInt(i, false)));
      
      ElementRegion* ER = MRMgr.getElementRegion(Idx, BaseR);

      if (CAT->getElementType()->isStructureType()) {
        store = BindStructToSymVal(store, ER);
      }
      else {
        SVal V = SVal::getSymbolValue(getSymbolManager(), BaseR, 
                                      &Idx.getValue(), CAT->getElementType());
        store = Bind(store, loc::MemRegionVal(ER), V);
      }
    }
  }

  return store;
}

Store RegionStoreManager::InitializeStruct(Store store, const TypedRegion* R, 
                                           SVal Init) {
  QualType T = R->getType(getContext());
  assert(T->isStructureType());

  RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);
  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();
  RecordDecl::field_iterator FI = RD->field_begin(), FE = RD->field_end();

  for (; FI != FE; ++FI) {
    QualType FTy = (*FI)->getType();
    FieldRegion* FR = MRMgr.getFieldRegion(*FI, R);

    if (Loc::IsLocType(FTy) || FTy->isIntegerType()) {
      if (VI != VE) {
        store = Bind(store, loc::MemRegionVal(FR), *VI);
        ++VI;
      } else
        store = Bind(store, loc::MemRegionVal(FR), UndefinedVal());
    } 
    else if (FTy->isArrayType()) {
      if (VI != VE) {
        store = InitializeArray(store, FR, *VI);
        ++VI;
      } else
        store = BindArrayToVal(store, FR, UndefinedVal());
    }
    else if (FTy->isStructureType()) {
      if (VI != VE) {
        store = InitializeStruct(store, FR, *VI);
        ++VI;
      } else
        store = BindStructToVal(store, FR, UndefinedVal());
    }
  }
  return store;
}

// Bind all fields of the struct to some value.
Store RegionStoreManager::BindStructToVal(Store store, const TypedRegion* BaseR,
                                          SVal V) {
  QualType T = BaseR->getType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();

  for (; I != E; ++I) {
    
    QualType FTy = (*I)->getType();
    FieldRegion* FR = MRMgr.getFieldRegion(*I, BaseR);
    
    if (Loc::IsLocType(FTy) || FTy->isIntegerType()) {
      store = Bind(store, loc::MemRegionVal(FR), V);

    } else if (FTy->isArrayType()) {
      store = BindArrayToVal(store, FR, V);

    } else if (FTy->isStructureType()) {
      store = BindStructToVal(store, FR, V);
    }
  }

  return store;
}

Store RegionStoreManager::BindStructToSymVal(Store store, 
                                             const TypedRegion* BaseR) {
  QualType T = BaseR->getType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();

  for (; I != E; ++I) {
    QualType FTy = (*I)->getType();
    FieldRegion* FR = MRMgr.getFieldRegion(*I, BaseR);

    if (Loc::IsLocType(FTy) || FTy->isIntegerType()) {
      store = Bind(store, loc::MemRegionVal(FR), 
                   SVal::getSymbolValue(getSymbolManager(), BaseR, *I, FTy));
    } 
    else if (FTy->isArrayType()) {
      store = BindArrayToSymVal(store, FR);
    } 
    else if (FTy->isStructureType()) {
      store = BindStructToSymVal(store, FR);
    }
  }

  return store;
}

const GRState* RegionStoreManager::AddRegionView(const GRState* St,
                                                 const MemRegion* View,
                                                 const MemRegion* Base) {
  GRStateRef state(St, StateMgr);

  // First, retrieve the region view of the base region.
  RegionViewMapTy::data_type* d = state.get<RegionViewMapTy>(Base);
  RegionViewTy L = d ? *d : RVFactory.GetEmptyList();

  // Now add View to the region view.
  L = RVFactory.Add(View, L);

  // Create a new state with the new region view.
  return state.set<RegionViewMapTy>(Base, L);
}
