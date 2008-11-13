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
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

typedef llvm::ImmutableMap<const MemRegion*, SVal> RegionBindingsTy;

namespace {

class VISIBILITY_HIDDEN RegionStoreManager : public StoreManager {
  RegionBindingsTy::Factory RBFactory;
  GRStateManager& StateMgr;
  MemRegionManager MRMgr;

public:
  RegionStoreManager(GRStateManager& mgr) 
    : StateMgr(mgr), MRMgr(StateMgr.getAllocator()) {}

  virtual ~RegionStoreManager() {}

  MemRegionManager& getRegionManager() { return MRMgr; }

  // FIXME: Is this function necessary?
  SVal GetRegionSVal(Store St, const MemRegion* R) {
    return Retrieve(St, loc::MemRegionVal(R));
  }
  
  Store BindCompoundLiteral(Store store, const CompoundLiteralExpr* CL, SVal V);

  SVal getLValueString(const GRState* St, const StringLiteral* S);

  SVal getLValueCompoundLiteral(const GRState* St, const CompoundLiteralExpr*);

  SVal getLValueVar(const GRState* St, const VarDecl* VD);
  
  SVal getLValueIvar(const GRState* St, const ObjCIvarDecl* D, SVal Base);

  SVal getLValueField(const GRState* St, SVal Base, const FieldDecl* D);

  SVal getLValueElement(const GRState* St, SVal Base, SVal Offset);

  SVal ArrayToPointer(SVal Array);

  SVal Retrieve(Store S, Loc L, QualType T = QualType());

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

  Store RemoveDeadBindings(Store store, Stmt* Loc, const LiveVariables& Live,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                           LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols);

  Store BindDecl(Store store, const VarDecl* VD, SVal* InitVal, unsigned Count);

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
  Store InitializeStruct(Store store, const TypedRegion* R, SVal Init);
  Store BindStructToVal(Store store, const TypedRegion* BaseR, SVal V);

  SVal RetrieveStruct(Store store, const TypedRegion* R);
  Store BindStruct(Store store, const TypedRegion* R, SVal V);
  // Utility methods.
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
  ASTContext& getContext() { return StateMgr.getContext(); }
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

  // We expect BaseR is an ElementRegion, not a base VarRegion.

  const ElementRegion* ElemR = cast<ElementRegion>(BaseL.getRegion());

  SVal Idx = ElemR->getIndex();

  nonloc::ConcreteInt *CI1, *CI2;

  // Only handle integer indices for now.
  if ((CI1 = dyn_cast<nonloc::ConcreteInt>(&Idx)) &&
      (CI2 = dyn_cast<nonloc::ConcreteInt>(&Offset))) {
    SVal NewIdx = CI1->EvalBinOp(StateMgr.getBasicVals(), BinaryOperator::Add,
                                 *CI2);
    return loc::MemRegionVal(MRMgr.getElementRegion(NewIdx, 
                                                    ElemR->getSuperRegion()));
  }

  return UnknownVal();
}

// Cast 'pointer to array' to 'pointer to the first element of array'.

SVal RegionStoreManager::ArrayToPointer(SVal Array) {
  const MemRegion* ArrayR = cast<loc::MemRegionVal>(&Array)->getRegion();
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

  // FIXME: Find a better way to get bit width.
  nonloc::ConcreteInt Idx(BasicVals.getValue(0, 32, false));
  ElementRegion* ER = MRMgr.getElementRegion(Idx, ArrayR);
  
  return loc::MemRegionVal(ER);                    
}

SVal RegionStoreManager::Retrieve(Store S, Loc L, QualType T) {
  assert(!isa<UnknownVal>(L) && "location unknown");
  assert(!isa<UndefinedVal>(L) && "location undefined");

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
    break;
  }
}

SVal RegionStoreManager::RetrieveStruct(Store store, const TypedRegion* R) {
  QualType T = R->getType(getContext());
  assert(T->isStructureType());

  const RecordType* RT = cast<RecordType>(T.getTypePtr());
  RecordDecl* RD = RT->getDecl();
  assert(RD->isDefinition());

  llvm::ImmutableList<SVal> StructVal = getBasicVals().getEmptySValList();

  for (int i = RD->getNumMembers() - 1; i >= 0; --i) {
    FieldRegion* FR = MRMgr.getFieldRegion(RD->getMember(i), R);
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
    // This can only occur when a pointer of imcomplete struct type is used as a
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

      QualType T = VD->getType();
      // Only handle pointers and integers for now.
      if (Loc::IsLocType(T) || T->isIntegerType()) {
        // Initialize globals and parameters to symbolic values.
        // Initialize local variables to undefined.
        SVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                  isa<ImplicitParamDecl>(VD))
                 ? SVal::GetSymbolValue(StateMgr.getSymbolManager(), VD)
                 : UndefinedVal();

        St = Bind(St, getVarLoc(VD), X);
      }
    }
  }
  return St;
}

Store RegionStoreManager::BindDecl(Store store, const VarDecl* VD,
                                   SVal* InitVal, unsigned Count) {
  
  BasicValueFactory& BasicVals = StateMgr.getBasicVals();

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
                       loc::ConcreteInt(BasicVals.getValue(0, T)));

        else if (T->isIntegerType())
          store = Bind(store, getVarLoc(VD),
                       loc::ConcreteInt(BasicVals.getValue(0, T)));

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

Store RegionStoreManager::RemoveDeadBindings(Store store, Stmt* Loc, 
                                             const LiveVariables& Live,
                           llvm::SmallVectorImpl<const MemRegion*>& RegionRoots,
                           LiveSymbolsTy& LSymbols, DeadSymbolsTy& DSymbols) {

  RegionBindingsTy B = GetRegionBindings(store);
  typedef SVal::symbol_iterator symbol_iterator;

  // FIXME: Mark all region binding value's symbol as live. We also omit symbols
  // in SymbolicRegions.
  for (RegionBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    SVal X = I.getData();
    for (symbol_iterator SI=X.symbol_begin(), SE=X.symbol_end(); SI!=SE; ++SI)
      LSymbols.insert(*SI);
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

  llvm::APInt Size = CAT->getSize();

  llvm::APInt i = llvm::APInt::getNullValue(Size.getBitWidth());

  nonloc::CompoundVal& CV = cast<nonloc::CompoundVal>(Init);

  nonloc::CompoundVal::iterator VI = CV.begin(), VE = CV.end();

  for (; i != Size; ++i) {
    nonloc::ConcreteInt Idx(getBasicVals().getValue(llvm::APSInt(i)));

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
      nonloc::ConcreteInt Idx(getBasicVals().getValue(llvm::APSInt(i)));

      ElementRegion* ER = MRMgr.getElementRegion(Idx, BaseR);

      store = Bind(store, loc::MemRegionVal(ER), V);
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
