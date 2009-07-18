//== Store.cpp - Interface for maps from Locations to Values ----*- C++ -*--==//
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

#include "clang/Analysis/PathSensitive/Store.h"
#include "clang/Analysis/PathSensitive/GRState.h"

using namespace clang;

StoreManager::StoreManager(GRStateManager &stateMgr, bool useNewCastRegion)
  : ValMgr(stateMgr.getValueManager()),
    StateMgr(stateMgr),
    UseNewCastRegion(useNewCastRegion),
    MRMgr(ValMgr.getRegionManager()) {}

StoreManager::CastResult
StoreManager::MakeElementRegion(const GRState *state, const MemRegion *region,
                                QualType pointeeTy, QualType castToTy) {
  
  // Record the cast type of the region.
  state = setCastType(state, region, castToTy);
  
  // Create a new ElementRegion at offset 0.
  SVal idx = ValMgr.makeZeroArrayIndex();
  return CastResult(state, MRMgr.getElementRegion(pointeeTy, idx, region,
                                                  ValMgr.getContext()));  
}

static bool IsCompleteType(ASTContext &Ctx, QualType Ty) {
  if (const RecordType *RT = Ty->getAsRecordType()) {
    const RecordDecl *D = RT->getDecl();
    if (!D->getDefinition(Ctx))
      return false;
  }
  
  return true;
}

StoreManager::CastResult
StoreManager::NewCastRegion(const GRState *state, const MemRegion* R,
                            QualType CastToTy) {
  
  ASTContext& Ctx = StateMgr.getContext();
  
  // We need to know the real type of CastToTy.
  QualType ToTy = Ctx.getCanonicalType(CastToTy);

  // Handle casts to Objective-C objects.
  if (CastToTy->isObjCObjectPointerType()) {
    state = setCastType(state, R, CastToTy);
    return CastResult(state, R);
  }
  
  if (CastToTy->isBlockPointerType()) {
    if (isa<CodeTextRegion>(R))
      return CastResult(state, R);
    
    // FIXME: This may not be the right approach, depending on the symbol
    // involved.  Blocks can be casted to/from 'id', as they can be treated
    // as Objective-C objects.
    if (SymbolRef sym = loc::MemRegionVal(R).getAsSymbol()) {
      R = MRMgr.getCodeTextRegion(sym, CastToTy);
      return CastResult(state, R);
    }

    // We don't know what to make of it.  Return a NULL region, which
    // will be interpretted as UnknownVal.
    return CastResult(state, NULL);
  }

  // Now assume we are casting from pointer to pointer. Other cases should
  // already be handled.
  QualType PointeeTy = CastToTy->getAsPointerType()->getPointeeType();
  
  // Process region cast according to the kind of the region being cast.
  switch (R->getKind()) {
    case MemRegion::BEG_TYPED_REGIONS:
    case MemRegion::MemSpaceRegionKind:
    case MemRegion::BEG_DECL_REGIONS:
    case MemRegion::END_DECL_REGIONS:
    case MemRegion::END_TYPED_REGIONS:
    case MemRegion::TypedViewRegionKind: {
      assert(0 && "Invalid region cast");
      break;
    }
      
    case MemRegion::CodeTextRegionKind: {
      // CodeTextRegion should be cast to only a function or block pointer type,
      // although they can in practice be casted to anything, e.g, void*,
      // char*, etc.
      // Just pass the region through.
      break;
    }
      
    case MemRegion::StringRegionKind:
      // Handle casts of string literals.
      return MakeElementRegion(state, R, PointeeTy, CastToTy);

    case MemRegion::ObjCObjectRegionKind:
    case MemRegion::SymbolicRegionKind:
      // FIXME: Need to handle arbitrary downcasts.
    case MemRegion::AllocaRegionKind: {  
      state = setCastType(state, R, CastToTy);
      break;
    }

    case MemRegion::CompoundLiteralRegionKind:
    case MemRegion::ElementRegionKind:
    case MemRegion::FieldRegionKind:
    case MemRegion::ObjCIvarRegionKind:
    case MemRegion::VarRegionKind: {
      // VarRegion, ElementRegion, and FieldRegion has an inherent type.
      // Normally they should not be cast. We only layer an ElementRegion when
      // the cast-to pointee type is of smaller size. In other cases, we return
      // the original VarRegion.
      
      // If the pointee or object type is incomplete, do not compute their
      // sizes, and return the original region.
      QualType ObjTy = cast<TypedRegion>(R)->getValueType(Ctx);
      
      if (!IsCompleteType(Ctx, PointeeTy) || !IsCompleteType(Ctx, ObjTy)) {
        state = setCastType(state, R, ToTy);
        break;
      }

      uint64_t PointeeTySize = Ctx.getTypeSize(PointeeTy);
      uint64_t ObjTySize = Ctx.getTypeSize(ObjTy);
      
      if ((PointeeTySize > 0 && PointeeTySize < ObjTySize) ||
          (ObjTy->isAggregateType() && PointeeTy->isScalarType()) ||
          ObjTySize == 0 /* R has 'void*' type. */)
        return MakeElementRegion(state, R, PointeeTy, ToTy);
        
      state = setCastType(state, R, ToTy);
      break;
    }
  }
  
  return CastResult(state, R);
}


StoreManager::CastResult
StoreManager::OldCastRegion(const GRState* state, const MemRegion* R,
                         QualType CastToTy) {
  
  ASTContext& Ctx = StateMgr.getContext();

  // We need to know the real type of CastToTy.
  QualType ToTy = Ctx.getCanonicalType(CastToTy);

  // Return the same region if the region types are compatible.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    QualType Ta = Ctx.getCanonicalType(TR->getLocationType(Ctx));

    if (Ta == ToTy)
      return CastResult(state, R);
  }
  
  if (const PointerType* PTy = dyn_cast<PointerType>(ToTy.getTypePtr())) {
    // Check if we are casting to 'void*'.
    // FIXME: Handle arbitrary upcasts.
    QualType Pointee = PTy->getPointeeType();
    if (Pointee->isVoidType()) {
      while (true) {
        if (const TypedViewRegion *TR = dyn_cast<TypedViewRegion>(R)) {
          // Casts to void* removes TypedViewRegion. This happens when:
          //
          // void foo(void*);
          // ...
          // void bar() {
          //   int x;
          //   foo(&x);
          // }
          //
          R = TR->removeViews();
          continue;
        }
        else if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
          // Casts to void* also removes ElementRegions. This happens when:
          //
          // void foo(void*);
          // ...
          // void bar() {
          //   int x;
          //   foo((char*)&x);
          // }                
          //
          R = ER->getSuperRegion();
          continue;
        }

        break;
      }
      
      return CastResult(state, R);
    }
    else if (Pointee->isIntegerType()) {
      // FIXME: At some point, it stands to reason that this 'dyn_cast' should
      //  become a 'cast' and that 'R' will always be a TypedRegion.
      if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
        // Check if we are casting to a region with an integer type.  We now
        // the types aren't the same, so we construct an ElementRegion.
        SVal Idx = ValMgr.makeZeroArrayIndex();
        
        // If the super region is an element region, strip it away.
        // FIXME: Is this the right thing to do in all cases?
        const MemRegion *Base = isa<ElementRegion>(TR) ? TR->getSuperRegion()
                                                       : TR;
        ElementRegion* ER = MRMgr.getElementRegion(Pointee, Idx, Base, 
                                                   StateMgr.getContext());
        return CastResult(state, ER);
      }
    }
  }

  // FIXME: Need to handle arbitrary downcasts.
  // FIXME: Handle the case where a TypedViewRegion (layering a SymbolicRegion
  //         or an AllocaRegion is cast to another view, thus causing the memory
  //         to be re-used for a different purpose.
  if (isa<SymbolicRegion>(R) || isa<AllocaRegion>(R)) {
    const MemRegion* ViewR = MRMgr.getTypedViewRegion(CastToTy, R);  
    return CastResult(AddRegionView(state, ViewR, R), ViewR);
  }
  
  return CastResult(state, R);
}

const GRState *StoreManager::InvalidateRegion(const GRState *state,
                                              const MemRegion *R,
                                              const Expr *E, unsigned Count) {
  ASTContext& Ctx = StateMgr.getContext();

  if (!R->isBoundable())
    return state;

  if (isa<AllocaRegion>(R) || isa<SymbolicRegion>(R) 
      || isa<ObjCObjectRegion>(R)) {
    // Invalidate the alloca region by setting its default value to 
    // conjured symbol. The type of the symbol is irrelavant.
    SVal V = ValMgr.getConjuredSymbolVal(E, Ctx.IntTy, Count);
    state = setDefaultValue(state, R, V);
    
    // FIXME: This form of invalidation is a little bogus; we actually need
    // to invalidate all subregions as well.
    return state;
  }

  const TypedRegion *TR = cast<TypedRegion>(R);
  QualType T = TR->getValueType(Ctx);

  // If the region is cast to another type, use that type.  
  if (const QualType *CastTy = getCastType(state, R)) {
    assert(!(*CastTy)->isObjCObjectPointerType());
    QualType NewT = (*CastTy)->getAsPointerType()->getPointeeType();    

    // The only exception is if the original region had a location type as its
    // value type we always want to treat the region as binding to a location.
    // This issue can arise when pointers are casted to integers and back.

    if (!(Loc::IsLocType(T) && !Loc::IsLocType(NewT)))
      T = NewT;
  }
  
  if (Loc::IsLocType(T) || (T->isIntegerType() && T->isScalarType())) {
    SVal V = ValMgr.getConjuredSymbolVal(E, T, Count);
    return Bind(state, ValMgr.makeLoc(TR), V);
  }
  else if (const RecordType *RT = T->getAsStructureType()) {
    // FIXME: handle structs with default region value.
    const RecordDecl *RD = RT->getDecl()->getDefinition(Ctx);

    // No record definition.  There is nothing we can do.
    if (!RD)
      return state;

    // Iterate through the fields and construct new symbols.
    for (RecordDecl::field_iterator FI=RD->field_begin(),
           FE=RD->field_end(); FI!=FE; ++FI) {
      
      // For now just handle scalar fields.
      FieldDecl *FD = *FI;
      QualType FT = FD->getType();
      const FieldRegion* FR = MRMgr.getFieldRegion(FD, TR);
      
      if (Loc::IsLocType(FT) || 
          (FT->isIntegerType() && FT->isScalarType())) {
        SVal V = ValMgr.getConjuredSymbolVal(E, FT, Count);
        state = state->bindLoc(ValMgr.makeLoc(FR), V);
      }
      else if (FT->isStructureType()) {
        // set the default value of the struct field to conjured
        // symbol. Note that the type of the symbol is irrelavant.
        // We cannot use the type of the struct otherwise ValMgr won't
        // give us the conjured symbol.
        SVal V = ValMgr.getConjuredSymbolVal(E, Ctx.IntTy, Count);
        state = setDefaultValue(state, FR, V);
      }
    }
  } else if (const ArrayType *AT = Ctx.getAsArrayType(T)) {
    // Set the default value of the array to conjured symbol.
    SVal V = ValMgr.getConjuredSymbolVal(E, AT->getElementType(),
                                         Count);
    state = setDefaultValue(state, TR, V);
  } else {
    // Just blast away other values.
    state = Bind(state, ValMgr.makeLoc(TR), UnknownVal());
  }
  
  return state;
}
