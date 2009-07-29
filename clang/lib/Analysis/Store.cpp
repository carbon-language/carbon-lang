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

StoreManager::StoreManager(GRStateManager &stateMgr)
  : ValMgr(stateMgr.getValueManager()), StateMgr(stateMgr),
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
StoreManager::CastRegion(const GRState *state, const MemRegion* R,
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
    case MemRegion::END_TYPED_REGIONS: {
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
