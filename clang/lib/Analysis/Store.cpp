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
                                QualType pointeeTy, QualType castToTy,
                                uint64_t index) {
  // Create a new ElementRegion.
  SVal idx = ValMgr.makeArrayIndex(index);
  return CastResult(state, MRMgr.getElementRegion(pointeeTy, idx, region,
                                                  ValMgr.getContext()));  
}

// FIXME: Merge with the implementation of the same method in MemRegion.cpp
static bool IsCompleteType(ASTContext &Ctx, QualType Ty) {
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
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
  
  // Handle casts to Objective-C objects.
  if (CastToTy->isObjCObjectPointerType())
    return CastResult(state, R->getBaseRegion());

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
  QualType PointeeTy = CastToTy->getAs<PointerType>()->getPointeeType();
  QualType CanonPointeeTy = Ctx.getCanonicalType(PointeeTy);

  // Handle casts to void*.  We just pass the region through.
  if (CanonPointeeTy.getUnqualifiedType() == Ctx.VoidTy)
    return CastResult(state, R);
  
  // Handle casts from compatible types.
  if (R->isBoundable())
    if (const TypedRegion *TR = dyn_cast<TypedRegion>(R)) {
      QualType ObjTy = Ctx.getCanonicalType(TR->getValueType(Ctx));
      if (CanonPointeeTy == ObjTy)
        return CastResult(state, R);
    }

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
    case MemRegion::ObjCObjectRegionKind:
      // FIXME: Need to handle arbitrary downcasts.
    case MemRegion::SymbolicRegionKind:
    case MemRegion::AllocaRegionKind:
    case MemRegion::CompoundLiteralRegionKind:
    case MemRegion::FieldRegionKind:
    case MemRegion::ObjCIvarRegionKind:
    case MemRegion::VarRegionKind:   
      return MakeElementRegion(state, R, PointeeTy, CastToTy);
      
    case MemRegion::ElementRegionKind: {
      // If we are casting from an ElementRegion to another type, the
      // algorithm is as follows:
      //
      // (1) Compute the "raw offset" of the ElementRegion from the
      //     base region.  This is done by calling 'getAsRawOffset()'.
      //
      // (2a) If we get a 'RegionRawOffset' after calling 
      //      'getAsRawOffset()', determine if the absolute offset
      //      can be exactly divided into chunks of the size of the 
      //      casted-pointee type.  If so, create a new ElementRegion with 
      //      the pointee-cast type as the new ElementType and the index
      //      being the offset divded by the chunk size.  If not, create
      //      a new ElementRegion at offset 0 off the raw offset region.
      //
      // (2b) If we don't a get a 'RegionRawOffset' after calling
      //      'getAsRawOffset()', it means that we are at offset 0.
      //      
      // FIXME: Handle symbolic raw offsets.
      
      const ElementRegion *elementR = cast<ElementRegion>(R);
      const RegionRawOffset &rawOff = elementR->getAsRawOffset();
      const MemRegion *baseR = rawOff.getRegion();
      
      // If we cannot compute a raw offset, throw up our hands and return
      // a NULL MemRegion*.
      if (!baseR)
        return CastResult(state, NULL);
      
      int64_t off = rawOff.getByteOffset();
      
      if (off == 0) {
        // Edge case: we are at 0 bytes off the beginning of baseR.  We
        // check to see if type we are casting to is the same as the base
        // region.  If so, just return the base region.        
        if (const TypedRegion *TR = dyn_cast<TypedRegion>(baseR)) {
          QualType ObjTy = Ctx.getCanonicalType(TR->getValueType(Ctx));
          QualType CanonPointeeTy = Ctx.getCanonicalType(PointeeTy);
          if (CanonPointeeTy == ObjTy)
            return CastResult(state, baseR);
        }
        
        // Otherwise, create a new ElementRegion at offset 0.
        return MakeElementRegion(state, baseR, PointeeTy, CastToTy, 0);
      }
      
      // We have a non-zero offset from the base region.  We want to determine
      // if the offset can be evenly divided by sizeof(PointeeTy).  If so,
      // we create an ElementRegion whose index is that value.  Otherwise, we
      // create two ElementRegions, one that reflects a raw offset and the other
      // that reflects the cast.
      
      // Compute the index for the new ElementRegion.
      int64_t newIndex = 0;
      const MemRegion *newSuperR = 0;

      // We can only compute sizeof(PointeeTy) if it is a complete type.
      if (IsCompleteType(Ctx, PointeeTy)) {
        // Compute the size in **bytes**.
        int64_t pointeeTySize = (int64_t) (Ctx.getTypeSize(PointeeTy) / 8);

        // Is the offset a multiple of the size?  If so, we can layer the
        // ElementRegion (with elementType == PointeeTy) directly on top of
        // the base region.
        if (off % pointeeTySize == 0) {
          newIndex = off / pointeeTySize;
          newSuperR = baseR;
        }
      }
      
      if (!newSuperR) {
        // Create an intermediate ElementRegion to represent the raw byte.
        // This will be the super region of the final ElementRegion.
        SVal idx = ValMgr.makeArrayIndex(off);
        newSuperR = MRMgr.getElementRegion(Ctx.CharTy, idx, baseR, Ctx);
      }
            
      return MakeElementRegion(state, newSuperR, PointeeTy, CastToTy, newIndex);
    }
  }
  
  return CastResult(state, R);
}


/// CastRetrievedVal - Used by subclasses of StoreManager to implement
///  implicit casts that arise from loads from regions that are reinterpreted
///  as another region.
SValuator::CastResult StoreManager::CastRetrievedVal(SVal V,
                                                     const GRState *state,
                                                     const TypedRegion *R,
                                                     QualType castTy) {
  if (castTy.isNull())
    return SValuator::CastResult(state, V);
  
  ASTContext &Ctx = ValMgr.getContext();  
  return ValMgr.getSValuator().EvalCast(V, state, castTy, R->getValueType(Ctx));
}

