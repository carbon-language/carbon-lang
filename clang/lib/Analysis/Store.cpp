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
  : ValMgr(stateMgr.getValueManager()),
    StateMgr(stateMgr),
    MRMgr(ValMgr.getRegionManager()) {}

StoreManager::CastResult
StoreManager::CastRegion(const GRState* state, const MemRegion* R,
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

      do {
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
        else
          break;
      }
      while (0);
      
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
        const TypedRegion *Base = isa<ElementRegion>(TR) ?
                                  cast<TypedRegion>(TR->getSuperRegion()) : TR;
        ElementRegion* ER = MRMgr.getElementRegion(Pointee, Idx, Base);
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
