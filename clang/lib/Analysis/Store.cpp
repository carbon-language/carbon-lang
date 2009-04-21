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
    QualType Ta = Ctx.getCanonicalType(TR->getLValueType(Ctx));

    if (Ta == ToTy)
      return CastResult(state, R);
  }
  
  // Check if we are casting to 'void*'.
  // FIXME: Handle arbitrary upcasts.
  if (const PointerType* PTy = dyn_cast<PointerType>(ToTy.getTypePtr()))
    if (PTy->getPointeeType()->isVoidType()) {

      // Casts to void* only removes TypedViewRegion. If there is no
      // TypedViewRegion, leave the region untouched. This happens when:
      //
      // void foo(void*);
      // ...
      // void bar() {
      //   int x;
      //   foo(&x);
      // }

      if (const TypedViewRegion *TR = dyn_cast<TypedViewRegion>(R))
        R = TR->removeViews();
      
      return CastResult(state, R);
    }

  // FIXME: We don't want to layer region views.  Need to handle
  // arbitrary downcasts.

  const MemRegion* ViewR = MRMgr.getTypedViewRegion(CastToTy, R);  
  return CastResult(AddRegionView(state, ViewR, R), ViewR);
}
