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
  
  // Return the same region if the region types are compatible.
  if (const TypedRegion* TR = dyn_cast<TypedRegion>(R)) {
    ASTContext& Ctx = StateMgr.getContext();
    QualType Ta = Ctx.getCanonicalType(TR->getLValueType(Ctx));
    QualType Tb = Ctx.getCanonicalType(CastToTy);
    
    if (Ta == Tb)
      return CastResult(state, R);
  }
  
  // FIXME: We should handle the case when we are casting *back* to a
  // previous type. For example:
  //
  //      void* x = ...;
  //      char* y = (char*) x;
  //      void* z = (void*) y; // <-- we should get the same region that is 
  //                                  bound to 'x'
  const MemRegion* ViewR = MRMgr.getTypedViewRegion(CastToTy, R);  
  return CastResult(AddRegionView(state, ViewR, R), ViewR);
}
