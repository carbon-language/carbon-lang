//===- ObjCARCAliasAnalysis.cpp - ObjC ARC Optimization -*- mode: c++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a simple ARC-aware AliasAnalysis using special knowledge
/// of Objective C to enhance other optimization passes which rely on the Alias
/// Analysis infrastructure.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "objc-arc-aa"
#include "ObjCARC.h"
#include "ObjCARCAliasAnalysis.h"
#include "llvm/IR/Instruction.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/PassSupport.h"

namespace llvm {
  class Function;
  class Value;
}

using namespace llvm;
using namespace llvm::objcarc;

// Register this pass...
char ObjCARCAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(ObjCARCAliasAnalysis, AliasAnalysis, "objc-arc-aa",
                   "ObjC-ARC-Based Alias Analysis", false, true, false)

ImmutablePass *llvm::createObjCARCAliasAnalysisPass() {
  return new ObjCARCAliasAnalysis();
}

void
ObjCARCAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

AliasAnalysis::AliasResult
ObjCARCAliasAnalysis::alias(const Location &LocA, const Location &LocB) {
  if (!EnableARCOpts)
    return AliasAnalysis::alias(LocA, LocB);

  // First, strip off no-ops, including ObjC-specific no-ops, and try making a
  // precise alias query.
  const Value *SA = StripPointerCastsAndObjCCalls(LocA.Ptr);
  const Value *SB = StripPointerCastsAndObjCCalls(LocB.Ptr);
  AliasResult Result =
    AliasAnalysis::alias(Location(SA, LocA.Size, LocA.TBAATag),
                         Location(SB, LocB.Size, LocB.TBAATag));
  if (Result != MayAlias)
    return Result;

  // If that failed, climb to the underlying object, including climbing through
  // ObjC-specific no-ops, and try making an imprecise alias query.
  const Value *UA = GetUnderlyingObjCPtr(SA);
  const Value *UB = GetUnderlyingObjCPtr(SB);
  if (UA != SA || UB != SB) {
    Result = AliasAnalysis::alias(Location(UA), Location(UB));
    // We can't use MustAlias or PartialAlias results here because
    // GetUnderlyingObjCPtr may return an offsetted pointer value.
    if (Result == NoAlias)
      return NoAlias;
  }

  // If that failed, fail. We don't need to chain here, since that's covered
  // by the earlier precise query.
  return MayAlias;
}

bool
ObjCARCAliasAnalysis::pointsToConstantMemory(const Location &Loc,
                                             bool OrLocal) {
  if (!EnableARCOpts)
    return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  // First, strip off no-ops, including ObjC-specific no-ops, and try making
  // a precise alias query.
  const Value *S = StripPointerCastsAndObjCCalls(Loc.Ptr);
  if (AliasAnalysis::pointsToConstantMemory(Location(S, Loc.Size, Loc.TBAATag),
                                            OrLocal))
    return true;

  // If that failed, climb to the underlying object, including climbing through
  // ObjC-specific no-ops, and try making an imprecise alias query.
  const Value *U = GetUnderlyingObjCPtr(S);
  if (U != S)
    return AliasAnalysis::pointsToConstantMemory(Location(U), OrLocal);

  // If that failed, fail. We don't need to chain here, since that's covered
  // by the earlier precise query.
  return false;
}

AliasAnalysis::ModRefBehavior
ObjCARCAliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  // We have nothing to do. Just chain to the next AliasAnalysis.
  return AliasAnalysis::getModRefBehavior(CS);
}

AliasAnalysis::ModRefBehavior
ObjCARCAliasAnalysis::getModRefBehavior(const Function *F) {
  if (!EnableARCOpts)
    return AliasAnalysis::getModRefBehavior(F);

  switch (GetFunctionClass(F)) {
  case IC_NoopCast:
    return DoesNotAccessMemory;
  default:
    break;
  }

  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
ObjCARCAliasAnalysis::getModRefInfo(ImmutableCallSite CS, const Location &Loc) {
  if (!EnableARCOpts)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  switch (GetBasicInstructionClass(CS.getInstruction())) {
  case IC_Retain:
  case IC_RetainRV:
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_NoopCast:
  case IC_AutoreleasepoolPush:
  case IC_FusedRetainAutorelease:
  case IC_FusedRetainAutoreleaseRV:
    // These functions don't access any memory visible to the compiler.
    // Note that this doesn't include objc_retainBlock, because it updates
    // pointers when it copies block data.
    return NoModRef;
  default:
    break;
  }

  // Handle special objective c calls defaulting to chaining.
  const Function *F = CS.getCalledFunction();
  if (F)
    return StringSwitch<AliasAnalysis::ModRefResult>(F->getName())
      .Case("objc_sync_start", NoModRef)
      .Case("objc_sync_stop", NoModRef)
      .Default(AliasAnalysis::getModRefInfo(CS, Loc));

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
ObjCARCAliasAnalysis::getModRefInfo(ImmutableCallSite CS1,
                                    ImmutableCallSite CS2) {
  // TODO: Theoretically we could check for dependencies between objc_* calls
  // and OnlyAccessesArgumentPointees calls or other well-behaved calls.
  return AliasAnalysis::getModRefInfo(CS1, CS2);
}
