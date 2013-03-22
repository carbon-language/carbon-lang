//===- DependencyAnalysis.cpp - ObjC ARC Optimization ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines special dependency analysis routines used in Objective C
/// ARC Optimizations.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "objc-arc-dependency"
#include "ObjCARC.h"
#include "DependencyAnalysis.h"
#include "ProvenanceAnalysis.h"
#include "llvm/Support/CFG.h"

using namespace llvm;
using namespace llvm::objcarc;

/// Test whether the given instruction can result in a reference count
/// modification (positive or negative) for the pointer's object.
bool
llvm::objcarc::CanAlterRefCount(const Instruction *Inst, const Value *Ptr,
                                ProvenanceAnalysis &PA,
                                InstructionClass Class) {
  switch (Class) {
  case IC_Autorelease:
  case IC_AutoreleaseRV:
  case IC_IntrinsicUser:
  case IC_User:
    // These operations never directly modify a reference count.
    return false;
  default: break;
  }

  ImmutableCallSite CS = static_cast<const Value *>(Inst);
  assert(CS && "Only calls can alter reference counts!");

  // See if AliasAnalysis can help us with the call.
  AliasAnalysis::ModRefBehavior MRB = PA.getAA()->getModRefBehavior(CS);
  if (AliasAnalysis::onlyReadsMemory(MRB))
    return false;
  if (AliasAnalysis::onlyAccessesArgPointees(MRB)) {
    for (ImmutableCallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
         I != E; ++I) {
      const Value *Op = *I;
      if (IsPotentialRetainableObjPtr(Op, *PA.getAA()) && PA.related(Ptr, Op))
        return true;
    }
    return false;
  }

  // Assume the worst.
  return true;
}

/// Test whether the given instruction can "use" the given pointer's object in a
/// way that requires the reference count to be positive.
bool
llvm::objcarc::CanUse(const Instruction *Inst, const Value *Ptr,
                      ProvenanceAnalysis &PA, InstructionClass Class) {
  // IC_Call operations (as opposed to IC_CallOrUser) never "use" objc pointers.
  if (Class == IC_Call)
    return false;

  // Consider various instructions which may have pointer arguments which are
  // not "uses".
  if (const ICmpInst *ICI = dyn_cast<ICmpInst>(Inst)) {
    // Comparing a pointer with null, or any other constant, isn't really a use,
    // because we don't care what the pointer points to, or about the values
    // of any other dynamic reference-counted pointers.
    if (!IsPotentialRetainableObjPtr(ICI->getOperand(1), *PA.getAA()))
      return false;
  } else if (ImmutableCallSite CS = static_cast<const Value *>(Inst)) {
    // For calls, just check the arguments (and not the callee operand).
    for (ImmutableCallSite::arg_iterator OI = CS.arg_begin(),
         OE = CS.arg_end(); OI != OE; ++OI) {
      const Value *Op = *OI;
      if (IsPotentialRetainableObjPtr(Op, *PA.getAA()) && PA.related(Ptr, Op))
        return true;
    }
    return false;
  } else if (const StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
    // Special-case stores, because we don't care about the stored value, just
    // the store address.
    const Value *Op = GetUnderlyingObjCPtr(SI->getPointerOperand());
    // If we can't tell what the underlying object was, assume there is a
    // dependence.
    return IsPotentialRetainableObjPtr(Op, *PA.getAA()) && PA.related(Op, Ptr);
  }

  // Check each operand for a match.
  for (User::const_op_iterator OI = Inst->op_begin(), OE = Inst->op_end();
       OI != OE; ++OI) {
    const Value *Op = *OI;
    if (IsPotentialRetainableObjPtr(Op, *PA.getAA()) && PA.related(Ptr, Op))
      return true;
  }
  return false;
}

/// Test if there can be dependencies on Inst through Arg. This function only
/// tests dependencies relevant for removing pairs of calls.
bool
llvm::objcarc::Depends(DependenceKind Flavor, Instruction *Inst,
                       const Value *Arg, ProvenanceAnalysis &PA) {
  // If we've reached the definition of Arg, stop.
  if (Inst == Arg)
    return true;

  switch (Flavor) {
  case NeedsPositiveRetainCount: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
    case IC_None:
      return false;
    default:
      return CanUse(Inst, Arg, PA, Class);
    }
  }

  case AutoreleasePoolBoundary: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
      // These mark the end and begin of an autorelease pool scope.
      return true;
    default:
      // Nothing else does this.
      return false;
    }
  }

  case CanChangeRetainCount: {
    InstructionClass Class = GetInstructionClass(Inst);
    switch (Class) {
    case IC_AutoreleasepoolPop:
      // Conservatively assume this can decrement any count.
      return true;
    case IC_AutoreleasepoolPush:
    case IC_None:
      return false;
    default:
      return CanAlterRefCount(Inst, Arg, PA, Class);
    }
  }

  case RetainAutoreleaseDep:
    switch (GetBasicInstructionClass(Inst)) {
    case IC_AutoreleasepoolPop:
    case IC_AutoreleasepoolPush:
      // Don't merge an objc_autorelease with an objc_retain inside a different
      // autoreleasepool scope.
      return true;
    case IC_Retain:
    case IC_RetainRV:
      // Check for a retain of the same pointer for merging.
      return GetObjCArg(Inst) == Arg;
    default:
      // Nothing else matters for objc_retainAutorelease formation.
      return false;
    }

  case RetainAutoreleaseRVDep: {
    InstructionClass Class = GetBasicInstructionClass(Inst);
    switch (Class) {
    case IC_Retain:
    case IC_RetainRV:
      // Check for a retain of the same pointer for merging.
      return GetObjCArg(Inst) == Arg;
    default:
      // Anything that can autorelease interrupts
      // retainAutoreleaseReturnValue formation.
      return CanInterruptRV(Class);
    }
  }

  case RetainRVDep:
    return CanInterruptRV(GetBasicInstructionClass(Inst));
  }

  llvm_unreachable("Invalid dependence flavor");
}

/// Walk up the CFG from StartPos (which is in StartBB) and find local and
/// non-local dependencies on Arg.
///
/// TODO: Cache results?
void
llvm::objcarc::FindDependencies(DependenceKind Flavor,
                                const Value *Arg,
                                BasicBlock *StartBB, Instruction *StartInst,
                                SmallPtrSet<Instruction *, 4> &DependingInsts,
                                SmallPtrSet<const BasicBlock *, 4> &Visited,
                                ProvenanceAnalysis &PA) {
  BasicBlock::iterator StartPos = StartInst;

  SmallVector<std::pair<BasicBlock *, BasicBlock::iterator>, 4> Worklist;
  Worklist.push_back(std::make_pair(StartBB, StartPos));
  do {
    std::pair<BasicBlock *, BasicBlock::iterator> Pair =
      Worklist.pop_back_val();
    BasicBlock *LocalStartBB = Pair.first;
    BasicBlock::iterator LocalStartPos = Pair.second;
    BasicBlock::iterator StartBBBegin = LocalStartBB->begin();
    for (;;) {
      if (LocalStartPos == StartBBBegin) {
        pred_iterator PI(LocalStartBB), PE(LocalStartBB, false);
        if (PI == PE)
          // If we've reached the function entry, produce a null dependence.
          DependingInsts.insert(0);
        else
          // Add the predecessors to the worklist.
          do {
            BasicBlock *PredBB = *PI;
            if (Visited.insert(PredBB))
              Worklist.push_back(std::make_pair(PredBB, PredBB->end()));
          } while (++PI != PE);
        break;
      }

      Instruction *Inst = --LocalStartPos;
      if (Depends(Flavor, Inst, Arg, PA)) {
        DependingInsts.insert(Inst);
        break;
      }
    }
  } while (!Worklist.empty());

  // Determine whether the original StartBB post-dominates all of the blocks we
  // visited. If not, insert a sentinal indicating that most optimizations are
  // not safe.
  for (SmallPtrSet<const BasicBlock *, 4>::const_iterator I = Visited.begin(),
       E = Visited.end(); I != E; ++I) {
    const BasicBlock *BB = *I;
    if (BB == StartBB)
      continue;
    const TerminatorInst *TI = cast<TerminatorInst>(&BB->back());
    for (succ_const_iterator SI(TI), SE(TI, false); SI != SE; ++SI) {
      const BasicBlock *Succ = *SI;
      if (Succ != StartBB && !Visited.count(Succ)) {
        DependingInsts.insert(reinterpret_cast<Instruction *>(-1));
        return;
      }
    }
  }
}
