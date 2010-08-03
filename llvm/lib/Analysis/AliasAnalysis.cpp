//===- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

// Register the AliasAnalysis interface, providing a nice name to refer to.
static RegisterAnalysisGroup<AliasAnalysis> Z("Alias Analysis");
char AliasAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult
AliasAnalysis::alias(const Value *V1, unsigned V1Size,
                     const Value *V2, unsigned V2Size) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->alias(V1, V1Size, V2, V2Size);
}

bool AliasAnalysis::pointsToConstantMemory(const Value *P) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->pointsToConstantMemory(P);
}

void AliasAnalysis::deleteValue(Value *V) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->deleteValue(V);
}

void AliasAnalysis::copyValue(Value *From, Value *To) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->copyValue(From, To);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(CallSite CS1, CallSite CS2) {
  // FIXME: we can do better.
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefInfo(CS1, CS2);
}


//===----------------------------------------------------------------------===//
// AliasAnalysis non-virtual helper method implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(LoadInst *L, Value *P, unsigned Size) {
  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (!alias(L->getOperand(0), getTypeStoreSize(L->getType()), P, Size))
    return NoModRef;

  // Be conservative in the face of volatile.
  if (L->isVolatile())
    return ModRef;

  // Otherwise, a load just reads.
  return Ref;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(StoreInst *S, Value *P, unsigned Size) {
  // If the stored address cannot alias the pointer in question, then the
  // pointer cannot be modified by the store.
  if (!alias(S->getOperand(1),
             getTypeStoreSize(S->getOperand(0)->getType()), P, Size))
    return NoModRef;

  // Be conservative in the face of volatile.
  if (S->isVolatile())
    return ModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this store.
  if (pointsToConstantMemory(P))
    return NoModRef;

  // Otherwise, a store just writes.
  return Mod;
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(CallSite CS,
                                 std::vector<PointerAccessInfo> *Info) {
  if (CS.doesNotAccessMemory())
    // Can't do better than this.
    return DoesNotAccessMemory;
  ModRefBehavior MRB = getModRefBehavior(CS.getCalledFunction(), Info);
  if (MRB != DoesNotAccessMemory && CS.onlyReadsMemory())
    return OnlyReadsMemory;
  return MRB;
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(Function *F,
                                 std::vector<PointerAccessInfo> *Info) {
  if (F) {
    if (F->doesNotAccessMemory())
      // Can't do better than this.
      return DoesNotAccessMemory;
    if (F->onlyReadsMemory())
      return OnlyReadsMemory;
    if (unsigned id = F->getIntrinsicID())
      return getModRefBehavior(id);
  }
  return UnknownModRefBehavior;
}

AliasAnalysis::ModRefBehavior AliasAnalysis::getModRefBehavior(unsigned iid) {
#define GET_INTRINSIC_MODREF_BEHAVIOR
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_MODREF_BEHAVIOR
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  ModRefBehavior MRB = getModRefBehavior(CS);
  if (MRB == DoesNotAccessMemory)
    return NoModRef;
  
  ModRefResult Mask = ModRef;
  if (MRB == OnlyReadsMemory)
    Mask = Ref;
  else if (MRB == AliasAnalysis::AccessesArguments) {
    bool doesAlias = false;
    for (CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
         AI != AE; ++AI)
      if (!isNoAlias(*AI, ~0U, P, Size)) {
        doesAlias = true;
        break;
      }

    if (!doesAlias)
      return NoModRef;
  }

  if (!AA) return Mask;

  // If P points to a constant memory location, the call definitely could not
  // modify the memory location.
  if ((Mask & Mod) && AA->pointsToConstantMemory(P))
    Mask = ModRefResult(Mask & ~Mod);

  return ModRefResult(Mask & AA->getModRefInfo(CS, P, Size));
}

// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

/// InitializeAliasAnalysis - Subclasses must call this method to initialize the
/// AliasAnalysis interface before any other methods are called.
///
void AliasAnalysis::InitializeAliasAnalysis(Pass *P) {
  TD = P->getAnalysisIfAvailable<TargetData>();
  AA = &P->getAnalysis<AliasAnalysis>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)).
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();         // All AA's chain
}

/// getTypeStoreSize - Return the TargetData store size for the given type,
/// if known, or a conservative value otherwise.
///
unsigned AliasAnalysis::getTypeStoreSize(const Type *Ty) {
  return TD ? TD->getTypeStoreSize(Ty) : ~0u;
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const Value *Ptr, unsigned Size) {
  return canInstructionRangeModify(BB.front(), BB.back(), Ptr, Size);
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr, unsigned Size) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::iterator I = const_cast<Instruction*>(&I1);
  BasicBlock::iterator E = const_cast<Instruction*>(&I2);
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, const_cast<Value*>(Ptr), Size) & Mod)
      return true;
  return false;
}

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool llvm::isNoAliasCall(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return CallSite(const_cast<Instruction*>(cast<Instruction>(V)))
      .paramHasAttr(0, Attribute::NoAlias);
  return false;
}

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas and Mallocs
///    ByVal and NoAlias Arguments
///    NoAlias returns
///
bool llvm::isIdentifiedObject(const Value *V) {
  if (isa<AllocaInst>(V))
    return true;
  if (isa<GlobalValue>(V) && !isa<GlobalAlias>(V))
    return true;
  if (isNoAliasCall(V))
    return true;
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr() || A->hasByValAttr();
  return false;
}

// Because of the way .a files work, we must force the BasicAA implementation to
// be pulled in if the AliasAnalysis classes are pulled in.  Otherwise we run
// the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.
DEFINING_FILE_FOR(AliasAnalysis)
