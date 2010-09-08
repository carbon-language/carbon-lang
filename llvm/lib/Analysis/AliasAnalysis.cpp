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
AliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                             const Value *P, unsigned Size) {
  // Don't assert AA because BasicAA calls us in order to make use of the
  // logic here.

  ModRefBehavior MRB = getModRefBehavior(CS);
  if (MRB == DoesNotAccessMemory)
    return NoModRef;

  ModRefResult Mask = ModRef;
  if (MRB == OnlyReadsMemory)
    Mask = Ref;
  else if (MRB == AliasAnalysis::AccessesArguments) {
    bool doesAlias = false;
    for (ImmutableCallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
         AI != AE; ++AI)
      if (!isNoAlias(*AI, ~0U, P, Size)) {
        doesAlias = true;
        break;
      }

    if (!doesAlias)
      return NoModRef;
  }

  // If P points to a constant memory location, the call definitely could not
  // modify the memory location.
  if ((Mask & Mod) && pointsToConstantMemory(P))
    Mask = ModRefResult(Mask & ~Mod);

  // If this is BasicAA, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS, P, Size) & Mask);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2) {
  // Don't assert AA because BasicAA calls us in order to make use of the
  // logic here.

  // If CS1 or CS2 are readnone, they don't interact.
  ModRefBehavior CS1B = getModRefBehavior(CS1);
  if (CS1B == DoesNotAccessMemory) return NoModRef;

  ModRefBehavior CS2B = getModRefBehavior(CS2);
  if (CS2B == DoesNotAccessMemory) return NoModRef;

  // If they both only read from memory, there is no dependence.
  if (CS1B == OnlyReadsMemory && CS2B == OnlyReadsMemory)
    return NoModRef;

  AliasAnalysis::ModRefResult Mask = ModRef;

  // If CS1 only reads memory, the only dependence on CS2 can be
  // from CS1 reading memory written by CS2.
  if (CS1B == OnlyReadsMemory)
    Mask = ModRefResult(Mask & Ref);

  // If CS2 only access memory through arguments, accumulate the mod/ref
  // information from CS1's references to the memory referenced by
  // CS2's arguments.
  if (CS2B == AccessesArguments) {
    AliasAnalysis::ModRefResult R = NoModRef;
    for (ImmutableCallSite::arg_iterator
         I = CS2.arg_begin(), E = CS2.arg_end(); I != E; ++I) {
      R = ModRefResult((R | getModRefInfo(CS1, *I, UnknownSize)) & Mask);
      if (R == Mask)
        break;
    }
    return R;
  }

  // If CS1 only accesses memory through arguments, check if CS2 references
  // any of the memory referenced by CS1's arguments. If not, return NoModRef.
  if (CS1B == AccessesArguments) {
    AliasAnalysis::ModRefResult R = NoModRef;
    for (ImmutableCallSite::arg_iterator
         I = CS1.arg_begin(), E = CS1.arg_end(); I != E; ++I)
      if (getModRefInfo(CS2, *I, UnknownSize) != NoModRef) {
        R = Mask;
        break;
      }
    if (R == NoModRef)
      return R;
  }

  // If this is BasicAA, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS1, CS2) & Mask);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  // Don't assert AA because BasicAA calls us in order to make use of the
  // logic here.

  ModRefBehavior Min = UnknownModRefBehavior;

  // Call back into the alias analysis with the other form of getModRefBehavior
  // to see if it can give a better response.
  if (const Function *F = CS.getCalledFunction())
    Min = getModRefBehavior(F);

  // If this is BasicAA, don't forward.
  if (!AA) return Min;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any result we've managed to compute.
  return std::min(AA->getModRefBehavior(CS), Min);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(const Function *F) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefBehavior(F);
}

AliasAnalysis::DependenceResult
AliasAnalysis::getDependence(const Instruction *First,
                             DependenceQueryFlags FirstFlags,
                             const Instruction *Second,
                             DependenceQueryFlags SecondFlags) {
  assert(AA && "AA didn't call InitializeAliasAnalyais in its run method!");
  return AA->getDependence(First, FirstFlags, Second, SecondFlags);
}

//===----------------------------------------------------------------------===//
// AliasAnalysis non-virtual helper method implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const LoadInst *L, const Value *P, unsigned Size) {
  // Be conservative in the face of volatile.
  if (L->isVolatile())
    return ModRef;

  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (!alias(L->getOperand(0), getTypeStoreSize(L->getType()), P, Size))
    return NoModRef;

  // Otherwise, a load just reads.
  return Ref;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const StoreInst *S, const Value *P, unsigned Size) {
  // Be conservative in the face of volatile.
  if (S->isVolatile())
    return ModRef;

  // If the store address cannot alias the pointer in question, then the
  // specified memory cannot be modified by the store.
  if (!alias(S->getOperand(1),
             getTypeStoreSize(S->getOperand(0)->getType()), P, Size))
    return NoModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this store.
  if (pointsToConstantMemory(P))
    return NoModRef;

  // Otherwise, a store just writes.
  return Mod;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const VAArgInst *V, const Value *P, unsigned Size) {
  // If the va_arg address cannot alias the pointer in question, then the
  // specified memory cannot be accessed by the va_arg.
  if (!alias(V->getOperand(0), UnknownSize, P, Size))
    return NoModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this va_arg.
  if (pointsToConstantMemory(P))
    return NoModRef;

  // Otherwise, a va_arg reads and writes.
  return ModRef;
}

AliasAnalysis::DependenceResult
AliasAnalysis::getDependenceViaModRefInfo(const Instruction *First,
                                          DependenceQueryFlags FirstFlags,
                                          const Instruction *Second,
                                          DependenceQueryFlags SecondFlags) {
  if (const LoadInst *L = dyn_cast<LoadInst>(First)) {
    // Be over-conservative with volatile for now.
    if (L->isVolatile())
      return Unknown;

    // Forward this query to getModRefInfo.
    switch (getModRefInfo(Second,
                          L->getPointerOperand(),
                          getTypeStoreSize(L->getType()))) {
    case NoModRef:
      // Second doesn't reference First's memory, so they're independent.
      return Independent;

    case Ref:
      // Second only reads from the memory read from by First. If it
      // also writes to any other memory, be conservative.
      if (Second->mayWriteToMemory())
        return Unknown;

      // If it's loading the same size from the same address, we can
      // give a more precise result.
      if (const LoadInst *SecondL = dyn_cast<LoadInst>(Second)) {
        unsigned LSize = getTypeStoreSize(L->getType());
        unsigned SecondLSize = getTypeStoreSize(SecondL->getType());
        if (alias(L->getPointerOperand(), LSize,
                  SecondL->getPointerOperand(), SecondLSize) ==
            MustAlias) {
          // If the loads are the same size, it's ReadThenRead.
          if (LSize == SecondLSize)
            return ReadThenRead;

          // If the second load is smaller, it's only ReadThenReadSome.
          if (LSize > SecondLSize)
            return ReadThenReadSome;
        }
      }

      // Otherwise it's just two loads.
      return Independent;

    case Mod:
      // Second only writes to the memory read from by First. If it
      // also reads from any other memory, be conservative.
      if (Second->mayReadFromMemory())
        return Unknown;

      // If it's storing the same size to the same address, we can
      // give a more precise result.
      if (const StoreInst *SecondS = dyn_cast<StoreInst>(Second)) {
        unsigned LSize = getTypeStoreSize(L->getType());
        unsigned SecondSSize = getTypeStoreSize(SecondS->getType());
        if (alias(L->getPointerOperand(), LSize,
                  SecondS->getPointerOperand(), SecondSSize) ==
            MustAlias) {
          // If the load and the store are the same size, it's ReadThenWrite.
          if (LSize == SecondSSize)
            return ReadThenWrite;
        }
      }

      // Otherwise we don't know if it could be writing to other memory.
      return Unknown;

    case ModRef:
      // Second reads and writes to the memory read from by First.
      // We don't have a way to express that.
      return Unknown;
    }

  } else if (const StoreInst *S = dyn_cast<StoreInst>(First)) {
    // Be over-conservative with volatile for now.
    if (S->isVolatile())
      return Unknown;

    // Forward this query to getModRefInfo.
    switch (getModRefInfo(Second,
                          S->getPointerOperand(),
                          getTypeStoreSize(S->getValueOperand()->getType()))) {
    case NoModRef:
      // Second doesn't reference First's memory, so they're independent.
      return Independent;

    case Ref:
      // Second only reads from the memory written to by First. If it
      // also writes to any other memory, be conservative.
      if (Second->mayWriteToMemory())
        return Unknown;

      // If it's loading the same size from the same address, we can
      // give a more precise result.
      if (const LoadInst *SecondL = dyn_cast<LoadInst>(Second)) {
        unsigned SSize = getTypeStoreSize(S->getValueOperand()->getType());
        unsigned SecondLSize = getTypeStoreSize(SecondL->getType());
        if (alias(S->getPointerOperand(), SSize,
                  SecondL->getPointerOperand(), SecondLSize) ==
            MustAlias) {
          // If the store and the load are the same size, it's WriteThenRead.
          if (SSize == SecondLSize)
            return WriteThenRead;

          // If the load is smaller, it's only WriteThenReadSome.
          if (SSize > SecondLSize)
            return WriteThenReadSome;
        }
      }

      // Otherwise we don't know if it could be reading from other memory.
      return Unknown;

    case Mod:
      // Second only writes to the memory written to by First. If it
      // also reads from any other memory, be conservative.
      if (Second->mayReadFromMemory())
        return Unknown;

      // If it's storing the same size to the same address, we can
      // give a more precise result.
      if (const StoreInst *SecondS = dyn_cast<StoreInst>(Second)) {
        unsigned SSize = getTypeStoreSize(S->getValueOperand()->getType());
        unsigned SecondSSize = getTypeStoreSize(SecondS->getType());
        if (alias(S->getPointerOperand(), SSize,
                  SecondS->getPointerOperand(), SecondSSize) ==
            MustAlias) {
          // If the stores are the same size, it's WriteThenWrite.
          if (SSize == SecondSSize)
            return WriteThenWrite;

          // If the second store is larger, it's only WriteSomeThenWrite.
          if (SSize < SecondSSize)
            return WriteSomeThenWrite;
        }
      }

      // Otherwise we don't know if it could be writing to other memory.
      return Unknown;

    case ModRef:
      // Second reads and writes to the memory written to by First.
      // We don't have a way to express that.
      return Unknown;
    }

  } else if (const VAArgInst *V = dyn_cast<VAArgInst>(First)) {
    // Forward this query to getModRefInfo.
    if (getModRefInfo(Second, V->getOperand(0), UnknownSize) == NoModRef)
      // Second doesn't reference First's memory, so they're independent.
      return Independent;

  } else if (ImmutableCallSite FirstCS = cast<Value>(First)) {
    // If both instructions are calls/invokes we can use the two-callsite
    // form of getModRefInfo.
    if (ImmutableCallSite SecondCS = cast<Value>(Second))
      // getModRefInfo's arguments are backwards from intuition.
      switch (getModRefInfo(SecondCS, FirstCS)) {
      case NoModRef:
        // Second doesn't reference First's memory, so they're independent.
        return Independent;

      case Ref:
        // If they're both read-only, there's no dependence.
        if (FirstCS.onlyReadsMemory() && SecondCS.onlyReadsMemory())
          return Independent;

        // Otherwise it's not obvious what we can do here.
        return Unknown;

      case Mod:
        // It's not obvious what we can do here.
        return Unknown;

      case ModRef:
        // I know, right?
        return Unknown;
      }
  }

  // For anything else, be conservative.
  return Unknown;
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getIntrinsicModRefBehavior(unsigned iid) {
#define GET_INTRINSIC_MODREF_BEHAVIOR
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_MODREF_BEHAVIOR
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
  BasicBlock::const_iterator I = &I1;
  BasicBlock::const_iterator E = &I2;
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, Ptr, Size) & Mod)
      return true;
  return false;
}

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool llvm::isNoAliasCall(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return ImmutableCallSite(cast<Instruction>(V))
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
