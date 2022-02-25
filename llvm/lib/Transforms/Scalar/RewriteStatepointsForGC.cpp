//===- RewriteStatepointsForGC.cpp - Make GC relocations explicit ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrite call/invoke instructions so as to make potential relocations
// performed by the garbage collector explicit in the IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "rewrite-statepoints-for-gc"

using namespace llvm;

// Print the liveset found at the insert location
static cl::opt<bool> PrintLiveSet("spp-print-liveset", cl::Hidden,
                                  cl::init(false));
static cl::opt<bool> PrintLiveSetSize("spp-print-liveset-size", cl::Hidden,
                                      cl::init(false));

// Print out the base pointers for debugging
static cl::opt<bool> PrintBasePointers("spp-print-base-pointers", cl::Hidden,
                                       cl::init(false));

// Cost threshold measuring when it is profitable to rematerialize value instead
// of relocating it
static cl::opt<unsigned>
RematerializationThreshold("spp-rematerialization-threshold", cl::Hidden,
                           cl::init(6));

#ifdef EXPENSIVE_CHECKS
static bool ClobberNonLive = true;
#else
static bool ClobberNonLive = false;
#endif

static cl::opt<bool, true> ClobberNonLiveOverride("rs4gc-clobber-non-live",
                                                  cl::location(ClobberNonLive),
                                                  cl::Hidden);

static cl::opt<bool>
    AllowStatepointWithNoDeoptInfo("rs4gc-allow-statepoint-with-no-deopt-info",
                                   cl::Hidden, cl::init(true));

/// The IR fed into RewriteStatepointsForGC may have had attributes and
/// metadata implying dereferenceability that are no longer valid/correct after
/// RewriteStatepointsForGC has run. This is because semantically, after
/// RewriteStatepointsForGC runs, all calls to gc.statepoint "free" the entire
/// heap. stripNonValidData (conservatively) restores
/// correctness by erasing all attributes in the module that externally imply
/// dereferenceability. Similar reasoning also applies to the noalias
/// attributes and metadata. gc.statepoint can touch the entire heap including
/// noalias objects.
/// Apart from attributes and metadata, we also remove instructions that imply
/// constant physical memory: llvm.invariant.start.
static void stripNonValidData(Module &M);

static bool shouldRewriteStatepointsIn(Function &F);

PreservedAnalyses RewriteStatepointsForGC::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  bool Changed = false;
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (Function &F : M) {
    // Nothing to do for declarations.
    if (F.isDeclaration() || F.empty())
      continue;

    // Policy choice says not to rewrite - the most common reason is that we're
    // compiling code without a GCStrategy.
    if (!shouldRewriteStatepointsIn(F))
      continue;

    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
    auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    Changed |= runOnFunction(F, DT, TTI, TLI);
  }
  if (!Changed)
    return PreservedAnalyses::all();

  // stripNonValidData asserts that shouldRewriteStatepointsIn
  // returns true for at least one function in the module.  Since at least
  // one function changed, we know that the precondition is satisfied.
  stripNonValidData(M);

  PreservedAnalyses PA;
  PA.preserve<TargetIRAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  return PA;
}

namespace {

class RewriteStatepointsForGCLegacyPass : public ModulePass {
  RewriteStatepointsForGC Impl;

public:
  static char ID; // Pass identification, replacement for typeid

  RewriteStatepointsForGCLegacyPass() : ModulePass(ID), Impl() {
    initializeRewriteStatepointsForGCLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Function &F : M) {
      // Nothing to do for declarations.
      if (F.isDeclaration() || F.empty())
        continue;

      // Policy choice says not to rewrite - the most common reason is that
      // we're compiling code without a GCStrategy.
      if (!shouldRewriteStatepointsIn(F))
        continue;

      TargetTransformInfo &TTI =
          getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
      const TargetLibraryInfo &TLI =
          getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
      auto &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

      Changed |= Impl.runOnFunction(F, DT, TTI, TLI);
    }

    if (!Changed)
      return false;

    // stripNonValidData asserts that shouldRewriteStatepointsIn
    // returns true for at least one function in the module.  Since at least
    // one function changed, we know that the precondition is satisfied.
    stripNonValidData(M);
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We add and rewrite a bunch of instructions, but don't really do much
    // else.  We could in theory preserve a lot more analyses here.
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

} // end anonymous namespace

char RewriteStatepointsForGCLegacyPass::ID = 0;

ModulePass *llvm::createRewriteStatepointsForGCLegacyPass() {
  return new RewriteStatepointsForGCLegacyPass();
}

INITIALIZE_PASS_BEGIN(RewriteStatepointsForGCLegacyPass,
                      "rewrite-statepoints-for-gc",
                      "Make relocations explicit at statepoints", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(RewriteStatepointsForGCLegacyPass,
                    "rewrite-statepoints-for-gc",
                    "Make relocations explicit at statepoints", false, false)

namespace {

struct GCPtrLivenessData {
  /// Values defined in this block.
  MapVector<BasicBlock *, SetVector<Value *>> KillSet;

  /// Values used in this block (and thus live); does not included values
  /// killed within this block.
  MapVector<BasicBlock *, SetVector<Value *>> LiveSet;

  /// Values live into this basic block (i.e. used by any
  /// instruction in this basic block or ones reachable from here)
  MapVector<BasicBlock *, SetVector<Value *>> LiveIn;

  /// Values live out of this basic block (i.e. live into
  /// any successor block)
  MapVector<BasicBlock *, SetVector<Value *>> LiveOut;
};

// The type of the internal cache used inside the findBasePointers family
// of functions.  From the callers perspective, this is an opaque type and
// should not be inspected.
//
// In the actual implementation this caches two relations:
// - The base relation itself (i.e. this pointer is based on that one)
// - The base defining value relation (i.e. before base_phi insertion)
// Generally, after the execution of a full findBasePointer call, only the
// base relation will remain.  Internally, we add a mixture of the two
// types, then update all the second type to the first type
using DefiningValueMapTy = MapVector<Value *, Value *>;
using StatepointLiveSetTy = SetVector<Value *>;
using RematerializedValueMapTy =
    MapVector<AssertingVH<Instruction>, AssertingVH<Value>>;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live across this safepoint
  StatepointLiveSetTy LiveSet;

  /// Mapping from live pointers to a base-defining-value
  MapVector<Value *, Value *> PointerToBase;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  GCStatepointInst *StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  Instruction *UnwindToken;

  /// Record live values we are rematerialized instead of relocating.
  /// They are not included into 'LiveSet' field.
  /// Maps rematerialized copy to it's original value.
  RematerializedValueMapTy RematerializedValues;
};

} // end anonymous namespace

static ArrayRef<Use> GetDeoptBundleOperands(const CallBase *Call) {
  Optional<OperandBundleUse> DeoptBundle =
      Call->getOperandBundle(LLVMContext::OB_deopt);

  if (!DeoptBundle.hasValue()) {
    assert(AllowStatepointWithNoDeoptInfo &&
           "Found non-leaf call without deopt info!");
    return None;
  }

  return DeoptBundle.getValue().Inputs;
}

/// Compute the live-in set for every basic block in the function
static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data);

/// Given results from the dataflow liveness computation, find the set of live
/// Values at a particular instruction.
static void findLiveSetAtInst(Instruction *inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &out);

// TODO: Once we can get to the GCStrategy, this becomes
// Optional<bool> isGCManagedPointer(const Type *Ty) const override {

static bool isGCPointerType(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return PT->getAddressSpace() == 1;
  return false;
}

// Return true if this type is one which a) is a gc pointer or contains a GC
// pointer and b) is of a type this code expects to encounter as a live value.
// (The insertion code will assert that a type which matches (a) and not (b)
// is not encountered.)
static bool isHandledGCPointerType(Type *T) {
  // We fully support gc pointers
  if (isGCPointerType(T))
    return true;
  // We partially support vectors of gc pointers. The code will assert if it
  // can't handle something.
  if (auto VT = dyn_cast<VectorType>(T))
    if (isGCPointerType(VT->getElementType()))
      return true;
  return false;
}

#ifndef NDEBUG
/// Returns true if this type contains a gc pointer whether we know how to
/// handle that type or not.
static bool containsGCPtrType(Type *Ty) {
  if (isGCPointerType(Ty))
    return true;
  if (VectorType *VT = dyn_cast<VectorType>(Ty))
    return isGCPointerType(VT->getScalarType());
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty))
    return containsGCPtrType(AT->getElementType());
  if (StructType *ST = dyn_cast<StructType>(Ty))
    return llvm::any_of(ST->elements(), containsGCPtrType);
  return false;
}

// Returns true if this is a type which a) is a gc pointer or contains a GC
// pointer and b) is of a type which the code doesn't expect (i.e. first class
// aggregates).  Used to trip assertions.
static bool isUnhandledGCPointerType(Type *Ty) {
  return containsGCPtrType(Ty) && !isHandledGCPointerType(Ty);
}
#endif

// Return the name of the value suffixed with the provided value, or if the
// value didn't have a name, the default value specified.
static std::string suffixed_name_or(Value *V, StringRef Suffix,
                                    StringRef DefaultName) {
  return V->hasName() ? (V->getName() + Suffix).str() : DefaultName.str();
}

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
static void analyzeParsePointLiveness(
    DominatorTree &DT, GCPtrLivenessData &OriginalLivenessData, CallBase *Call,
    PartiallyConstructedSafepointRecord &Result) {
  StatepointLiveSetTy LiveSet;
  findLiveSetAtInst(Call, OriginalLivenessData, LiveSet);

  if (PrintLiveSet) {
    dbgs() << "Live Variables:\n";
    for (Value *V : LiveSet)
      dbgs() << " " << V->getName() << " " << *V << "\n";
  }
  if (PrintLiveSetSize) {
    dbgs() << "Safepoint For: " << Call->getCalledOperand()->getName() << "\n";
    dbgs() << "Number live values: " << LiveSet.size() << "\n";
  }
  Result.LiveSet = LiveSet;
}

// Returns true is V is a knownBaseResult.
static bool isKnownBaseResult(Value *V);

// Returns true if V is a BaseResult that already exists in the IR, i.e. it is
// not created by the findBasePointers algorithm.
static bool isOriginalBaseResult(Value *V);

namespace {

/// A single base defining value - An immediate base defining value for an
/// instruction 'Def' is an input to 'Def' whose base is also a base of 'Def'.
/// For instructions which have multiple pointer [vector] inputs or that
/// transition between vector and scalar types, there is no immediate base
/// defining value.  The 'base defining value' for 'Def' is the transitive
/// closure of this relation stopping at the first instruction which has no
/// immediate base defining value.  The b.d.v. might itself be a base pointer,
/// but it can also be an arbitrary derived pointer.
struct BaseDefiningValueResult {
  /// Contains the value which is the base defining value.
  Value * const BDV;

  /// True if the base defining value is also known to be an actual base
  /// pointer.
  const bool IsKnownBase;

  BaseDefiningValueResult(Value *BDV, bool IsKnownBase)
    : BDV(BDV), IsKnownBase(IsKnownBase) {
#ifndef NDEBUG
    // Check consistency between new and old means of checking whether a BDV is
    // a base.
    bool MustBeBase = isKnownBaseResult(BDV);
    assert(!MustBeBase || MustBeBase == IsKnownBase);
#endif
  }
};

} // end anonymous namespace

static BaseDefiningValueResult findBaseDefiningValue(Value *I);

/// Return a base defining value for the 'Index' element of the given vector
/// instruction 'I'.  If Index is null, returns a BDV for the entire vector
/// 'I'.  As an optimization, this method will try to determine when the
/// element is known to already be a base pointer.  If this can be established,
/// the second value in the returned pair will be true.  Note that either a
/// vector or a pointer typed value can be returned.  For the former, the
/// vector returned is a BDV (and possibly a base) of the entire vector 'I'.
/// If the later, the return pointer is a BDV (or possibly a base) for the
/// particular element in 'I'.
static BaseDefiningValueResult
findBaseDefiningValueOfVector(Value *I) {
  // Each case parallels findBaseDefiningValue below, see that code for
  // detailed motivation.

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    return BaseDefiningValueResult(I, true);

  if (isa<Constant>(I))
    // Base of constant vector consists only of constant null pointers.
    // For reasoning see similar case inside 'findBaseDefiningValue' function.
    return BaseDefiningValueResult(ConstantAggregateZero::get(I->getType()),
                                   true);

  if (isa<LoadInst>(I))
    return BaseDefiningValueResult(I, true);

  if (isa<InsertElementInst>(I))
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    return BaseDefiningValueResult(I, false);

  if (isa<ShuffleVectorInst>(I))
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    // TODO: There a number of local optimizations which could be applied here
    // for particular sufflevector patterns.
    return BaseDefiningValueResult(I, false);

  // The behavior of getelementptr instructions is the same for vector and
  // non-vector data types.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I))
    return findBaseDefiningValue(GEP->getPointerOperand());

  // If the pointer comes through a bitcast of a vector of pointers to
  // a vector of another type of pointer, then look through the bitcast
  if (auto *BC = dyn_cast<BitCastInst>(I))
    return findBaseDefiningValue(BC->getOperand(0));

  // We assume that functions in the source language only return base
  // pointers.  This should probably be generalized via attributes to support
  // both source language and internal functions.
  if (isa<CallInst>(I) || isa<InvokeInst>(I))
    return BaseDefiningValueResult(I, true);

  // A PHI or Select is a base defining value.  The outer findBasePointer
  // algorithm is responsible for constructing a base value for this BDV.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "unknown vector instruction - no base found for vector element");
  return BaseDefiningValueResult(I, false);
}

/// Helper function for findBasePointer - Will return a value which either a)
/// defines the base pointer for the input, b) blocks the simple search
/// (i.e. a PHI or Select of two derived pointers), or c) involves a change
/// from pointer to vector type or back.
static BaseDefiningValueResult findBaseDefiningValue(Value *I) {
  assert(I->getType()->isPtrOrPtrVectorTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  if (I->getType()->isVectorTy())
    return findBaseDefiningValueOfVector(I);

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    // We should have never reached here if this argument isn't an gc value
    return BaseDefiningValueResult(I, true);

  if (isa<Constant>(I)) {
    // We assume that objects with a constant base (e.g. a global) can't move
    // and don't need to be reported to the collector because they are always
    // live. Besides global references, all kinds of constants (e.g. undef,
    // constant expressions, null pointers) can be introduced by the inliner or
    // the optimizer, especially on dynamically dead paths.
    // Here we treat all of them as having single null base. By doing this we
    // trying to avoid problems reporting various conflicts in a form of
    // "phi (const1, const2)" or "phi (const, regular gc ptr)".
    // See constant.ll file for relevant test cases.

    return BaseDefiningValueResult(
        ConstantPointerNull::get(cast<PointerType>(I->getType())), true);
  }

  // inttoptrs in an integral address space are currently ill-defined.  We
  // treat them as defining base pointers here for consistency with the
  // constant rule above and because we don't really have a better semantic
  // to give them.  Note that the optimizer is always free to insert undefined
  // behavior on dynamically dead paths as well.
  if (isa<IntToPtrInst>(I))
    return BaseDefiningValueResult(I, true);

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value *Def = CI->stripPointerCasts();
    // If stripping pointer casts changes the address space there is an
    // addrspacecast in between.
    assert(cast<PointerType>(Def->getType())->getAddressSpace() ==
               cast<PointerType>(CI->getType())->getAddressSpace() &&
           "unsupported addrspacecast");
    // If we find a cast instruction here, it means we've found a cast which is
    // not simply a pointer cast (i.e. an inttoptr).  We don't know how to
    // handle int->ptr conversion.
    assert(!isa<CastInst>(Def) && "shouldn't find another cast here");
    return findBaseDefiningValue(Def);
  }

  if (isa<LoadInst>(I))
    // The value loaded is an gc base itself
    return BaseDefiningValueResult(I, true);

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I))
    // The base of this GEP is the base
    return findBaseDefiningValue(GEP->getPointerOperand());

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      // fall through to general call handling
      break;
    case Intrinsic::experimental_gc_statepoint:
      llvm_unreachable("statepoints don't produce pointers");
    case Intrinsic::experimental_gc_relocate:
      // Rerunning safepoint insertion after safepoints are already
      // inserted is not supported.  It could probably be made to work,
      // but why are you doing this?  There's no good reason.
      llvm_unreachable("repeat safepoint insertion is not supported");
    case Intrinsic::gcroot:
      // Currently, this mechanism hasn't been extended to work with gcroot.
      // There's no reason it couldn't be, but I haven't thought about the
      // implications much.
      llvm_unreachable(
          "interaction with the gcroot mechanism is not supported");
    case Intrinsic::experimental_gc_get_pointer_base:
      return findBaseDefiningValue(II->getOperand(0));
    }
  }
  // We assume that functions in the source language only return base
  // pointers.  This should probably be generalized via attributes to support
  // both source language and internal functions.
  if (isa<CallInst>(I) || isa<InvokeInst>(I))
    return BaseDefiningValueResult(I, true);

  // TODO: I have absolutely no idea how to implement this part yet.  It's not
  // necessarily hard, I just haven't really looked at it yet.
  assert(!isa<LandingPadInst>(I) && "Landing Pad is unimplemented");

  if (isa<AtomicCmpXchgInst>(I))
    // A CAS is effectively a atomic store and load combined under a
    // predicate.  From the perspective of base pointers, we just treat it
    // like a load.
    return BaseDefiningValueResult(I, true);

  assert(!isa<AtomicRMWInst>(I) && "Xchg handled above, all others are "
                                   "binary ops which don't apply to pointers");

  // The aggregate ops.  Aggregates can either be in the heap or on the
  // stack, but in either case, this is simply a field load.  As a result,
  // this is a defining definition of the base just like a load is.
  if (isa<ExtractValueInst>(I))
    return BaseDefiningValueResult(I, true);

  // We should never see an insert vector since that would require we be
  // tracing back a struct value not a pointer value.
  assert(!isa<InsertValueInst>(I) &&
         "Base pointer for a struct is meaningless");

  // This value might have been generated by findBasePointer() called when
  // substituting gc.get.pointer.base() intrinsic.
  bool IsKnownBase =
      isa<Instruction>(I) && cast<Instruction>(I)->getMetadata("is_base_value");

  // An extractelement produces a base result exactly when it's input does.
  // We may need to insert a parallel instruction to extract the appropriate
  // element out of the base vector corresponding to the input. Given this,
  // it's analogous to the phi and select case even though it's not a merge.
  if (isa<ExtractElementInst>(I))
    // Note: There a lot of obvious peephole cases here.  This are deliberately
    // handled after the main base pointer inference algorithm to make writing
    // test cases to exercise that code easier.
    return BaseDefiningValueResult(I, IsKnownBase);

  // The last two cases here don't return a base pointer.  Instead, they
  // return a value which dynamically selects from among several base
  // derived pointers (each with it's own base potentially).  It's the job of
  // the caller to resolve these.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "missing instruction case in findBaseDefiningValing");
  return BaseDefiningValueResult(I, IsKnownBase);
}

/// Returns the base defining value for this value.
static Value *findBaseDefiningValueCached(Value *I, DefiningValueMapTy &Cache) {
  Value *&Cached = Cache[I];
  if (!Cached) {
    Cached = findBaseDefiningValue(I).BDV;
    LLVM_DEBUG(dbgs() << "fBDV-cached: " << I->getName() << " -> "
                      << Cached->getName() << "\n");
  }
  assert(Cache[I] != nullptr);
  return Cached;
}

/// Return a base pointer for this value if known.  Otherwise, return it's
/// base defining value.
static Value *findBaseOrBDV(Value *I, DefiningValueMapTy &Cache) {
  Value *Def = findBaseDefiningValueCached(I, Cache);
  auto Found = Cache.find(Def);
  if (Found != Cache.end()) {
    // Either a base-of relation, or a self reference.  Caller must check.
    return Found->second;
  }
  // Only a BDV available
  return Def;
}

/// This value is a base pointer that is not generated by RS4GC, i.e. it already
/// exists in the code.
static bool isOriginalBaseResult(Value *V) {
  // no recursion possible
  return !isa<PHINode>(V) && !isa<SelectInst>(V) &&
         !isa<ExtractElementInst>(V) && !isa<InsertElementInst>(V) &&
         !isa<ShuffleVectorInst>(V);
}

/// Given the result of a call to findBaseDefiningValue, or findBaseOrBDV,
/// is it known to be a base pointer?  Or do we need to continue searching.
static bool isKnownBaseResult(Value *V) {
  if (isOriginalBaseResult(V))
    return true;
  if (isa<Instruction>(V) &&
      cast<Instruction>(V)->getMetadata("is_base_value")) {
    // This is a previously inserted base phi or select.  We know
    // that this is a base value.
    return true;
  }

  // We need to keep searching
  return false;
}

// Returns true if First and Second values are both scalar or both vector.
static bool areBothVectorOrScalar(Value *First, Value *Second) {
  return isa<VectorType>(First->getType()) ==
         isa<VectorType>(Second->getType());
}

namespace {

/// Models the state of a single base defining value in the findBasePointer
/// algorithm for determining where a new instruction is needed to propagate
/// the base of this BDV.
class BDVState {
public:
  enum StatusTy {
     // Starting state of lattice
     Unknown,
     // Some specific base value -- does *not* mean that instruction
     // propagates the base of the object
     // ex: gep %arg, 16 -> %arg is the base value
     Base,
     // Need to insert a node to represent a merge.
     Conflict
  };

  BDVState() {
    llvm_unreachable("missing state in map");
  }

  explicit BDVState(Value *OriginalValue)
    : OriginalValue(OriginalValue) {}
  explicit BDVState(Value *OriginalValue, StatusTy Status, Value *BaseValue = nullptr)
    : OriginalValue(OriginalValue), Status(Status), BaseValue(BaseValue) {
    assert(Status != Base || BaseValue);
  }

  StatusTy getStatus() const { return Status; }
  Value *getOriginalValue() const { return OriginalValue; }
  Value *getBaseValue() const { return BaseValue; }

  bool isBase() const { return getStatus() == Base; }
  bool isUnknown() const { return getStatus() == Unknown; }
  bool isConflict() const { return getStatus() == Conflict; }

  // Values of type BDVState form a lattice, and this function implements the
  // meet
  // operation.
  void meet(const BDVState &Other) {
    auto markConflict = [&]() {
      Status = BDVState::Conflict;
      BaseValue = nullptr;
    };
    // Conflict is a final state.
    if (isConflict())
      return;
    // if we are not known - just take other state.
    if (isUnknown()) {
      Status = Other.getStatus();
      BaseValue = Other.getBaseValue();
      return;
    }
    // We are base.
    assert(isBase() && "Unknown state");
    // If other is unknown - just keep our state.
    if (Other.isUnknown())
      return;
    // If other is conflict - it is a final state.
    if (Other.isConflict())
      return markConflict();
    // Other is base as well.
    assert(Other.isBase() && "Unknown state");
    // If bases are different - Conflict.
    if (getBaseValue() != Other.getBaseValue())
      return markConflict();
    // We are identical, do nothing.
  }

  bool operator==(const BDVState &Other) const {
    return OriginalValue == OriginalValue && BaseValue == Other.BaseValue &&
      Status == Other.Status;
  }

  bool operator!=(const BDVState &other) const { return !(*this == other); }

  LLVM_DUMP_METHOD
  void dump() const {
    print(dbgs());
    dbgs() << '\n';
  }

  void print(raw_ostream &OS) const {
    switch (getStatus()) {
    case Unknown:
      OS << "U";
      break;
    case Base:
      OS << "B";
      break;
    case Conflict:
      OS << "C";
      break;
    }
    OS << " (base " << getBaseValue() << " - "
       << (getBaseValue() ? getBaseValue()->getName() : "nullptr") << ")"
       << " for  "  << OriginalValue->getName() << ":";
  }

private:
  AssertingVH<Value> OriginalValue; // instruction this state corresponds to
  StatusTy Status = Unknown;
  AssertingVH<Value> BaseValue = nullptr; // Non-null only if Status == Base.
};

} // end anonymous namespace

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS, const BDVState &State) {
  State.print(OS);
  return OS;
}
#endif

/// For a given value or instruction, figure out what base ptr its derived from.
/// For gc objects, this is simply itself.  On success, returns a value which is
/// the base pointer.  (This is reliable and can be used for relocation.)  On
/// failure, returns nullptr.
static Value *findBasePointer(Value *I, DefiningValueMapTy &Cache) {
  Value *Def = findBaseOrBDV(I, Cache);

  if (isKnownBaseResult(Def) && areBothVectorOrScalar(Def, I))
    return Def;

  // Here's the rough algorithm:
  // - For every SSA value, construct a mapping to either an actual base
  //   pointer or a PHI which obscures the base pointer.
  // - Construct a mapping from PHI to unknown TOP state.  Use an
  //   optimistic algorithm to propagate base pointer information.  Lattice
  //   looks like:
  //   UNKNOWN
  //   b1 b2 b3 b4
  //   CONFLICT
  //   When algorithm terminates, all PHIs will either have a single concrete
  //   base or be in a conflict state.
  // - For every conflict, insert a dummy PHI node without arguments.  Add
  //   these to the base[Instruction] = BasePtr mapping.  For every
  //   non-conflict, add the actual base.
  //  - For every conflict, add arguments for the base[a] of each input
  //   arguments.
  //
  // Note: A simpler form of this would be to add the conflict form of all
  // PHIs without running the optimistic algorithm.  This would be
  // analogous to pessimistic data flow and would likely lead to an
  // overall worse solution.

#ifndef NDEBUG
  auto isExpectedBDVType = [](Value *BDV) {
    return isa<PHINode>(BDV) || isa<SelectInst>(BDV) ||
           isa<ExtractElementInst>(BDV) || isa<InsertElementInst>(BDV) ||
           isa<ShuffleVectorInst>(BDV);
  };
#endif

  // Once populated, will contain a mapping from each potentially non-base BDV
  // to a lattice value (described above) which corresponds to that BDV.
  // We use the order of insertion (DFS over the def/use graph) to provide a
  // stable deterministic ordering for visiting DenseMaps (which are unordered)
  // below.  This is important for deterministic compilation.
  MapVector<Value *, BDVState> States;

#ifndef NDEBUG
  auto VerifyStates = [&]() {
    for (auto &Entry : States) {
      assert(Entry.first == Entry.second.getOriginalValue());
    }
  };
#endif

  auto visitBDVOperands = [](Value *BDV, std::function<void (Value*)> F) {
    if (PHINode *PN = dyn_cast<PHINode>(BDV)) {
      for (Value *InVal : PN->incoming_values())
        F(InVal);
    } else if (SelectInst *SI = dyn_cast<SelectInst>(BDV)) {
      F(SI->getTrueValue());
      F(SI->getFalseValue());
    } else if (auto *EE = dyn_cast<ExtractElementInst>(BDV)) {
      F(EE->getVectorOperand());
    } else if (auto *IE = dyn_cast<InsertElementInst>(BDV)) {
      F(IE->getOperand(0));
      F(IE->getOperand(1));
    } else if (auto *SV = dyn_cast<ShuffleVectorInst>(BDV)) {
      // For a canonical broadcast, ignore the undef argument
      // (without this, we insert a parallel base shuffle for every broadcast)
      F(SV->getOperand(0));
      if (!SV->isZeroEltSplat())
        F(SV->getOperand(1));
    } else {
      llvm_unreachable("unexpected BDV type");
    }
  };


  // Recursively fill in all base defining values reachable from the initial
  // one for which we don't already know a definite base value for
  /* scope */ {
    SmallVector<Value*, 16> Worklist;
    Worklist.push_back(Def);
    States.insert({Def, BDVState(Def)});
    while (!Worklist.empty()) {
      Value *Current = Worklist.pop_back_val();
      assert(!isOriginalBaseResult(Current) && "why did it get added?");

      auto visitIncomingValue = [&](Value *InVal) {
        Value *Base = findBaseOrBDV(InVal, Cache);
        if (isKnownBaseResult(Base) && areBothVectorOrScalar(Base, InVal))
          // Known bases won't need new instructions introduced and can be
          // ignored safely. However, this can only be done when InVal and Base
          // are both scalar or both vector. Otherwise, we need to find a
          // correct BDV for InVal, by creating an entry in the lattice
          // (States).
          return;
        assert(isExpectedBDVType(Base) && "the only non-base values "
               "we see should be base defining values");
        if (States.insert(std::make_pair(Base, BDVState(Base))).second)
          Worklist.push_back(Base);
      };

      visitBDVOperands(Current, visitIncomingValue);
    }
  }

#ifndef NDEBUG
  VerifyStates();
  LLVM_DEBUG(dbgs() << "States after initialization:\n");
  for (auto Pair : States) {
    LLVM_DEBUG(dbgs() << " " << Pair.second << " for " << *Pair.first << "\n");
  }
#endif

  // Iterate forward through the value graph pruning any node from the state
  // list where all of the inputs are base pointers.  The purpose of this is to
  // reuse existing values when the derived pointer we were asked to materialize
  // a base pointer for happens to be a base pointer itself.  (Or a sub-graph
  // feeding it does.)
  SmallVector<Value *> ToRemove;
  do {
    ToRemove.clear();
    for (auto Pair : States) {
      Value *BDV = Pair.first;
      auto canPruneInput = [&](Value *V) {
        Value *BDV = findBaseOrBDV(V, Cache);
        if (V->stripPointerCasts() != BDV)
          return false;
        // The assumption is that anything not in the state list is
        // propagates a base pointer.
        return States.count(BDV) == 0;
      };

      bool CanPrune = true;
      visitBDVOperands(BDV, [&](Value *Op) {
        CanPrune = CanPrune && canPruneInput(Op);
      });
      if (CanPrune)
        ToRemove.push_back(BDV);
    }
    for (Value *V : ToRemove) {
      States.erase(V);
      // Cache the fact V is it's own base for later usage.
      Cache[V] = V;
    }
  } while (!ToRemove.empty());

  // Did we manage to prove that Def itself must be a base pointer?
  if (!States.count(Def))
    return Def;

  // Return a phi state for a base defining value.  We'll generate a new
  // base state for known bases and expect to find a cached state otherwise.
  auto GetStateForBDV = [&](Value *BaseValue, Value *Input) {
    auto I = States.find(BaseValue);
    if (I != States.end())
      return I->second;
    assert(areBothVectorOrScalar(BaseValue, Input));
    return BDVState(BaseValue, BDVState::Base, BaseValue);
  };

  bool Progress = true;
  while (Progress) {
#ifndef NDEBUG
    const size_t OldSize = States.size();
#endif
    Progress = false;
    // We're only changing values in this loop, thus safe to keep iterators.
    // Since this is computing a fixed point, the order of visit does not
    // effect the result.  TODO: We could use a worklist here and make this run
    // much faster.
    for (auto Pair : States) {
      Value *BDV = Pair.first;
      // Only values that do not have known bases or those that have differing
      // type (scalar versus vector) from a possible known base should be in the
      // lattice.
      assert((!isKnownBaseResult(BDV) ||
             !areBothVectorOrScalar(BDV, Pair.second.getBaseValue())) &&
                 "why did it get added?");

      BDVState NewState(BDV);
      visitBDVOperands(BDV, [&](Value *Op) {
        Value *BDV = findBaseOrBDV(Op, Cache);
        auto OpState = GetStateForBDV(BDV, Op);
        NewState.meet(OpState);
      });

      BDVState OldState = States[BDV];
      if (OldState != NewState) {
        Progress = true;
        States[BDV] = NewState;
      }
    }

    assert(OldSize == States.size() &&
           "fixed point shouldn't be adding any new nodes to state");
  }

#ifndef NDEBUG
  VerifyStates();
  LLVM_DEBUG(dbgs() << "States after meet iteration:\n");
  for (auto Pair : States) {
    LLVM_DEBUG(dbgs() << " " << Pair.second << " for " << *Pair.first << "\n");
  }
#endif

  // Handle all instructions that have a vector BDV, but the instruction itself
  // is of scalar type.
  for (auto Pair : States) {
    Instruction *I = cast<Instruction>(Pair.first);
    BDVState State = Pair.second;
    auto *BaseValue = State.getBaseValue();
    // Only values that do not have known bases or those that have differing
    // type (scalar versus vector) from a possible known base should be in the
    // lattice.
    assert((!isKnownBaseResult(I) || !areBothVectorOrScalar(I, BaseValue)) &&
           "why did it get added?");
    assert(!State.isUnknown() && "Optimistic algorithm didn't complete!");

    if (!State.isBase() || !isa<VectorType>(BaseValue->getType()))
      continue;
    // extractelement instructions are a bit special in that we may need to
    // insert an extract even when we know an exact base for the instruction.
    // The problem is that we need to convert from a vector base to a scalar
    // base for the particular indice we're interested in.
    if (isa<ExtractElementInst>(I)) {
      auto *EE = cast<ExtractElementInst>(I);
      // TODO: In many cases, the new instruction is just EE itself.  We should
      // exploit this, but can't do it here since it would break the invariant
      // about the BDV not being known to be a base.
      auto *BaseInst = ExtractElementInst::Create(
          State.getBaseValue(), EE->getIndexOperand(), "base_ee", EE);
      BaseInst->setMetadata("is_base_value", MDNode::get(I->getContext(), {}));
      States[I] = BDVState(I, BDVState::Base, BaseInst);
    } else if (!isa<VectorType>(I->getType())) {
      // We need to handle cases that have a vector base but the instruction is
      // a scalar type (these could be phis or selects or any instruction that
      // are of scalar type, but the base can be a vector type).  We
      // conservatively set this as conflict.  Setting the base value for these
      // conflicts is handled in the next loop which traverses States.
      States[I] = BDVState(I, BDVState::Conflict);
    }
  }

#ifndef NDEBUG
  VerifyStates();
#endif

  // Insert Phis for all conflicts
  // TODO: adjust naming patterns to avoid this order of iteration dependency
  for (auto Pair : States) {
    Instruction *I = cast<Instruction>(Pair.first);
    BDVState State = Pair.second;
    // Only values that do not have known bases or those that have differing
    // type (scalar versus vector) from a possible known base should be in the
    // lattice.
    assert((!isKnownBaseResult(I) || !areBothVectorOrScalar(I, State.getBaseValue())) &&
           "why did it get added?");
    assert(!State.isUnknown() && "Optimistic algorithm didn't complete!");

    // Since we're joining a vector and scalar base, they can never be the
    // same.  As a result, we should always see insert element having reached
    // the conflict state.
    assert(!isa<InsertElementInst>(I) || State.isConflict());

    if (!State.isConflict())
      continue;

    auto getMangledName = [](Instruction *I) -> std::string {
      if (isa<PHINode>(I)) {
        return suffixed_name_or(I, ".base", "base_phi");
      } else if (isa<SelectInst>(I)) {
        return suffixed_name_or(I, ".base", "base_select");
      } else if (isa<ExtractElementInst>(I)) {
        return suffixed_name_or(I, ".base", "base_ee");
      } else if (isa<InsertElementInst>(I)) {
        return suffixed_name_or(I, ".base", "base_ie");
      } else {
        return suffixed_name_or(I, ".base", "base_sv");
      }
    };

    Instruction *BaseInst = I->clone();
    BaseInst->insertBefore(I);
    BaseInst->setName(getMangledName(I));
    // Add metadata marking this as a base value
    BaseInst->setMetadata("is_base_value", MDNode::get(I->getContext(), {}));
    States[I] = BDVState(I, BDVState::Conflict, BaseInst);
  }

#ifndef NDEBUG
  VerifyStates();
#endif

  // Returns a instruction which produces the base pointer for a given
  // instruction.  The instruction is assumed to be an input to one of the BDVs
  // seen in the inference algorithm above.  As such, we must either already
  // know it's base defining value is a base, or have inserted a new
  // instruction to propagate the base of it's BDV and have entered that newly
  // introduced instruction into the state table.  In either case, we are
  // assured to be able to determine an instruction which produces it's base
  // pointer.
  auto getBaseForInput = [&](Value *Input, Instruction *InsertPt) {
    Value *BDV = findBaseOrBDV(Input, Cache);
    Value *Base = nullptr;
    if (!States.count(BDV)) {
      assert(areBothVectorOrScalar(BDV, Input));
      Base = BDV;
    } else {
      // Either conflict or base.
      assert(States.count(BDV));
      Base = States[BDV].getBaseValue();
    }
    assert(Base && "Can't be null");
    // The cast is needed since base traversal may strip away bitcasts
    if (Base->getType() != Input->getType() && InsertPt)
      Base = new BitCastInst(Base, Input->getType(), "cast", InsertPt);
    return Base;
  };

  // Fixup all the inputs of the new PHIs.  Visit order needs to be
  // deterministic and predictable because we're naming newly created
  // instructions.
  for (auto Pair : States) {
    Instruction *BDV = cast<Instruction>(Pair.first);
    BDVState State = Pair.second;

    // Only values that do not have known bases or those that have differing
    // type (scalar versus vector) from a possible known base should be in the
    // lattice.
    assert((!isKnownBaseResult(BDV) ||
            !areBothVectorOrScalar(BDV, State.getBaseValue())) &&
           "why did it get added?");
    assert(!State.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!State.isConflict())
      continue;

    if (PHINode *BasePHI = dyn_cast<PHINode>(State.getBaseValue())) {
      PHINode *PN = cast<PHINode>(BDV);
      const unsigned NumPHIValues = PN->getNumIncomingValues();

      // The IR verifier requires phi nodes with multiple entries from the
      // same basic block to have the same incoming value for each of those
      // entries.  Since we're inserting bitcasts in the loop, make sure we
      // do so at least once per incoming block.
      DenseMap<BasicBlock *, Value*> BlockToValue;
      for (unsigned i = 0; i < NumPHIValues; i++) {
        Value *InVal = PN->getIncomingValue(i);
        BasicBlock *InBB = PN->getIncomingBlock(i);
        if (!BlockToValue.count(InBB))
          BlockToValue[InBB] = getBaseForInput(InVal, InBB->getTerminator());
        else {
#ifndef NDEBUG
          Value *OldBase = BlockToValue[InBB];
          Value *Base = getBaseForInput(InVal, nullptr);
          // In essence this assert states: the only way two values
          // incoming from the same basic block may be different is by
          // being different bitcasts of the same value.  A cleanup
          // that remains TODO is changing findBaseOrBDV to return an
          // llvm::Value of the correct type (and still remain pure).
          // This will remove the need to add bitcasts.
          assert(Base->stripPointerCasts() == OldBase->stripPointerCasts() &&
                 "Sanity -- findBaseOrBDV should be pure!");
#endif
        }
        Value *Base = BlockToValue[InBB];
        BasePHI->setIncomingValue(i, Base);
      }
    } else if (SelectInst *BaseSI =
                   dyn_cast<SelectInst>(State.getBaseValue())) {
      SelectInst *SI = cast<SelectInst>(BDV);

      // Find the instruction which produces the base for each input.
      // We may need to insert a bitcast.
      BaseSI->setTrueValue(getBaseForInput(SI->getTrueValue(), BaseSI));
      BaseSI->setFalseValue(getBaseForInput(SI->getFalseValue(), BaseSI));
    } else if (auto *BaseEE =
                   dyn_cast<ExtractElementInst>(State.getBaseValue())) {
      Value *InVal = cast<ExtractElementInst>(BDV)->getVectorOperand();
      // Find the instruction which produces the base for each input.  We may
      // need to insert a bitcast.
      BaseEE->setOperand(0, getBaseForInput(InVal, BaseEE));
    } else if (auto *BaseIE = dyn_cast<InsertElementInst>(State.getBaseValue())){
      auto *BdvIE = cast<InsertElementInst>(BDV);
      auto UpdateOperand = [&](int OperandIdx) {
        Value *InVal = BdvIE->getOperand(OperandIdx);
        Value *Base = getBaseForInput(InVal, BaseIE);
        BaseIE->setOperand(OperandIdx, Base);
      };
      UpdateOperand(0); // vector operand
      UpdateOperand(1); // scalar operand
    } else {
      auto *BaseSV = cast<ShuffleVectorInst>(State.getBaseValue());
      auto *BdvSV = cast<ShuffleVectorInst>(BDV);
      auto UpdateOperand = [&](int OperandIdx) {
        Value *InVal = BdvSV->getOperand(OperandIdx);
        Value *Base = getBaseForInput(InVal, BaseSV);
        BaseSV->setOperand(OperandIdx, Base);
      };
      UpdateOperand(0); // vector operand
      if (!BdvSV->isZeroEltSplat())
        UpdateOperand(1); // vector operand
      else {
        // Never read, so just use undef
        Value *InVal = BdvSV->getOperand(1);
        BaseSV->setOperand(1, UndefValue::get(InVal->getType()));
      }
    }
  }

#ifndef NDEBUG
  VerifyStates();
#endif

  // Cache all of our results so we can cheaply reuse them
  // NOTE: This is actually two caches: one of the base defining value
  // relation and one of the base pointer relation!  FIXME
  for (auto Pair : States) {
    auto *BDV = Pair.first;
    Value *Base = Pair.second.getBaseValue();
    assert(BDV && Base);
    // Only values that do not have known bases or those that have differing
    // type (scalar versus vector) from a possible known base should be in the
    // lattice.
    assert((!isKnownBaseResult(BDV) || !areBothVectorOrScalar(BDV, Base)) &&
           "why did it get added?");

    LLVM_DEBUG(
        dbgs() << "Updating base value cache"
               << " for: " << BDV->getName() << " from: "
               << (Cache.count(BDV) ? Cache[BDV]->getName().str() : "none")
               << " to: " << Base->getName() << "\n");

    Cache[BDV] = Base;
  }
  assert(Cache.count(Def));
  return Cache[Def];
}

// For a set of live pointers (base and/or derived), identify the base
// pointer of the object which they are derived from.  This routine will
// mutate the IR graph as needed to make the 'base' pointer live at the
// definition site of 'derived'.  This ensures that any use of 'derived' can
// also use 'base'.  This may involve the insertion of a number of
// additional PHI nodes.
//
// preconditions: live is a set of pointer type Values
//
// side effects: may insert PHI nodes into the existing CFG, will preserve
// CFG, will not remove or mutate any existing nodes
//
// post condition: PointerToBase contains one (derived, base) pair for every
// pointer in live.  Note that derived can be equal to base if the original
// pointer was a base pointer.
static void
findBasePointers(const StatepointLiveSetTy &live,
                 MapVector<Value *, Value *> &PointerToBase,
                 DominatorTree *DT, DefiningValueMapTy &DVCache) {
  for (Value *ptr : live) {
    Value *base = findBasePointer(ptr, DVCache);
    assert(base && "failed to find base pointer");
    PointerToBase[ptr] = base;
    assert((!isa<Instruction>(base) || !isa<Instruction>(ptr) ||
            DT->dominates(cast<Instruction>(base)->getParent(),
                          cast<Instruction>(ptr)->getParent())) &&
           "The base we found better dominate the derived pointer");
  }
}

/// Find the required based pointers (and adjust the live set) for the given
/// parse point.
static void findBasePointers(DominatorTree &DT, DefiningValueMapTy &DVCache,
                             CallBase *Call,
                             PartiallyConstructedSafepointRecord &result) {
  MapVector<Value *, Value *> PointerToBase;
  StatepointLiveSetTy PotentiallyDerivedPointers = result.LiveSet;
  // We assume that all pointers passed to deopt are base pointers; as an
  // optimization, we can use this to avoid seperately materializing the base
  // pointer graph.  This is only relevant since we're very conservative about
  // generating new conflict nodes during base pointer insertion.  If we were
  // smarter there, this would be irrelevant.
  if (auto Opt = Call->getOperandBundle(LLVMContext::OB_deopt))
    for (Value *V : Opt->Inputs) {
      if (!PotentiallyDerivedPointers.count(V))
        continue;
      PotentiallyDerivedPointers.remove(V);
      PointerToBase[V] = V;
    }
  findBasePointers(PotentiallyDerivedPointers, PointerToBase, &DT, DVCache);

  if (PrintBasePointers) {
    errs() << "Base Pairs (w/o Relocation):\n";
    for (auto &Pair : PointerToBase) {
      errs() << " derived ";
      Pair.first->printAsOperand(errs(), false);
      errs() << " base ";
      Pair.second->printAsOperand(errs(), false);
      errs() << "\n";;
    }
  }

  result.PointerToBase = PointerToBase;
}

/// Given an updated version of the dataflow liveness results, update the
/// liveset and base pointer maps for the call site CS.
static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  CallBase *Call,
                                  PartiallyConstructedSafepointRecord &result);

static void recomputeLiveInValues(
    Function &F, DominatorTree &DT, ArrayRef<CallBase *> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  // TODO-PERF: reuse the original liveness, then simply run the dataflow
  // again.  The old values are still live and will help it stabilize quickly.
  GCPtrLivenessData RevisedLivenessData;
  computeLiveInValues(DT, F, RevisedLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    recomputeLiveInValues(RevisedLivenessData, toUpdate[i], info);
  }
}

// When inserting gc.relocate and gc.result calls, we need to ensure there are
// no uses of the original value / return value between the gc.statepoint and
// the gc.relocate / gc.result call.  One case which can arise is a phi node
// starting one of the successor blocks.  We also need to be able to insert the
// gc.relocates only on the path which goes through the statepoint.  We might
// need to split an edge to make this possible.
static BasicBlock *
normalizeForInvokeSafepoint(BasicBlock *BB, BasicBlock *InvokeParent,
                            DominatorTree &DT) {
  BasicBlock *Ret = BB;
  if (!BB->getUniquePredecessor())
    Ret = SplitBlockPredecessors(BB, InvokeParent, "", &DT);

  // Now that 'Ret' has unique predecessor we can safely remove all phi nodes
  // from it
  FoldSingleEntryPHINodes(Ret);
  assert(!isa<PHINode>(Ret->begin()) &&
         "All PHI nodes should have been removed!");

  // At this point, we can safely insert a gc.relocate or gc.result as the first
  // instruction in Ret if needed.
  return Ret;
}

// List of all function attributes which must be stripped when lowering from
// abstract machine model to physical machine model.  Essentially, these are
// all the effects a safepoint might have which we ignored in the abstract
// machine model for purposes of optimization.  We have to strip these on
// both function declarations and call sites.
static constexpr Attribute::AttrKind FnAttrsToStrip[] =
  {Attribute::ReadNone, Attribute::ReadOnly, Attribute::WriteOnly,
   Attribute::ArgMemOnly, Attribute::InaccessibleMemOnly,
   Attribute::InaccessibleMemOrArgMemOnly,
   Attribute::NoSync, Attribute::NoFree};

// List of all parameter and return attributes which must be stripped when
// lowering from the abstract machine model.  Note that we list attributes
// here which aren't valid as return attributes, that is okay.  There are
// also some additional attributes with arguments which are handled
// explicitly and are not in this list.
static constexpr Attribute::AttrKind ParamAttrsToStrip[] =
  {Attribute::ReadNone, Attribute::ReadOnly, Attribute::WriteOnly,
   Attribute::NoAlias, Attribute::NoFree};


// Create new attribute set containing only attributes which can be transferred
// from original call to the safepoint.
static AttributeList legalizeCallAttributes(LLVMContext &Ctx,
                                            AttributeList AL) {
  if (AL.isEmpty())
    return AL;

  // Remove the readonly, readnone, and statepoint function attributes.
  AttrBuilder FnAttrs = AL.getFnAttrs();
  for (auto Attr : FnAttrsToStrip)
    FnAttrs.removeAttribute(Attr);

  for (Attribute A : AL.getFnAttrs()) {
    if (isStatepointDirectiveAttr(A))
      FnAttrs.remove(A);
  }

  // Just skip parameter and return attributes for now
  return AttributeList::get(Ctx, AttributeList::FunctionIndex,
                            AttributeSet::get(Ctx, FnAttrs));
}

/// Helper function to place all gc relocates necessary for the given
/// statepoint.
/// Inputs:
///   liveVariables - list of variables to be relocated.
///   basePtrs - base pointers.
///   statepointToken - statepoint instruction to which relocates should be
///   bound.
///   Builder - Llvm IR builder to be used to construct new calls.
static void CreateGCRelocates(ArrayRef<Value *> LiveVariables,
                              ArrayRef<Value *> BasePtrs,
                              Instruction *StatepointToken,
                              IRBuilder<> &Builder) {
  if (LiveVariables.empty())
    return;

  auto FindIndex = [](ArrayRef<Value *> LiveVec, Value *Val) {
    auto ValIt = llvm::find(LiveVec, Val);
    assert(ValIt != LiveVec.end() && "Val not found in LiveVec!");
    size_t Index = std::distance(LiveVec.begin(), ValIt);
    assert(Index < LiveVec.size() && "Bug in std::find?");
    return Index;
  };
  Module *M = StatepointToken->getModule();

  // All gc_relocate are generated as i8 addrspace(1)* (or a vector type whose
  // element type is i8 addrspace(1)*). We originally generated unique
  // declarations for each pointer type, but this proved problematic because
  // the intrinsic mangling code is incomplete and fragile.  Since we're moving
  // towards a single unified pointer type anyways, we can just cast everything
  // to an i8* of the right address space.  A bitcast is added later to convert
  // gc_relocate to the actual value's type.
  auto getGCRelocateDecl = [&] (Type *Ty) {
    assert(isHandledGCPointerType(Ty));
    auto AS = Ty->getScalarType()->getPointerAddressSpace();
    Type *NewTy = Type::getInt8PtrTy(M->getContext(), AS);
    if (auto *VT = dyn_cast<VectorType>(Ty))
      NewTy = FixedVectorType::get(NewTy,
                                   cast<FixedVectorType>(VT)->getNumElements());
    return Intrinsic::getDeclaration(M, Intrinsic::experimental_gc_relocate,
                                     {NewTy});
  };

  // Lazily populated map from input types to the canonicalized form mentioned
  // in the comment above.  This should probably be cached somewhere more
  // broadly.
  DenseMap<Type *, Function *> TypeToDeclMap;

  for (unsigned i = 0; i < LiveVariables.size(); i++) {
    // Generate the gc.relocate call and save the result
    Value *BaseIdx = Builder.getInt32(FindIndex(LiveVariables, BasePtrs[i]));
    Value *LiveIdx = Builder.getInt32(i);

    Type *Ty = LiveVariables[i]->getType();
    if (!TypeToDeclMap.count(Ty))
      TypeToDeclMap[Ty] = getGCRelocateDecl(Ty);
    Function *GCRelocateDecl = TypeToDeclMap[Ty];

    // only specify a debug name if we can give a useful one
    CallInst *Reloc = Builder.CreateCall(
        GCRelocateDecl, {StatepointToken, BaseIdx, LiveIdx},
        suffixed_name_or(LiveVariables[i], ".relocated", ""));
    // Trick CodeGen into thinking there are lots of free registers at this
    // fake call.
    Reloc->setCallingConv(CallingConv::Cold);
  }
}

namespace {

/// This struct is used to defer RAUWs and `eraseFromParent` s.  Using this
/// avoids having to worry about keeping around dangling pointers to Values.
class DeferredReplacement {
  AssertingVH<Instruction> Old;
  AssertingVH<Instruction> New;
  bool IsDeoptimize = false;

  DeferredReplacement() = default;

public:
  static DeferredReplacement createRAUW(Instruction *Old, Instruction *New) {
    assert(Old != New && Old && New &&
           "Cannot RAUW equal values or to / from null!");

    DeferredReplacement D;
    D.Old = Old;
    D.New = New;
    return D;
  }

  static DeferredReplacement createDelete(Instruction *ToErase) {
    DeferredReplacement D;
    D.Old = ToErase;
    return D;
  }

  static DeferredReplacement createDeoptimizeReplacement(Instruction *Old) {
#ifndef NDEBUG
    auto *F = cast<CallInst>(Old)->getCalledFunction();
    assert(F && F->getIntrinsicID() == Intrinsic::experimental_deoptimize &&
           "Only way to construct a deoptimize deferred replacement");
#endif
    DeferredReplacement D;
    D.Old = Old;
    D.IsDeoptimize = true;
    return D;
  }

  /// Does the task represented by this instance.
  void doReplacement() {
    Instruction *OldI = Old;
    Instruction *NewI = New;

    assert(OldI != NewI && "Disallowed at construction?!");
    assert((!IsDeoptimize || !New) &&
           "Deoptimize intrinsics are not replaced!");

    Old = nullptr;
    New = nullptr;

    if (NewI)
      OldI->replaceAllUsesWith(NewI);

    if (IsDeoptimize) {
      // Note: we've inserted instructions, so the call to llvm.deoptimize may
      // not necessarily be followed by the matching return.
      auto *RI = cast<ReturnInst>(OldI->getParent()->getTerminator());
      new UnreachableInst(RI->getContext(), RI);
      RI->eraseFromParent();
    }

    OldI->eraseFromParent();
  }
};

} // end anonymous namespace

static StringRef getDeoptLowering(CallBase *Call) {
  const char *DeoptLowering = "deopt-lowering";
  if (Call->hasFnAttr(DeoptLowering)) {
    // FIXME: Calls have a *really* confusing interface around attributes
    // with values.
    const AttributeList &CSAS = Call->getAttributes();
    if (CSAS.hasFnAttr(DeoptLowering))
      return CSAS.getFnAttr(DeoptLowering).getValueAsString();
    Function *F = Call->getCalledFunction();
    assert(F && F->hasFnAttribute(DeoptLowering));
    return F->getFnAttribute(DeoptLowering).getValueAsString();
  }
  return "live-through";
}

static void
makeStatepointExplicitImpl(CallBase *Call, /* to replace */
                           const SmallVectorImpl<Value *> &BasePtrs,
                           const SmallVectorImpl<Value *> &LiveVariables,
                           PartiallyConstructedSafepointRecord &Result,
                           std::vector<DeferredReplacement> &Replacements) {
  assert(BasePtrs.size() == LiveVariables.size());

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  IRBuilder<> Builder(Call);

  ArrayRef<Value *> GCArgs(LiveVariables);
  uint64_t StatepointID = StatepointDirectives::DefaultStatepointID;
  uint32_t NumPatchBytes = 0;
  uint32_t Flags = uint32_t(StatepointFlags::None);

  SmallVector<Value *, 8> CallArgs(Call->args());
  Optional<ArrayRef<Use>> DeoptArgs;
  if (auto Bundle = Call->getOperandBundle(LLVMContext::OB_deopt))
    DeoptArgs = Bundle->Inputs;
  Optional<ArrayRef<Use>> TransitionArgs;
  if (auto Bundle = Call->getOperandBundle(LLVMContext::OB_gc_transition)) {
    TransitionArgs = Bundle->Inputs;
    // TODO: This flag no longer serves a purpose and can be removed later
    Flags |= uint32_t(StatepointFlags::GCTransition);
  }

  // Instead of lowering calls to @llvm.experimental.deoptimize as normal calls
  // with a return value, we lower then as never returning calls to
  // __llvm_deoptimize that are followed by unreachable to get better codegen.
  bool IsDeoptimize = false;

  StatepointDirectives SD =
      parseStatepointDirectivesFromAttrs(Call->getAttributes());
  if (SD.NumPatchBytes)
    NumPatchBytes = *SD.NumPatchBytes;
  if (SD.StatepointID)
    StatepointID = *SD.StatepointID;

  // Pass through the requested lowering if any.  The default is live-through.
  StringRef DeoptLowering = getDeoptLowering(Call);
  if (DeoptLowering.equals("live-in"))
    Flags |= uint32_t(StatepointFlags::DeoptLiveIn);
  else {
    assert(DeoptLowering.equals("live-through") && "Unsupported value!");
  }

  Value *CallTarget = Call->getCalledOperand();
  if (Function *F = dyn_cast<Function>(CallTarget)) {
    auto IID = F->getIntrinsicID();
    if (IID == Intrinsic::experimental_deoptimize) {
      // Calls to llvm.experimental.deoptimize are lowered to calls to the
      // __llvm_deoptimize symbol.  We want to resolve this now, since the
      // verifier does not allow taking the address of an intrinsic function.

      SmallVector<Type *, 8> DomainTy;
      for (Value *Arg : CallArgs)
        DomainTy.push_back(Arg->getType());
      auto *FTy = FunctionType::get(Type::getVoidTy(F->getContext()), DomainTy,
                                    /* isVarArg = */ false);

      // Note: CallTarget can be a bitcast instruction of a symbol if there are
      // calls to @llvm.experimental.deoptimize with different argument types in
      // the same module.  This is fine -- we assume the frontend knew what it
      // was doing when generating this kind of IR.
      CallTarget = F->getParent()
                       ->getOrInsertFunction("__llvm_deoptimize", FTy)
                       .getCallee();

      IsDeoptimize = true;
    } else if (IID == Intrinsic::memcpy_element_unordered_atomic ||
               IID == Intrinsic::memmove_element_unordered_atomic) {
      // Unordered atomic memcpy and memmove intrinsics which are not explicitly
      // marked as "gc-leaf-function" should be lowered in a GC parseable way.
      // Specifically, these calls should be lowered to the
      // __llvm_{memcpy|memmove}_element_unordered_atomic_safepoint symbols.
      // Similarly to __llvm_deoptimize we want to resolve this now, since the
      // verifier does not allow taking the address of an intrinsic function.
      //
      // Moreover we need to shuffle the arguments for the call in order to
      // accommodate GC. The underlying source and destination objects might be
      // relocated during copy operation should the GC occur. To relocate the
      // derived source and destination pointers the implementation of the
      // intrinsic should know the corresponding base pointers.
      //
      // To make the base pointers available pass them explicitly as arguments:
      //   memcpy(dest_derived, source_derived, ...) =>
      //   memcpy(dest_base, dest_offset, source_base, source_offset, ...)
      auto &Context = Call->getContext();
      auto &DL = Call->getModule()->getDataLayout();
      auto GetBaseAndOffset = [&](Value *Derived) {
        assert(Result.PointerToBase.count(Derived));
        unsigned AddressSpace = Derived->getType()->getPointerAddressSpace();
        unsigned IntPtrSize = DL.getPointerSizeInBits(AddressSpace);
        Value *Base = Result.PointerToBase.find(Derived)->second;
        Value *Base_int = Builder.CreatePtrToInt(
            Base, Type::getIntNTy(Context, IntPtrSize));
        Value *Derived_int = Builder.CreatePtrToInt(
            Derived, Type::getIntNTy(Context, IntPtrSize));
        return std::make_pair(Base, Builder.CreateSub(Derived_int, Base_int));
      };

      auto *Dest = CallArgs[0];
      Value *DestBase, *DestOffset;
      std::tie(DestBase, DestOffset) = GetBaseAndOffset(Dest);

      auto *Source = CallArgs[1];
      Value *SourceBase, *SourceOffset;
      std::tie(SourceBase, SourceOffset) = GetBaseAndOffset(Source);

      auto *LengthInBytes = CallArgs[2];
      auto *ElementSizeCI = cast<ConstantInt>(CallArgs[3]);

      CallArgs.clear();
      CallArgs.push_back(DestBase);
      CallArgs.push_back(DestOffset);
      CallArgs.push_back(SourceBase);
      CallArgs.push_back(SourceOffset);
      CallArgs.push_back(LengthInBytes);

      SmallVector<Type *, 8> DomainTy;
      for (Value *Arg : CallArgs)
        DomainTy.push_back(Arg->getType());
      auto *FTy = FunctionType::get(Type::getVoidTy(F->getContext()), DomainTy,
                                    /* isVarArg = */ false);

      auto GetFunctionName = [](Intrinsic::ID IID, ConstantInt *ElementSizeCI) {
        uint64_t ElementSize = ElementSizeCI->getZExtValue();
        if (IID == Intrinsic::memcpy_element_unordered_atomic) {
          switch (ElementSize) {
          case 1:
            return "__llvm_memcpy_element_unordered_atomic_safepoint_1";
          case 2:
            return "__llvm_memcpy_element_unordered_atomic_safepoint_2";
          case 4:
            return "__llvm_memcpy_element_unordered_atomic_safepoint_4";
          case 8:
            return "__llvm_memcpy_element_unordered_atomic_safepoint_8";
          case 16:
            return "__llvm_memcpy_element_unordered_atomic_safepoint_16";
          default:
            llvm_unreachable("unexpected element size!");
          }
        }
        assert(IID == Intrinsic::memmove_element_unordered_atomic);
        switch (ElementSize) {
        case 1:
          return "__llvm_memmove_element_unordered_atomic_safepoint_1";
        case 2:
          return "__llvm_memmove_element_unordered_atomic_safepoint_2";
        case 4:
          return "__llvm_memmove_element_unordered_atomic_safepoint_4";
        case 8:
          return "__llvm_memmove_element_unordered_atomic_safepoint_8";
        case 16:
          return "__llvm_memmove_element_unordered_atomic_safepoint_16";
        default:
          llvm_unreachable("unexpected element size!");
        }
      };

      CallTarget =
          F->getParent()
              ->getOrInsertFunction(GetFunctionName(IID, ElementSizeCI), FTy)
              .getCallee();
    }
  }

  // Create the statepoint given all the arguments
  GCStatepointInst *Token = nullptr;
  if (auto *CI = dyn_cast<CallInst>(Call)) {
    CallInst *SPCall = Builder.CreateGCStatepointCall(
        StatepointID, NumPatchBytes, CallTarget, Flags, CallArgs,
        TransitionArgs, DeoptArgs, GCArgs, "safepoint_token");

    SPCall->setTailCallKind(CI->getTailCallKind());
    SPCall->setCallingConv(CI->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.  In case if we can handle this set of attributes -
    // set up function attrs directly on statepoint and return attrs later for
    // gc_result intrinsic.
    SPCall->setAttributes(
        legalizeCallAttributes(CI->getContext(), CI->getAttributes()));

    Token = cast<GCStatepointInst>(SPCall);

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    assert(CI->getNextNode() && "Not a terminator, must have next!");
    Builder.SetInsertPoint(CI->getNextNode());
    Builder.SetCurrentDebugLocation(CI->getNextNode()->getDebugLoc());
  } else {
    auto *II = cast<InvokeInst>(Call);

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    InvokeInst *SPInvoke = Builder.CreateGCStatepointInvoke(
        StatepointID, NumPatchBytes, CallTarget, II->getNormalDest(),
        II->getUnwindDest(), Flags, CallArgs, TransitionArgs, DeoptArgs, GCArgs,
        "statepoint_token");

    SPInvoke->setCallingConv(II->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.  In case if we can handle this set of attributes -
    // set up function attrs directly on statepoint and return attrs later for
    // gc_result intrinsic.
    SPInvoke->setAttributes(
        legalizeCallAttributes(II->getContext(), II->getAttributes()));

    Token = cast<GCStatepointInst>(SPInvoke);

    // Generate gc relocates in exceptional path
    BasicBlock *UnwindBlock = II->getUnwindDest();
    assert(!isa<PHINode>(UnwindBlock->begin()) &&
           UnwindBlock->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Builder.SetInsertPoint(&*UnwindBlock->getFirstInsertionPt());
    Builder.SetCurrentDebugLocation(II->getDebugLoc());

    // Attach exceptional gc relocates to the landingpad.
    Instruction *ExceptionalToken = UnwindBlock->getLandingPadInst();
    Result.UnwindToken = ExceptionalToken;

    CreateGCRelocates(LiveVariables, BasePtrs, ExceptionalToken, Builder);

    // Generate gc relocates and returns for normal block
    BasicBlock *NormalDest = II->getNormalDest();
    assert(!isa<PHINode>(NormalDest->begin()) &&
           NormalDest->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Builder.SetInsertPoint(&*NormalDest->getFirstInsertionPt());

    // gc relocates will be generated later as if it were regular call
    // statepoint
  }
  assert(Token && "Should be set in one of the above branches!");

  if (IsDeoptimize) {
    // If we're wrapping an @llvm.experimental.deoptimize in a statepoint, we
    // transform the tail-call like structure to a call to a void function
    // followed by unreachable to get better codegen.
    Replacements.push_back(
        DeferredReplacement::createDeoptimizeReplacement(Call));
  } else {
    Token->setName("statepoint_token");
    if (!Call->getType()->isVoidTy() && !Call->use_empty()) {
      StringRef Name = Call->hasName() ? Call->getName() : "";
      CallInst *GCResult = Builder.CreateGCResult(Token, Call->getType(), Name);
      GCResult->setAttributes(
          AttributeList::get(GCResult->getContext(), AttributeList::ReturnIndex,
                             Call->getAttributes().getRetAttrs()));

      // We cannot RAUW or delete CS.getInstruction() because it could be in the
      // live set of some other safepoint, in which case that safepoint's
      // PartiallyConstructedSafepointRecord will hold a raw pointer to this
      // llvm::Instruction.  Instead, we defer the replacement and deletion to
      // after the live sets have been made explicit in the IR, and we no longer
      // have raw pointers to worry about.
      Replacements.emplace_back(
          DeferredReplacement::createRAUW(Call, GCResult));
    } else {
      Replacements.emplace_back(DeferredReplacement::createDelete(Call));
    }
  }

  Result.StatepointToken = Token;

  // Second, create a gc.relocate for every live variable
  CreateGCRelocates(LiveVariables, BasePtrs, Token, Builder);
}

// Replace an existing gc.statepoint with a new one and a set of gc.relocates
// which make the relocations happening at this safepoint explicit.
//
// WARNING: Does not do any fixup to adjust users of the original live
// values.  That's the callers responsibility.
static void
makeStatepointExplicit(DominatorTree &DT, CallBase *Call,
                       PartiallyConstructedSafepointRecord &Result,
                       std::vector<DeferredReplacement> &Replacements) {
  const auto &LiveSet = Result.LiveSet;
  const auto &PointerToBase = Result.PointerToBase;

  // Convert to vector for efficient cross referencing.
  SmallVector<Value *, 64> BaseVec, LiveVec;
  LiveVec.reserve(LiveSet.size());
  BaseVec.reserve(LiveSet.size());
  for (Value *L : LiveSet) {
    LiveVec.push_back(L);
    assert(PointerToBase.count(L));
    Value *Base = PointerToBase.find(L)->second;
    BaseVec.push_back(Base);
  }
  assert(LiveVec.size() == BaseVec.size());

  // Do the actual rewriting and delete the old statepoint
  makeStatepointExplicitImpl(Call, BaseVec, LiveVec, Result, Replacements);
}

// Helper function for the relocationViaAlloca.
//
// It receives iterator to the statepoint gc relocates and emits a store to the
// assigned location (via allocaMap) for the each one of them.  It adds the
// visited values into the visitedLiveValues set, which we will later use them
// for sanity checking.
static void
insertRelocationStores(iterator_range<Value::user_iterator> GCRelocs,
                       DenseMap<Value *, AllocaInst *> &AllocaMap,
                       DenseSet<Value *> &VisitedLiveValues) {
  for (User *U : GCRelocs) {
    GCRelocateInst *Relocate = dyn_cast<GCRelocateInst>(U);
    if (!Relocate)
      continue;

    Value *OriginalValue = Relocate->getDerivedPtr();
    assert(AllocaMap.count(OriginalValue));
    Value *Alloca = AllocaMap[OriginalValue];

    // Emit store into the related alloca
    // All gc_relocates are i8 addrspace(1)* typed, and it must be bitcasted to
    // the correct type according to alloca.
    assert(Relocate->getNextNode() &&
           "Should always have one since it's not a terminator");
    IRBuilder<> Builder(Relocate->getNextNode());
    Value *CastedRelocatedValue =
      Builder.CreateBitCast(Relocate,
                            cast<AllocaInst>(Alloca)->getAllocatedType(),
                            suffixed_name_or(Relocate, ".casted", ""));

    new StoreInst(CastedRelocatedValue, Alloca,
                  cast<Instruction>(CastedRelocatedValue)->getNextNode());

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

// Helper function for the "relocationViaAlloca". Similar to the
// "insertRelocationStores" but works for rematerialized values.
static void insertRematerializationStores(
    const RematerializedValueMapTy &RematerializedValues,
    DenseMap<Value *, AllocaInst *> &AllocaMap,
    DenseSet<Value *> &VisitedLiveValues) {
  for (auto RematerializedValuePair: RematerializedValues) {
    Instruction *RematerializedValue = RematerializedValuePair.first;
    Value *OriginalValue = RematerializedValuePair.second;

    assert(AllocaMap.count(OriginalValue) &&
           "Can not find alloca for rematerialized value");
    Value *Alloca = AllocaMap[OriginalValue];

    new StoreInst(RematerializedValue, Alloca,
                  RematerializedValue->getNextNode());

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

/// Do all the relocation update via allocas and mem2reg
static void relocationViaAlloca(
    Function &F, DominatorTree &DT, ArrayRef<Value *> Live,
    ArrayRef<PartiallyConstructedSafepointRecord> Records) {
#ifndef NDEBUG
  // record initial number of (static) allocas; we'll check we have the same
  // number when we get done.
  int InitialAllocaNum = 0;
  for (Instruction &I : F.getEntryBlock())
    if (isa<AllocaInst>(I))
      InitialAllocaNum++;
#endif

  // TODO-PERF: change data structures, reserve
  DenseMap<Value *, AllocaInst *> AllocaMap;
  SmallVector<AllocaInst *, 200> PromotableAllocas;
  // Used later to chack that we have enough allocas to store all values
  std::size_t NumRematerializedValues = 0;
  PromotableAllocas.reserve(Live.size());

  // Emit alloca for "LiveValue" and record it in "allocaMap" and
  // "PromotableAllocas"
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto emitAllocaFor = [&](Value *LiveValue) {
    AllocaInst *Alloca = new AllocaInst(LiveValue->getType(),
                                        DL.getAllocaAddrSpace(), "",
                                        F.getEntryBlock().getFirstNonPHI());
    AllocaMap[LiveValue] = Alloca;
    PromotableAllocas.push_back(Alloca);
  };

  // Emit alloca for each live gc pointer
  for (Value *V : Live)
    emitAllocaFor(V);

  // Emit allocas for rematerialized values
  for (const auto &Info : Records)
    for (auto RematerializedValuePair : Info.RematerializedValues) {
      Value *OriginalValue = RematerializedValuePair.second;
      if (AllocaMap.count(OriginalValue) != 0)
        continue;

      emitAllocaFor(OriginalValue);
      ++NumRematerializedValues;
    }

  // The next two loops are part of the same conceptual operation.  We need to
  // insert a store to the alloca after the original def and at each
  // redefinition.  We need to insert a load before each use.  These are split
  // into distinct loops for performance reasons.

  // Update gc pointer after each statepoint: either store a relocated value or
  // null (if no relocated value was found for this gc pointer and it is not a
  // gc_result).  This must happen before we update the statepoint with load of
  // alloca otherwise we lose the link between statepoint and old def.
  for (const auto &Info : Records) {
    Value *Statepoint = Info.StatepointToken;

    // This will be used for consistency check
    DenseSet<Value *> VisitedLiveValues;

    // Insert stores for normal statepoint gc relocates
    insertRelocationStores(Statepoint->users(), AllocaMap, VisitedLiveValues);

    // In case if it was invoke statepoint
    // we will insert stores for exceptional path gc relocates.
    if (isa<InvokeInst>(Statepoint)) {
      insertRelocationStores(Info.UnwindToken->users(), AllocaMap,
                             VisitedLiveValues);
    }

    // Do similar thing with rematerialized values
    insertRematerializationStores(Info.RematerializedValues, AllocaMap,
                                  VisitedLiveValues);

    if (ClobberNonLive) {
      // As a debugging aid, pretend that an unrelocated pointer becomes null at
      // the gc.statepoint.  This will turn some subtle GC problems into
      // slightly easier to debug SEGVs.  Note that on large IR files with
      // lots of gc.statepoints this is extremely costly both memory and time
      // wise.
      SmallVector<AllocaInst *, 64> ToClobber;
      for (auto Pair : AllocaMap) {
        Value *Def = Pair.first;
        AllocaInst *Alloca = Pair.second;

        // This value was relocated
        if (VisitedLiveValues.count(Def)) {
          continue;
        }
        ToClobber.push_back(Alloca);
      }

      auto InsertClobbersAt = [&](Instruction *IP) {
        for (auto *AI : ToClobber) {
          auto PT = cast<PointerType>(AI->getAllocatedType());
          Constant *CPN = ConstantPointerNull::get(PT);
          new StoreInst(CPN, AI, IP);
        }
      };

      // Insert the clobbering stores.  These may get intermixed with the
      // gc.results and gc.relocates, but that's fine.
      if (auto II = dyn_cast<InvokeInst>(Statepoint)) {
        InsertClobbersAt(&*II->getNormalDest()->getFirstInsertionPt());
        InsertClobbersAt(&*II->getUnwindDest()->getFirstInsertionPt());
      } else {
        InsertClobbersAt(cast<Instruction>(Statepoint)->getNextNode());
      }
    }
  }

  // Update use with load allocas and add store for gc_relocated.
  for (auto Pair : AllocaMap) {
    Value *Def = Pair.first;
    AllocaInst *Alloca = Pair.second;

    // We pre-record the uses of allocas so that we dont have to worry about
    // later update that changes the user information..

    SmallVector<Instruction *, 20> Uses;
    // PERF: trade a linear scan for repeated reallocation
    Uses.reserve(Def->getNumUses());
    for (User *U : Def->users()) {
      if (!isa<ConstantExpr>(U)) {
        // If the def has a ConstantExpr use, then the def is either a
        // ConstantExpr use itself or null.  In either case
        // (recursively in the first, directly in the second), the oop
        // it is ultimately dependent on is null and this particular
        // use does not need to be fixed up.
        Uses.push_back(cast<Instruction>(U));
      }
    }

    llvm::sort(Uses);
    auto Last = std::unique(Uses.begin(), Uses.end());
    Uses.erase(Last, Uses.end());

    for (Instruction *Use : Uses) {
      if (isa<PHINode>(Use)) {
        PHINode *Phi = cast<PHINode>(Use);
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (Def == Phi->getIncomingValue(i)) {
            LoadInst *Load =
                new LoadInst(Alloca->getAllocatedType(), Alloca, "",
                             Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
        }
      } else {
        LoadInst *Load =
            new LoadInst(Alloca->getAllocatedType(), Alloca, "", Use);
        Use->replaceUsesOfWith(Def, Load);
      }
    }

    // Emit store for the initial gc value.  Store must be inserted after load,
    // otherwise store will be in alloca's use list and an extra load will be
    // inserted before it.
    StoreInst *Store = new StoreInst(Def, Alloca, /*volatile*/ false,
                                     DL.getABITypeAlign(Def->getType()));
    if (Instruction *Inst = dyn_cast<Instruction>(Def)) {
      if (InvokeInst *Invoke = dyn_cast<InvokeInst>(Inst)) {
        // InvokeInst is a terminator so the store need to be inserted into its
        // normal destination block.
        BasicBlock *NormalDest = Invoke->getNormalDest();
        Store->insertBefore(NormalDest->getFirstNonPHI());
      } else {
        assert(!Inst->isTerminator() &&
               "The only terminator that can produce a value is "
               "InvokeInst which is handled above.");
        Store->insertAfter(Inst);
      }
    } else {
      assert(isa<Argument>(Def));
      Store->insertAfter(cast<Instruction>(Alloca));
    }
  }

  assert(PromotableAllocas.size() == Live.size() + NumRematerializedValues &&
         "we must have the same allocas with lives");
  if (!PromotableAllocas.empty()) {
    // Apply mem2reg to promote alloca to SSA
    PromoteMemToReg(PromotableAllocas, DT);
  }

#ifndef NDEBUG
  for (auto &I : F.getEntryBlock())
    if (isa<AllocaInst>(I))
      InitialAllocaNum--;
  assert(InitialAllocaNum == 0 && "We must not introduce any extra allocas");
#endif
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(SmallVectorImpl<T> &Vec) {
  SmallSet<T, 8> Seen;
  erase_if(Vec, [&](const T &V) { return !Seen.insert(V).second; });
}

/// Insert holders so that each Value is obviously live through the entire
/// lifetime of the call.
static void insertUseHolderAfter(CallBase *Call, const ArrayRef<Value *> Values,
                                 SmallVectorImpl<CallInst *> &Holders) {
  if (Values.empty())
    // No values to hold live, might as well not insert the empty holder
    return;

  Module *M = Call->getModule();
  // Use a dummy vararg function to actually hold the values live
  FunctionCallee Func = M->getOrInsertFunction(
      "__tmp_use", FunctionType::get(Type::getVoidTy(M->getContext()), true));
  if (isa<CallInst>(Call)) {
    // For call safepoints insert dummy calls right after safepoint
    Holders.push_back(
        CallInst::Create(Func, Values, "", &*++Call->getIterator()));
    return;
  }
  // For invoke safepooints insert dummy calls both in normal and
  // exceptional destination blocks
  auto *II = cast<InvokeInst>(Call);
  Holders.push_back(CallInst::Create(
      Func, Values, "", &*II->getNormalDest()->getFirstInsertionPt()));
  Holders.push_back(CallInst::Create(
      Func, Values, "", &*II->getUnwindDest()->getFirstInsertionPt()));
}

static void findLiveReferences(
    Function &F, DominatorTree &DT, ArrayRef<CallBase *> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  GCPtrLivenessData OriginalLivenessData;
  computeLiveInValues(DT, F, OriginalLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    analyzeParsePointLiveness(DT, OriginalLivenessData, toUpdate[i], info);
  }
}

// Helper function for the "rematerializeLiveValues". It walks use chain
// starting from the "CurrentValue" until it reaches the root of the chain, i.e.
// the base or a value it cannot process. Only "simple" values are processed
// (currently it is GEP's and casts). The returned root is  examined by the
// callers of findRematerializableChainToBasePointer.  Fills "ChainToBase" array
// with all visited values.
static Value* findRematerializableChainToBasePointer(
  SmallVectorImpl<Instruction*> &ChainToBase,
  Value *CurrentValue) {
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurrentValue)) {
    ChainToBase.push_back(GEP);
    return findRematerializableChainToBasePointer(ChainToBase,
                                                  GEP->getPointerOperand());
  }

  if (CastInst *CI = dyn_cast<CastInst>(CurrentValue)) {
    if (!CI->isNoopCast(CI->getModule()->getDataLayout()))
      return CI;

    ChainToBase.push_back(CI);
    return findRematerializableChainToBasePointer(ChainToBase,
                                                  CI->getOperand(0));
  }

  // We have reached the root of the chain, which is either equal to the base or
  // is the first unsupported value along the use chain.
  return CurrentValue;
}

// Helper function for the "rematerializeLiveValues". Compute cost of the use
// chain we are going to rematerialize.
static InstructionCost
chainToBasePointerCost(SmallVectorImpl<Instruction *> &Chain,
                       TargetTransformInfo &TTI) {
  InstructionCost Cost = 0;

  for (Instruction *Instr : Chain) {
    if (CastInst *CI = dyn_cast<CastInst>(Instr)) {
      assert(CI->isNoopCast(CI->getModule()->getDataLayout()) &&
             "non noop cast is found during rematerialization");

      Type *SrcTy = CI->getOperand(0)->getType();
      Cost += TTI.getCastInstrCost(CI->getOpcode(), CI->getType(), SrcTy,
                                   TTI::getCastContextHint(CI),
                                   TargetTransformInfo::TCK_SizeAndLatency, CI);

    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Instr)) {
      // Cost of the address calculation
      Type *ValTy = GEP->getSourceElementType();
      Cost += TTI.getAddressComputationCost(ValTy);

      // And cost of the GEP itself
      // TODO: Use TTI->getGEPCost here (it exists, but appears to be not
      //       allowed for the external usage)
      if (!GEP->hasAllConstantIndices())
        Cost += 2;

    } else {
      llvm_unreachable("unsupported instruction type during rematerialization");
    }
  }

  return Cost;
}

static bool AreEquivalentPhiNodes(PHINode &OrigRootPhi, PHINode &AlternateRootPhi) {
  unsigned PhiNum = OrigRootPhi.getNumIncomingValues();
  if (PhiNum != AlternateRootPhi.getNumIncomingValues() ||
      OrigRootPhi.getParent() != AlternateRootPhi.getParent())
    return false;
  // Map of incoming values and their corresponding basic blocks of
  // OrigRootPhi.
  SmallDenseMap<Value *, BasicBlock *, 8> CurrentIncomingValues;
  for (unsigned i = 0; i < PhiNum; i++)
    CurrentIncomingValues[OrigRootPhi.getIncomingValue(i)] =
        OrigRootPhi.getIncomingBlock(i);

  // Both current and base PHIs should have same incoming values and
  // the same basic blocks corresponding to the incoming values.
  for (unsigned i = 0; i < PhiNum; i++) {
    auto CIVI =
        CurrentIncomingValues.find(AlternateRootPhi.getIncomingValue(i));
    if (CIVI == CurrentIncomingValues.end())
      return false;
    BasicBlock *CurrentIncomingBB = CIVI->second;
    if (CurrentIncomingBB != AlternateRootPhi.getIncomingBlock(i))
      return false;
  }
  return true;
}

// From the statepoint live set pick values that are cheaper to recompute then
// to relocate. Remove this values from the live set, rematerialize them after
// statepoint and record them in "Info" structure. Note that similar to
// relocated values we don't do any user adjustments here.
static void rematerializeLiveValues(CallBase *Call,
                                    PartiallyConstructedSafepointRecord &Info,
                                    TargetTransformInfo &TTI) {
  const unsigned int ChainLengthThreshold = 10;

  // Record values we are going to delete from this statepoint live set.
  // We can not di this in following loop due to iterator invalidation.
  SmallVector<Value *, 32> LiveValuesToBeDeleted;

  for (Value *LiveValue: Info.LiveSet) {
    // For each live pointer find its defining chain
    SmallVector<Instruction *, 3> ChainToBase;
    assert(Info.PointerToBase.count(LiveValue));
    Value *RootOfChain =
      findRematerializableChainToBasePointer(ChainToBase,
                                             LiveValue);

    // Nothing to do, or chain is too long
    if ( ChainToBase.size() == 0 ||
        ChainToBase.size() > ChainLengthThreshold)
      continue;

    // Handle the scenario where the RootOfChain is not equal to the
    // Base Value, but they are essentially the same phi values.
    if (RootOfChain != Info.PointerToBase[LiveValue]) {
      PHINode *OrigRootPhi = dyn_cast<PHINode>(RootOfChain);
      PHINode *AlternateRootPhi = dyn_cast<PHINode>(Info.PointerToBase[LiveValue]);
      if (!OrigRootPhi || !AlternateRootPhi)
        continue;
      // PHI nodes that have the same incoming values, and belonging to the same
      // basic blocks are essentially the same SSA value.  When the original phi
      // has incoming values with different base pointers, the original phi is
      // marked as conflict, and an additional `AlternateRootPhi` with the same
      // incoming values get generated by the findBasePointer function. We need
      // to identify the newly generated AlternateRootPhi (.base version of phi)
      // and RootOfChain (the original phi node itself) are the same, so that we
      // can rematerialize the gep and casts. This is a workaround for the
      // deficiency in the findBasePointer algorithm.
      if (!AreEquivalentPhiNodes(*OrigRootPhi, *AlternateRootPhi))
        continue;
      // Now that the phi nodes are proved to be the same, assert that
      // findBasePointer's newly generated AlternateRootPhi is present in the
      // liveset of the call.
      assert(Info.LiveSet.count(AlternateRootPhi));
    }
    // Compute cost of this chain
    InstructionCost Cost = chainToBasePointerCost(ChainToBase, TTI);
    // TODO: We can also account for cases when we will be able to remove some
    //       of the rematerialized values by later optimization passes. I.e if
    //       we rematerialized several intersecting chains. Or if original values
    //       don't have any uses besides this statepoint.

    // For invokes we need to rematerialize each chain twice - for normal and
    // for unwind basic blocks. Model this by multiplying cost by two.
    if (isa<InvokeInst>(Call)) {
      Cost *= 2;
    }
    // If it's too expensive - skip it
    if (Cost >= RematerializationThreshold)
      continue;

    // Remove value from the live set
    LiveValuesToBeDeleted.push_back(LiveValue);

    // Clone instructions and record them inside "Info" structure

    // Walk backwards to visit top-most instructions first
    std::reverse(ChainToBase.begin(), ChainToBase.end());

    // Utility function which clones all instructions from "ChainToBase"
    // and inserts them before "InsertBefore". Returns rematerialized value
    // which should be used after statepoint.
    auto rematerializeChain = [&ChainToBase](
        Instruction *InsertBefore, Value *RootOfChain, Value *AlternateLiveBase) {
      Instruction *LastClonedValue = nullptr;
      Instruction *LastValue = nullptr;
      for (Instruction *Instr: ChainToBase) {
        // Only GEP's and casts are supported as we need to be careful to not
        // introduce any new uses of pointers not in the liveset.
        // Note that it's fine to introduce new uses of pointers which were
        // otherwise not used after this statepoint.
        assert(isa<GetElementPtrInst>(Instr) || isa<CastInst>(Instr));

        Instruction *ClonedValue = Instr->clone();
        ClonedValue->insertBefore(InsertBefore);
        ClonedValue->setName(Instr->getName() + ".remat");

        // If it is not first instruction in the chain then it uses previously
        // cloned value. We should update it to use cloned value.
        if (LastClonedValue) {
          assert(LastValue);
          ClonedValue->replaceUsesOfWith(LastValue, LastClonedValue);
#ifndef NDEBUG
          for (auto OpValue : ClonedValue->operand_values()) {
            // Assert that cloned instruction does not use any instructions from
            // this chain other than LastClonedValue
            assert(!is_contained(ChainToBase, OpValue) &&
                   "incorrect use in rematerialization chain");
            // Assert that the cloned instruction does not use the RootOfChain
            // or the AlternateLiveBase.
            assert(OpValue != RootOfChain && OpValue != AlternateLiveBase);
          }
#endif
        } else {
          // For the first instruction, replace the use of unrelocated base i.e.
          // RootOfChain/OrigRootPhi, with the corresponding PHI present in the
          // live set. They have been proved to be the same PHI nodes.  Note
          // that the *only* use of the RootOfChain in the ChainToBase list is
          // the first Value in the list.
          if (RootOfChain != AlternateLiveBase)
            ClonedValue->replaceUsesOfWith(RootOfChain, AlternateLiveBase);
        }

        LastClonedValue = ClonedValue;
        LastValue = Instr;
      }
      assert(LastClonedValue);
      return LastClonedValue;
    };

    // Different cases for calls and invokes. For invokes we need to clone
    // instructions both on normal and unwind path.
    if (isa<CallInst>(Call)) {
      Instruction *InsertBefore = Call->getNextNode();
      assert(InsertBefore);
      Instruction *RematerializedValue = rematerializeChain(
          InsertBefore, RootOfChain, Info.PointerToBase[LiveValue]);
      Info.RematerializedValues[RematerializedValue] = LiveValue;
    } else {
      auto *Invoke = cast<InvokeInst>(Call);

      Instruction *NormalInsertBefore =
          &*Invoke->getNormalDest()->getFirstInsertionPt();
      Instruction *UnwindInsertBefore =
          &*Invoke->getUnwindDest()->getFirstInsertionPt();

      Instruction *NormalRematerializedValue = rematerializeChain(
          NormalInsertBefore, RootOfChain, Info.PointerToBase[LiveValue]);
      Instruction *UnwindRematerializedValue = rematerializeChain(
          UnwindInsertBefore, RootOfChain, Info.PointerToBase[LiveValue]);

      Info.RematerializedValues[NormalRematerializedValue] = LiveValue;
      Info.RematerializedValues[UnwindRematerializedValue] = LiveValue;
    }
  }

  // Remove rematerializaed values from the live set
  for (auto LiveValue: LiveValuesToBeDeleted) {
    Info.LiveSet.remove(LiveValue);
  }
}

static bool inlineGetBaseAndOffset(Function &F,
                                   SmallVectorImpl<CallInst *> &Intrinsics,
                                   DefiningValueMapTy &DVCache) {
  auto &Context = F.getContext();
  auto &DL = F.getParent()->getDataLayout();
  bool Changed = false;

  for (auto *Callsite : Intrinsics)
    switch (Callsite->getIntrinsicID()) {
    case Intrinsic::experimental_gc_get_pointer_base: {
      Changed = true;
      Value *Base = findBasePointer(Callsite->getOperand(0), DVCache);
      assert(!DVCache.count(Callsite));
      auto *BaseBC = IRBuilder<>(Callsite).CreateBitCast(
          Base, Callsite->getType(), suffixed_name_or(Base, ".cast", ""));
      if (BaseBC != Base)
        DVCache[BaseBC] = Base;
      Callsite->replaceAllUsesWith(BaseBC);
      if (!BaseBC->hasName())
        BaseBC->takeName(Callsite);
      Callsite->eraseFromParent();
      break;
    }
    case Intrinsic::experimental_gc_get_pointer_offset: {
      Changed = true;
      Value *Derived = Callsite->getOperand(0);
      Value *Base = findBasePointer(Derived, DVCache);
      assert(!DVCache.count(Callsite));
      unsigned AddressSpace = Derived->getType()->getPointerAddressSpace();
      unsigned IntPtrSize = DL.getPointerSizeInBits(AddressSpace);
      IRBuilder<> Builder(Callsite);
      Value *BaseInt =
          Builder.CreatePtrToInt(Base, Type::getIntNTy(Context, IntPtrSize),
                                 suffixed_name_or(Base, ".int", ""));
      Value *DerivedInt =
          Builder.CreatePtrToInt(Derived, Type::getIntNTy(Context, IntPtrSize),
                                 suffixed_name_or(Derived, ".int", ""));
      Value *Offset = Builder.CreateSub(DerivedInt, BaseInt);
      Callsite->replaceAllUsesWith(Offset);
      Offset->takeName(Callsite);
      Callsite->eraseFromParent();
      break;
    }
    default:
      llvm_unreachable("Unknown intrinsic");
    }

  return Changed;
}

static bool insertParsePoints(Function &F, DominatorTree &DT,
                              TargetTransformInfo &TTI,
                              SmallVectorImpl<CallBase *> &ToUpdate,
                              DefiningValueMapTy &DVCache) {
#ifndef NDEBUG
  // sanity check the input
  std::set<CallBase *> Uniqued;
  Uniqued.insert(ToUpdate.begin(), ToUpdate.end());
  assert(Uniqued.size() == ToUpdate.size() && "no duplicates please!");

  for (CallBase *Call : ToUpdate)
    assert(Call->getFunction() == &F);
#endif

  // When inserting gc.relocates for invokes, we need to be able to insert at
  // the top of the successor blocks.  See the comment on
  // normalForInvokeSafepoint on exactly what is needed.  Note that this step
  // may restructure the CFG.
  for (CallBase *Call : ToUpdate) {
    auto *II = dyn_cast<InvokeInst>(Call);
    if (!II)
      continue;
    normalizeForInvokeSafepoint(II->getNormalDest(), II->getParent(), DT);
    normalizeForInvokeSafepoint(II->getUnwindDest(), II->getParent(), DT);
  }

  // A list of dummy calls added to the IR to keep various values obviously
  // live in the IR.  We'll remove all of these when done.
  SmallVector<CallInst *, 64> Holders;

  // Insert a dummy call with all of the deopt operands we'll need for the
  // actual safepoint insertion as arguments.  This ensures reference operands
  // in the deopt argument list are considered live through the safepoint (and
  // thus makes sure they get relocated.)
  for (CallBase *Call : ToUpdate) {
    SmallVector<Value *, 64> DeoptValues;

    for (Value *Arg : GetDeoptBundleOperands(Call)) {
      assert(!isUnhandledGCPointerType(Arg->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(Arg->getType()))
        DeoptValues.push_back(Arg);
    }

    insertUseHolderAfter(Call, DeoptValues, Holders);
  }

  SmallVector<PartiallyConstructedSafepointRecord, 64> Records(ToUpdate.size());

  // A) Identify all gc pointers which are statically live at the given call
  // site.
  findLiveReferences(F, DT, ToUpdate, Records);

  // B) Find the base pointers for each live pointer
  for (size_t i = 0; i < Records.size(); i++) {
    PartiallyConstructedSafepointRecord &info = Records[i];
    findBasePointers(DT, DVCache, ToUpdate[i], info);
  }

  // The base phi insertion logic (for any safepoint) may have inserted new
  // instructions which are now live at some safepoint.  The simplest such
  // example is:
  // loop:
  //   phi a  <-- will be a new base_phi here
  //   safepoint 1 <-- that needs to be live here
  //   gep a + 1
  //   safepoint 2
  //   br loop
  // We insert some dummy calls after each safepoint to definitely hold live
  // the base pointers which were identified for that safepoint.  We'll then
  // ask liveness for _every_ base inserted to see what is now live.  Then we
  // remove the dummy calls.
  Holders.reserve(Holders.size() + Records.size());
  for (size_t i = 0; i < Records.size(); i++) {
    PartiallyConstructedSafepointRecord &Info = Records[i];

    SmallVector<Value *, 128> Bases;
    for (auto Pair : Info.PointerToBase)
      Bases.push_back(Pair.second);

    insertUseHolderAfter(ToUpdate[i], Bases, Holders);
  }

  // By selecting base pointers, we've effectively inserted new uses. Thus, we
  // need to rerun liveness.  We may *also* have inserted new defs, but that's
  // not the key issue.
  recomputeLiveInValues(F, DT, ToUpdate, Records);

  if (PrintBasePointers) {
    for (auto &Info : Records) {
      errs() << "Base Pairs: (w/Relocation)\n";
      for (auto Pair : Info.PointerToBase) {
        errs() << " derived ";
        Pair.first->printAsOperand(errs(), false);
        errs() << " base ";
        Pair.second->printAsOperand(errs(), false);
        errs() << "\n";
      }
    }
  }

  // It is possible that non-constant live variables have a constant base.  For
  // example, a GEP with a variable offset from a global.  In this case we can
  // remove it from the liveset.  We already don't add constants to the liveset
  // because we assume they won't move at runtime and the GC doesn't need to be
  // informed about them.  The same reasoning applies if the base is constant.
  // Note that the relocation placement code relies on this filtering for
  // correctness as it expects the base to be in the liveset, which isn't true
  // if the base is constant.
  for (auto &Info : Records)
    for (auto &BasePair : Info.PointerToBase)
      if (isa<Constant>(BasePair.second))
        Info.LiveSet.remove(BasePair.first);

  for (CallInst *CI : Holders)
    CI->eraseFromParent();

  Holders.clear();

  // In order to reduce live set of statepoint we might choose to rematerialize
  // some values instead of relocating them. This is purely an optimization and
  // does not influence correctness.
  for (size_t i = 0; i < Records.size(); i++)
    rematerializeLiveValues(ToUpdate[i], Records[i], TTI);

  // We need this to safely RAUW and delete call or invoke return values that
  // may themselves be live over a statepoint.  For details, please see usage in
  // makeStatepointExplicitImpl.
  std::vector<DeferredReplacement> Replacements;

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < Records.size(); i++)
    makeStatepointExplicit(DT, ToUpdate[i], Records[i], Replacements);

  ToUpdate.clear(); // prevent accident use of invalid calls.

  for (auto &PR : Replacements)
    PR.doReplacement();

  Replacements.clear();

  for (auto &Info : Records) {
    // These live sets may contain state Value pointers, since we replaced calls
    // with operand bundles with calls wrapped in gc.statepoint, and some of
    // those calls may have been def'ing live gc pointers.  Clear these out to
    // avoid accidentally using them.
    //
    // TODO: We should create a separate data structure that does not contain
    // these live sets, and migrate to using that data structure from this point
    // onward.
    Info.LiveSet.clear();
    Info.PointerToBase.clear();
  }

  // Do all the fixups of the original live variables to their relocated selves
  SmallVector<Value *, 128> Live;
  for (size_t i = 0; i < Records.size(); i++) {
    PartiallyConstructedSafepointRecord &Info = Records[i];

    // We can't simply save the live set from the original insertion.  One of
    // the live values might be the result of a call which needs a safepoint.
    // That Value* no longer exists and we need to use the new gc_result.
    // Thankfully, the live set is embedded in the statepoint (and updated), so
    // we just grab that.
    llvm::append_range(Live, Info.StatepointToken->gc_args());
#ifndef NDEBUG
    // Do some basic sanity checks on our liveness results before performing
    // relocation.  Relocation can and will turn mistakes in liveness results
    // into non-sensical code which is must harder to debug.
    // TODO: It would be nice to test consistency as well
    assert(DT.isReachableFromEntry(Info.StatepointToken->getParent()) &&
           "statepoint must be reachable or liveness is meaningless");
    for (Value *V : Info.StatepointToken->gc_args()) {
      if (!isa<Instruction>(V))
        // Non-instruction values trivial dominate all possible uses
        continue;
      auto *LiveInst = cast<Instruction>(V);
      assert(DT.isReachableFromEntry(LiveInst->getParent()) &&
             "unreachable values should never be live");
      assert(DT.dominates(LiveInst, Info.StatepointToken) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
#endif
  }
  unique_unsorted(Live);

#ifndef NDEBUG
  // sanity check
  for (auto *Ptr : Live)
    assert(isHandledGCPointerType(Ptr->getType()) &&
           "must be a gc pointer type");
#endif

  relocationViaAlloca(F, DT, Live, Records);
  return !Records.empty();
}

// Handles both return values and arguments for Functions and calls.
template <typename AttrHolder>
static void RemoveNonValidAttrAtIndex(LLVMContext &Ctx, AttrHolder &AH,
                                      unsigned Index) {
  AttrBuilder R;
  AttributeSet AS = AH.getAttributes().getAttributes(Index);
  if (AS.getDereferenceableBytes())
    R.addAttribute(Attribute::get(Ctx, Attribute::Dereferenceable,
                                  AS.getDereferenceableBytes()));
  if (AS.getDereferenceableOrNullBytes())
    R.addAttribute(Attribute::get(Ctx, Attribute::DereferenceableOrNull,
                                  AS.getDereferenceableOrNullBytes()));
  for (auto Attr : ParamAttrsToStrip)
    if (AS.hasAttribute(Attr))
      R.addAttribute(Attr);

  if (!R.empty())
    AH.setAttributes(AH.getAttributes().removeAttributes(Ctx, Index, R));
}

static void stripNonValidAttributesFromPrototype(Function &F) {
  LLVMContext &Ctx = F.getContext();

  // Intrinsics are very delicate.  Lowering sometimes depends the presence
  // of certain attributes for correctness, but we may have also inferred
  // additional ones in the abstract machine model which need stripped.  This
  // assumes that the attributes defined in Intrinsic.td are conservatively
  // correct for both physical and abstract model.
  if (Intrinsic::ID id = F.getIntrinsicID()) {
    F.setAttributes(Intrinsic::getAttributes(Ctx, id));
    return;
  }

  for (Argument &A : F.args())
    if (isa<PointerType>(A.getType()))
      RemoveNonValidAttrAtIndex(Ctx, F,
                                A.getArgNo() + AttributeList::FirstArgIndex);

  if (isa<PointerType>(F.getReturnType()))
    RemoveNonValidAttrAtIndex(Ctx, F, AttributeList::ReturnIndex);

  for (auto Attr : FnAttrsToStrip)
    F.removeFnAttr(Attr);
}

/// Certain metadata on instructions are invalid after running RS4GC.
/// Optimizations that run after RS4GC can incorrectly use this metadata to
/// optimize functions. We drop such metadata on the instruction.
static void stripInvalidMetadataFromInstruction(Instruction &I) {
  if (!isa<LoadInst>(I) && !isa<StoreInst>(I))
    return;
  // These are the attributes that are still valid on loads and stores after
  // RS4GC.
  // The metadata implying dereferenceability and noalias are (conservatively)
  // dropped.  This is because semantically, after RewriteStatepointsForGC runs,
  // all calls to gc.statepoint "free" the entire heap. Also, gc.statepoint can
  // touch the entire heap including noalias objects. Note: The reasoning is
  // same as stripping the dereferenceability and noalias attributes that are
  // analogous to the metadata counterparts.
  // We also drop the invariant.load metadata on the load because that metadata
  // implies the address operand to the load points to memory that is never
  // changed once it became dereferenceable. This is no longer true after RS4GC.
  // Similar reasoning applies to invariant.group metadata, which applies to
  // loads within a group.
  unsigned ValidMetadataAfterRS4GC[] = {LLVMContext::MD_tbaa,
                         LLVMContext::MD_range,
                         LLVMContext::MD_alias_scope,
                         LLVMContext::MD_nontemporal,
                         LLVMContext::MD_nonnull,
                         LLVMContext::MD_align,
                         LLVMContext::MD_type};

  // Drops all metadata on the instruction other than ValidMetadataAfterRS4GC.
  I.dropUnknownNonDebugMetadata(ValidMetadataAfterRS4GC);
}

static void stripNonValidDataFromBody(Function &F) {
  if (F.empty())
    return;

  LLVMContext &Ctx = F.getContext();
  MDBuilder Builder(Ctx);

  // Set of invariantstart instructions that we need to remove.
  // Use this to avoid invalidating the instruction iterator.
  SmallVector<IntrinsicInst*, 12> InvariantStartInstructions;

  for (Instruction &I : instructions(F)) {
    // invariant.start on memory location implies that the referenced memory
    // location is constant and unchanging. This is no longer true after
    // RewriteStatepointsForGC runs because there can be calls to gc.statepoint
    // which frees the entire heap and the presence of invariant.start allows
    // the optimizer to sink the load of a memory location past a statepoint,
    // which is incorrect.
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::invariant_start) {
        InvariantStartInstructions.push_back(II);
        continue;
      }

    if (MDNode *Tag = I.getMetadata(LLVMContext::MD_tbaa)) {
      MDNode *MutableTBAA = Builder.createMutableTBAAAccessTag(Tag);
      I.setMetadata(LLVMContext::MD_tbaa, MutableTBAA);
    }

    stripInvalidMetadataFromInstruction(I);

    if (auto *Call = dyn_cast<CallBase>(&I)) {
      for (int i = 0, e = Call->arg_size(); i != e; i++)
        if (isa<PointerType>(Call->getArgOperand(i)->getType()))
          RemoveNonValidAttrAtIndex(Ctx, *Call,
                                    i + AttributeList::FirstArgIndex);
      if (isa<PointerType>(Call->getType()))
        RemoveNonValidAttrAtIndex(Ctx, *Call, AttributeList::ReturnIndex);
    }
  }

  // Delete the invariant.start instructions and RAUW undef.
  for (auto *II : InvariantStartInstructions) {
    II->replaceAllUsesWith(UndefValue::get(II->getType()));
    II->eraseFromParent();
  }
}

/// Returns true if this function should be rewritten by this pass.  The main
/// point of this function is as an extension point for custom logic.
static bool shouldRewriteStatepointsIn(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const auto &FunctionGCName = F.getGC();
    const StringRef StatepointExampleName("statepoint-example");
    const StringRef CoreCLRName("coreclr");
    return (StatepointExampleName == FunctionGCName) ||
           (CoreCLRName == FunctionGCName);
  } else
    return false;
}

static void stripNonValidData(Module &M) {
#ifndef NDEBUG
  assert(llvm::any_of(M, shouldRewriteStatepointsIn) && "precondition!");
#endif

  for (Function &F : M)
    stripNonValidAttributesFromPrototype(F);

  for (Function &F : M)
    stripNonValidDataFromBody(F);
}

bool RewriteStatepointsForGC::runOnFunction(Function &F, DominatorTree &DT,
                                            TargetTransformInfo &TTI,
                                            const TargetLibraryInfo &TLI) {
  assert(!F.isDeclaration() && !F.empty() &&
         "need function body to rewrite statepoints in");
  assert(shouldRewriteStatepointsIn(F) && "mismatch in rewrite decision");

  auto NeedsRewrite = [&TLI](Instruction &I) {
    if (const auto *Call = dyn_cast<CallBase>(&I)) {
      if (isa<GCStatepointInst>(Call))
        return false;
      if (callsGCLeafFunction(Call, TLI))
        return false;

      // Normally it's up to the frontend to make sure that non-leaf calls also
      // have proper deopt state if it is required. We make an exception for
      // element atomic memcpy/memmove intrinsics here. Unlike other intrinsics
      // these are non-leaf by default. They might be generated by the optimizer
      // which doesn't know how to produce a proper deopt state. So if we see a
      // non-leaf memcpy/memmove without deopt state just treat it as a leaf
      // copy and don't produce a statepoint.
      if (!AllowStatepointWithNoDeoptInfo &&
          !Call->getOperandBundle(LLVMContext::OB_deopt)) {
        assert((isa<AtomicMemCpyInst>(Call) || isa<AtomicMemMoveInst>(Call)) &&
               "Don't expect any other calls here!");
        return false;
      }
      return true;
    }
    return false;
  };

  // Delete any unreachable statepoints so that we don't have unrewritten
  // statepoints surviving this pass.  This makes testing easier and the
  // resulting IR less confusing to human readers.
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  bool MadeChange = removeUnreachableBlocks(F, &DTU);
  // Flush the Dominator Tree.
  DTU.getDomTree();

  // Gather all the statepoints which need rewritten.  Be careful to only
  // consider those in reachable code since we need to ask dominance queries
  // when rewriting.  We'll delete the unreachable ones in a moment.
  SmallVector<CallBase *, 64> ParsePointNeeded;
  SmallVector<CallInst *, 64> Intrinsics;
  for (Instruction &I : instructions(F)) {
    // TODO: only the ones with the flag set!
    if (NeedsRewrite(I)) {
      // NOTE removeUnreachableBlocks() is stronger than
      // DominatorTree::isReachableFromEntry(). In other words
      // removeUnreachableBlocks can remove some blocks for which
      // isReachableFromEntry() returns true.
      assert(DT.isReachableFromEntry(I.getParent()) &&
            "no unreachable blocks expected");
      ParsePointNeeded.push_back(cast<CallBase>(&I));
    }
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (CI->getIntrinsicID() == Intrinsic::experimental_gc_get_pointer_base ||
          CI->getIntrinsicID() == Intrinsic::experimental_gc_get_pointer_offset)
        Intrinsics.emplace_back(CI);
  }

  // Return early if no work to do.
  if (ParsePointNeeded.empty() && Intrinsics.empty())
    return MadeChange;

  // As a prepass, go ahead and aggressively destroy single entry phi nodes.
  // These are created by LCSSA.  They have the effect of increasing the size
  // of liveness sets for no good reason.  It may be harder to do this post
  // insertion since relocations and base phis can confuse things.
  for (BasicBlock &BB : F)
    if (BB.getUniquePredecessor())
      MadeChange |= FoldSingleEntryPHINodes(&BB);

  // Before we start introducing relocations, we want to tweak the IR a bit to
  // avoid unfortunate code generation effects.  The main example is that we
  // want to try to make sure the comparison feeding a branch is after any
  // safepoints.  Otherwise, we end up with a comparison of pre-relocation
  // values feeding a branch after relocation.  This is semantically correct,
  // but results in extra register pressure since both the pre-relocation and
  // post-relocation copies must be available in registers.  For code without
  // relocations this is handled elsewhere, but teaching the scheduler to
  // reverse the transform we're about to do would be slightly complex.
  // Note: This may extend the live range of the inputs to the icmp and thus
  // increase the liveset of any statepoint we move over.  This is profitable
  // as long as all statepoints are in rare blocks.  If we had in-register
  // lowering for live values this would be a much safer transform.
  auto getConditionInst = [](Instruction *TI) -> Instruction * {
    if (auto *BI = dyn_cast<BranchInst>(TI))
      if (BI->isConditional())
        return dyn_cast<Instruction>(BI->getCondition());
    // TODO: Extend this to handle switches
    return nullptr;
  };
  for (BasicBlock &BB : F) {
    Instruction *TI = BB.getTerminator();
    if (auto *Cond = getConditionInst(TI))
      // TODO: Handle more than just ICmps here.  We should be able to move
      // most instructions without side effects or memory access.
      if (isa<ICmpInst>(Cond) && Cond->hasOneUse()) {
        MadeChange = true;
        Cond->moveBefore(TI);
      }
  }

  // Nasty workaround - The base computation code in the main algorithm doesn't
  // consider the fact that a GEP can be used to convert a scalar to a vector.
  // The right fix for this is to integrate GEPs into the base rewriting
  // algorithm properly, this is just a short term workaround to prevent
  // crashes by canonicalizing such GEPs into fully vector GEPs.
  for (Instruction &I : instructions(F)) {
    if (!isa<GetElementPtrInst>(I))
      continue;

    unsigned VF = 0;
    for (unsigned i = 0; i < I.getNumOperands(); i++)
      if (auto *OpndVTy = dyn_cast<VectorType>(I.getOperand(i)->getType())) {
        assert(VF == 0 ||
               VF == cast<FixedVectorType>(OpndVTy)->getNumElements());
        VF = cast<FixedVectorType>(OpndVTy)->getNumElements();
      }

    // It's the vector to scalar traversal through the pointer operand which
    // confuses base pointer rewriting, so limit ourselves to that case.
    if (!I.getOperand(0)->getType()->isVectorTy() && VF != 0) {
      IRBuilder<> B(&I);
      auto *Splat = B.CreateVectorSplat(VF, I.getOperand(0));
      I.setOperand(0, Splat);
      MadeChange = true;
    }
  }

  // Cache the 'defining value' relation used in the computation and
  // insertion of base phis and selects.  This ensures that we don't insert
  // large numbers of duplicate base_phis. Use one cache for both
  // inlineGetBaseAndOffset() and insertParsePoints().
  DefiningValueMapTy DVCache;

  if (!Intrinsics.empty())
    // Inline @gc.get.pointer.base() and @gc.get.pointer.offset() before finding
    // live references.
    MadeChange |= inlineGetBaseAndOffset(F, Intrinsics, DVCache);

  if (!ParsePointNeeded.empty())
    MadeChange |= insertParsePoints(F, DT, TTI, ParsePointNeeded, DVCache);

  return MadeChange;
}

// liveness computation via standard dataflow
// -------------------------------------------------------------------

// TODO: Consider using bitvectors for liveness, the set of potentially
// interesting values should be small and easy to pre-compute.

/// Compute the live-in set for the location rbegin starting from
/// the live-out set of the basic block
static void computeLiveInValues(BasicBlock::reverse_iterator Begin,
                                BasicBlock::reverse_iterator End,
                                SetVector<Value *> &LiveTmp) {
  for (auto &I : make_range(Begin, End)) {
    // KILL/Def - Remove this definition from LiveIn
    LiveTmp.remove(&I);

    // Don't consider *uses* in PHI nodes, we handle their contribution to
    // predecessor blocks when we seed the LiveOut sets
    if (isa<PHINode>(I))
      continue;

    // USE - Add to the LiveIn set for this instruction
    for (Value *V : I.operands()) {
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V)) {
        // The choice to exclude all things constant here is slightly subtle.
        // There are two independent reasons:
        // - We assume that things which are constant (from LLVM's definition)
        // do not move at runtime.  For example, the address of a global
        // variable is fixed, even though it's contents may not be.
        // - Second, we can't disallow arbitrary inttoptr constants even
        // if the language frontend does.  Optimization passes are free to
        // locally exploit facts without respect to global reachability.  This
        // can create sections of code which are dynamically unreachable and
        // contain just about anything.  (see constants.ll in tests)
        LiveTmp.insert(V);
      }
    }
  }
}

static void computeLiveOutSeed(BasicBlock *BB, SetVector<Value *> &LiveTmp) {
  for (BasicBlock *Succ : successors(BB)) {
    for (auto &I : *Succ) {
      PHINode *PN = dyn_cast<PHINode>(&I);
      if (!PN)
        break;

      Value *V = PN->getIncomingValueForBlock(BB);
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V))
        LiveTmp.insert(V);
    }
  }
}

static SetVector<Value *> computeKillSet(BasicBlock *BB) {
  SetVector<Value *> KillSet;
  for (Instruction &I : *BB)
    if (isHandledGCPointerType(I.getType()))
      KillSet.insert(&I);
  return KillSet;
}

#ifndef NDEBUG
/// Check that the items in 'Live' dominate 'TI'.  This is used as a basic
/// sanity check for the liveness computation.
static void checkBasicSSA(DominatorTree &DT, SetVector<Value *> &Live,
                          Instruction *TI, bool TermOkay = false) {
  for (Value *V : Live) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // The terminator can be a member of the LiveOut set.  LLVM's definition
      // of instruction dominance states that V does not dominate itself.  As
      // such, we need to special case this to allow it.
      if (TermOkay && TI == I)
        continue;
      assert(DT.dominates(I, TI) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
  }
}

/// Check that all the liveness sets used during the computation of liveness
/// obey basic SSA properties.  This is useful for finding cases where we miss
/// a def.
static void checkBasicSSA(DominatorTree &DT, GCPtrLivenessData &Data,
                          BasicBlock &BB) {
  checkBasicSSA(DT, Data.LiveSet[&BB], BB.getTerminator());
  checkBasicSSA(DT, Data.LiveOut[&BB], BB.getTerminator(), true);
  checkBasicSSA(DT, Data.LiveIn[&BB], BB.getTerminator());
}
#endif

static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data) {
  SmallSetVector<BasicBlock *, 32> Worklist;

  // Seed the liveness for each individual block
  for (BasicBlock &BB : F) {
    Data.KillSet[&BB] = computeKillSet(&BB);
    Data.LiveSet[&BB].clear();
    computeLiveInValues(BB.rbegin(), BB.rend(), Data.LiveSet[&BB]);

#ifndef NDEBUG
    for (Value *Kill : Data.KillSet[&BB])
      assert(!Data.LiveSet[&BB].count(Kill) && "live set contains kill");
#endif

    Data.LiveOut[&BB] = SetVector<Value *>();
    computeLiveOutSeed(&BB, Data.LiveOut[&BB]);
    Data.LiveIn[&BB] = Data.LiveSet[&BB];
    Data.LiveIn[&BB].set_union(Data.LiveOut[&BB]);
    Data.LiveIn[&BB].set_subtract(Data.KillSet[&BB]);
    if (!Data.LiveIn[&BB].empty())
      Worklist.insert(pred_begin(&BB), pred_end(&BB));
  }

  // Propagate that liveness until stable
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();

    // Compute our new liveout set, then exit early if it hasn't changed despite
    // the contribution of our successor.
    SetVector<Value *> LiveOut = Data.LiveOut[BB];
    const auto OldLiveOutSize = LiveOut.size();
    for (BasicBlock *Succ : successors(BB)) {
      assert(Data.LiveIn.count(Succ));
      LiveOut.set_union(Data.LiveIn[Succ]);
    }
    // assert OutLiveOut is a subset of LiveOut
    if (OldLiveOutSize == LiveOut.size()) {
      // If the sets are the same size, then we didn't actually add anything
      // when unioning our successors LiveIn.  Thus, the LiveIn of this block
      // hasn't changed.
      continue;
    }
    Data.LiveOut[BB] = LiveOut;

    // Apply the effects of this basic block
    SetVector<Value *> LiveTmp = LiveOut;
    LiveTmp.set_union(Data.LiveSet[BB]);
    LiveTmp.set_subtract(Data.KillSet[BB]);

    assert(Data.LiveIn.count(BB));
    const SetVector<Value *> &OldLiveIn = Data.LiveIn[BB];
    // assert: OldLiveIn is a subset of LiveTmp
    if (OldLiveIn.size() != LiveTmp.size()) {
      Data.LiveIn[BB] = LiveTmp;
      Worklist.insert(pred_begin(BB), pred_end(BB));
    }
  } // while (!Worklist.empty())

#ifndef NDEBUG
  // Sanity check our output against SSA properties.  This helps catch any
  // missing kills during the above iteration.
  for (BasicBlock &BB : F)
    checkBasicSSA(DT, Data, BB);
#endif
}

static void findLiveSetAtInst(Instruction *Inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &Out) {
  BasicBlock *BB = Inst->getParent();

  // Note: The copy is intentional and required
  assert(Data.LiveOut.count(BB));
  SetVector<Value *> LiveOut = Data.LiveOut[BB];

  // We want to handle the statepoint itself oddly.  It's
  // call result is not live (normal), nor are it's arguments
  // (unless they're used again later).  This adjustment is
  // specifically what we need to relocate
  computeLiveInValues(BB->rbegin(), ++Inst->getIterator().getReverse(),
                      LiveOut);
  LiveOut.remove(Inst);
  Out.insert(LiveOut.begin(), LiveOut.end());
}

static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  CallBase *Call,
                                  PartiallyConstructedSafepointRecord &Info) {
  StatepointLiveSetTy Updated;
  findLiveSetAtInst(Call, RevisedLivenessData, Updated);

  // We may have base pointers which are now live that weren't before.  We need
  // to update the PointerToBase structure to reflect this.
  for (auto V : Updated)
    Info.PointerToBase.insert({V, V});

#ifndef NDEBUG
  for (auto V : Updated)
    assert(Info.PointerToBase.count(V) &&
           "Must be able to find base for live value!");
#endif

  // Remove any stale base mappings - this can happen since our liveness is
  // more precise then the one inherent in the base pointer analysis.
  DenseSet<Value *> ToErase;
  for (auto KVPair : Info.PointerToBase)
    if (!Updated.count(KVPair.first))
      ToErase.insert(KVPair.first);

  for (auto *V : ToErase)
    Info.PointerToBase.erase(V);

#ifndef NDEBUG
  for (auto KVPair : Info.PointerToBase)
    assert(Updated.count(KVPair.first) && "record for non-live value");
#endif

  Info.LiveSet = Updated;
}
