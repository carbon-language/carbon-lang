//===- RewriteStatepointsForGC.cpp - Make GC relocations explicit ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite an existing set of gc.statepoints such that they make potential
// relocations performed by the garbage collector explicit in the IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#define DEBUG_TYPE "rewrite-statepoints-for-gc"

using namespace llvm;

// Print tracing output
static cl::opt<bool> TraceLSP("trace-rewrite-statepoints", cl::Hidden,
                              cl::init(false));

// Print the liveset found at the insert location
static cl::opt<bool> PrintLiveSet("spp-print-liveset", cl::Hidden,
                                  cl::init(false));
static cl::opt<bool> PrintLiveSetSize("spp-print-liveset-size",
                                      cl::Hidden, cl::init(false));
// Print out the base pointers for debugging
static cl::opt<bool> PrintBasePointers("spp-print-base-pointers",
                                       cl::Hidden, cl::init(false));

namespace {
struct RewriteStatepointsForGC : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  RewriteStatepointsForGC() : FunctionPass(ID) {
    initializeRewriteStatepointsForGCPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We add and rewrite a bunch of instructions, but don't really do much
    // else.  We could in theory preserve a lot more analyses here.
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};
} // namespace

char RewriteStatepointsForGC::ID = 0;

FunctionPass *llvm::createRewriteStatepointsForGCPass() {
  return new RewriteStatepointsForGC();
}

INITIALIZE_PASS_BEGIN(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                      "Make relocations explicit at statepoints", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                    "Make relocations explicit at statepoints", false, false)

namespace {
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
typedef DenseMap<Value *, Value *> DefiningValueMapTy;
typedef DenseSet<llvm::Value *> StatepointLiveSetTy;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live accross this safepoint
  StatepointLiveSetTy liveset;

  /// Mapping from live pointers to a base-defining-value
  DenseMap<llvm::Value *, llvm::Value *> PointerToBase;

  /// Any new values which were added to the IR during base pointer analysis
  /// for this safepoint
  DenseSet<llvm::Value *> NewInsertedDefs;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  Instruction *StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  Instruction *UnwindToken;
};
}

// TODO: Once we can get to the GCStrategy, this becomes
// Optional<bool> isGCManagedPointer(const Value *V) const override {

static bool isGCPointerType(const Type *T) {
  if (const PointerType *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return (1 == PT->getAddressSpace());
  return false;
}

/// Return true if the Value is a gc reference type which is potentially used
/// after the instruction 'loc'.  This is only used with the edge reachability
/// liveness code.  Note: It is assumed the V dominates loc.
static bool isLiveGCReferenceAt(Value &V, Instruction *loc, DominatorTree &DT,
                                LoopInfo *LI) {
  if (!isGCPointerType(V.getType()))
    return false;

  if (V.use_empty())
    return false;

  // Given assumption that V dominates loc, this may be live
  return true;
}

#ifndef NDEBUG
static bool isAggWhichContainsGCPtrType(Type *Ty) {
  if (VectorType *VT = dyn_cast<VectorType>(Ty))
    return isGCPointerType(VT->getScalarType());
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty))
    return isGCPointerType(AT->getElementType()) ||
           isAggWhichContainsGCPtrType(AT->getElementType());
  if (StructType *ST = dyn_cast<StructType>(Ty))
    return std::any_of(ST->subtypes().begin(), ST->subtypes().end(),
                       [](Type *SubType) {
                         return isGCPointerType(SubType) ||
                                isAggWhichContainsGCPtrType(SubType);
                       });
  return false;
}
#endif

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
//
// preconditions: valid IR graph, term is either a terminator instruction or
// a call instruction, pred is the basic block of term, DT, LI are valid
//
// side effects: none, does not mutate IR
//
//  postconditions: populates liveValues as discussed above
static void findLiveGCValuesAtInst(Instruction *term, BasicBlock *pred,
                                   DominatorTree &DT, LoopInfo *LI,
                                   StatepointLiveSetTy &liveValues) {
  liveValues.clear();

  assert(isa<CallInst>(term) || isa<InvokeInst>(term) || term->isTerminator());

  Function *F = pred->getParent();

  auto is_live_gc_reference =
      [&](Value &V) { return isLiveGCReferenceAt(V, term, DT, LI); };

  // Are there any gc pointer arguments live over this point?  This needs to be
  // special cased since arguments aren't defined in basic blocks.
  for (Argument &arg : F->args()) {
    assert(!isAggWhichContainsGCPtrType(arg.getType()) &&
           "support for FCA unimplemented");

    if (is_live_gc_reference(arg)) {
      liveValues.insert(&arg);
    }
  }

  // Walk through all dominating blocks - the ones which can contain
  // definitions used in this block - and check to see if any of the values
  // they define are used in locations potentially reachable from the
  // interesting instruction.
  BasicBlock *BBI = pred;
  while (true) {
    if (TraceLSP) {
      errs() << "[LSP] Looking at dominating block " << pred->getName() << "\n";
    }
    assert(DT.dominates(BBI, pred));
    assert(isPotentiallyReachable(BBI, pred, &DT) &&
           "dominated block must be reachable");

    // Walk through the instructions in dominating blocks and keep any
    // that have a use potentially reachable from the block we're
    // considering putting the safepoint in
    for (Instruction &inst : *BBI) {
      if (TraceLSP) {
        errs() << "[LSP] Looking at instruction ";
        inst.dump();
      }

      if (pred == BBI && (&inst) == term) {
        if (TraceLSP) {
          errs() << "[LSP] stopped because we encountered the safepoint "
                    "instruction.\n";
        }

        // If we're in the block which defines the interesting instruction,
        // we don't want to include any values as live which are defined
        // _after_ the interesting line or as part of the line itself
        // i.e. "term" is the call instruction for a call safepoint, the
        // results of the call should not be considered live in that stackmap
        break;
      }

      assert(!isAggWhichContainsGCPtrType(inst.getType()) &&
             "support for FCA unimplemented");

      if (is_live_gc_reference(inst)) {
        if (TraceLSP) {
          errs() << "[LSP] found live value for this safepoint ";
          inst.dump();
          term->dump();
        }
        liveValues.insert(&inst);
      }
    }
    if (!DT.getNode(BBI)->getIDom()) {
      assert(BBI == &F->getEntryBlock() &&
             "failed to find a dominator for something other than "
             "the entry block");
      break;
    }
    BBI = DT.getNode(BBI)->getIDom()->getBlock();
  }
}

static bool order_by_name(llvm::Value *a, llvm::Value *b) {
  if (a->hasName() && b->hasName()) {
    return -1 == a->getName().compare(b->getName());
  } else if (a->hasName() && !b->hasName()) {
    return true;
  } else if (!a->hasName() && b->hasName()) {
    return false;
  } else {
    // Better than nothing, but not stable
    return a < b;
  }
}

/// Find the initial live set. Note that due to base pointer
/// insertion, the live set may be incomplete.
static void
analyzeParsePointLiveness(DominatorTree &DT, const CallSite &CS,
                          PartiallyConstructedSafepointRecord &result) {
  Instruction *inst = CS.getInstruction();

  BasicBlock *BB = inst->getParent();
  StatepointLiveSetTy liveset;
  findLiveGCValuesAtInst(inst, BB, DT, nullptr, liveset);

  if (PrintLiveSet) {
    // Note: This output is used by several of the test cases
    // The order of elemtns in a set is not stable, put them in a vec and sort
    // by name
    SmallVector<Value *, 64> temp;
    temp.insert(temp.end(), liveset.begin(), liveset.end());
    std::sort(temp.begin(), temp.end(), order_by_name);
    errs() << "Live Variables:\n";
    for (Value *V : temp) {
      errs() << " " << V->getName(); // no newline
      V->dump();
    }
  }
  if (PrintLiveSetSize) {
    errs() << "Safepoint For: " << CS.getCalledValue()->getName() << "\n";
    errs() << "Number live values: " << liveset.size() << "\n";
  }
  result.liveset = liveset;
}

/// True iff this value is the null pointer constant (of any pointer type)
static bool LLVM_ATTRIBUTE_UNUSED isNullConstant(Value *V) {
  return isa<Constant>(V) && isa<PointerType>(V->getType()) &&
         cast<Constant>(V)->isNullValue();
}

/// Helper function for findBasePointer - Will return a value which either a)
/// defines the base pointer for the input or b) blocks the simple search
/// (i.e. a PHI or Select of two derived pointers)
static Value *findBaseDefiningValue(Value *I) {
  assert(I->getType()->isPointerTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  // There are instructions which can never return gc pointer values.  Sanity
  // check
  // that this is actually true.
  assert(!isa<InsertElementInst>(I) && !isa<ExtractElementInst>(I) &&
         !isa<ShuffleVectorInst>(I) && "Vector types are not gc pointers");
  assert((!isa<Instruction>(I) || isa<InvokeInst>(I) ||
          !cast<Instruction>(I)->isTerminator()) &&
         "With the exception of invoke terminators don't define values");
  assert(!isa<StoreInst>(I) && !isa<FenceInst>(I) &&
         "Can't be definitions to start with");
  assert(!isa<ICmpInst>(I) && !isa<FCmpInst>(I) &&
         "Comparisons don't give ops");
  // There's a bunch of instructions which just don't make sense to apply to
  // a pointer.  The only valid reason for this would be pointer bit
  // twiddling which we're just not going to support.
  assert((!isa<Instruction>(I) || !cast<Instruction>(I)->isBinaryOp()) &&
         "Binary ops on pointer values are meaningless.  Unless your "
         "bit-twiddling which we don't support");

  if (Argument *Arg = dyn_cast<Argument>(I)) {
    // An incoming argument to the function is a base pointer
    // We should have never reached here if this argument isn't an gc value
    assert(Arg->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return Arg;
  }

  if (GlobalVariable *global = dyn_cast<GlobalVariable>(I)) {
    // base case
    assert(global->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return global;
  }

  // inlining could possibly introduce phi node that contains
  // undef if callee has multiple returns
  if (UndefValue *undef = dyn_cast<UndefValue>(I)) {
    assert(undef->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return undef; // utterly meaningless, but useful for dealing with
                  // partially optimized code.
  }

  // Due to inheritance, this must be _after_ the global variable and undef
  // checks
  if (Constant *con = dyn_cast<Constant>(I)) {
    assert(!isa<GlobalVariable>(I) && !isa<UndefValue>(I) &&
           "order of checks wrong!");
    // Note: Finding a constant base for something marked for relocation
    // doesn't really make sense.  The most likely case is either a) some
    // screwed up the address space usage or b) your validating against
    // compiled C++ code w/o the proper separation.  The only real exception
    // is a null pointer.  You could have generic code written to index of
    // off a potentially null value and have proven it null.  We also use
    // null pointers in dead paths of relocation phis (which we might later
    // want to find a base pointer for).
    assert(con->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    assert(con->isNullValue() && "null is the only case which makes sense");
    return con;
  }

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value *def = CI->stripPointerCasts();
    assert(def->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    // If we find a cast instruction here, it means we've found a cast which is
    // not simply a pointer cast (i.e. an inttoptr).  We don't know how to
    // handle int->ptr conversion.
    assert(!isa<CastInst>(def) && "shouldn't find another cast here");
    return findBaseDefiningValue(def);
  }

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (LI->getType()->isPointerTy()) {
      Value *Op = LI->getOperand(0);
      (void)Op;
      // Has to be a pointer to an gc object, or possibly an array of such?
      assert(Op->getType()->isPointerTy());
      return LI; // The value loaded is an gc base itself
    }
  }
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    Value *Op = GEP->getOperand(0);
    if (Op->getType()->isPointerTy()) {
      return findBaseDefiningValue(Op); // The base of this GEP is the base
    }
  }

  if (AllocaInst *alloc = dyn_cast<AllocaInst>(I)) {
    // An alloca represents a conceptual stack slot.  It's the slot itself
    // that the GC needs to know about, not the value in the slot.
    assert(alloc->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return alloc;
  }

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      // fall through to general call handling
      break;
    case Intrinsic::experimental_gc_statepoint:
    case Intrinsic::experimental_gc_result_float:
    case Intrinsic::experimental_gc_result_int:
      llvm_unreachable("these don't produce pointers");
    case Intrinsic::experimental_gc_result_ptr:
      // This is just a special case of the CallInst check below to handle a
      // statepoint with deopt args which hasn't been rewritten for GC yet.
      // TODO: Assert that the statepoint isn't rewritten yet.
      return II;
    case Intrinsic::experimental_gc_relocate: {
      // Rerunning safepoint insertion after safepoints are already
      // inserted is not supported.  It could probably be made to work,
      // but why are you doing this?  There's no good reason.
      llvm_unreachable("repeat safepoint insertion is not supported");
    }
    case Intrinsic::gcroot:
      // Currently, this mechanism hasn't been extended to work with gcroot.
      // There's no reason it couldn't be, but I haven't thought about the
      // implications much.
      llvm_unreachable(
          "interaction with the gcroot mechanism is not supported");
    }
  }
  // We assume that functions in the source language only return base
  // pointers.  This should probably be generalized via attributes to support
  // both source language and internal functions.
  if (CallInst *call = dyn_cast<CallInst>(I)) {
    assert(call->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return call;
  }
  if (InvokeInst *invoke = dyn_cast<InvokeInst>(I)) {
    assert(invoke->getType()->isPointerTy() &&
           "Base for pointer must be another pointer");
    return invoke;
  }

  // I have absolutely no idea how to implement this part yet.  It's not
  // neccessarily hard, I just haven't really looked at it yet.
  assert(!isa<LandingPadInst>(I) && "Landing Pad is unimplemented");

  if (AtomicCmpXchgInst *cas = dyn_cast<AtomicCmpXchgInst>(I)) {
    // A CAS is effectively a atomic store and load combined under a
    // predicate.  From the perspective of base pointers, we just treat it
    // like a load.  We loaded a pointer from a address in memory, that value
    // had better be a valid base pointer.
    return cas->getPointerOperand();
  }
  if (AtomicRMWInst *atomic = dyn_cast<AtomicRMWInst>(I)) {
    assert(AtomicRMWInst::Xchg == atomic->getOperation() &&
           "All others are binary ops which don't apply to base pointers");
    // semantically, a load, store pair.  Treat it the same as a standard load
    return atomic->getPointerOperand();
  }

  // The aggregate ops.  Aggregates can either be in the heap or on the
  // stack, but in either case, this is simply a field load.  As a result,
  // this is a defining definition of the base just like a load is.
  if (ExtractValueInst *ev = dyn_cast<ExtractValueInst>(I)) {
    return ev;
  }

  // We should never see an insert vector since that would require we be
  // tracing back a struct value not a pointer value.
  assert(!isa<InsertValueInst>(I) &&
         "Base pointer for a struct is meaningless");

  // The last two cases here don't return a base pointer.  Instead, they
  // return a value which dynamically selects from amoung several base
  // derived pointers (each with it's own base potentially).  It's the job of
  // the caller to resolve these.
  if (SelectInst *select = dyn_cast<SelectInst>(I)) {
    return select;
  }

  return cast<PHINode>(I);
}

/// Returns the base defining value for this value.
static Value *findBaseDefiningValueCached(Value *I, DefiningValueMapTy &cache) {
  Value *&Cached = cache[I];
  if (!Cached) {
    Cached = findBaseDefiningValue(I);
  }
  assert(cache[I] != nullptr);

  if (TraceLSP) {
    errs() << "fBDV-cached: " << I->getName() << " -> " << Cached->getName()
           << "\n";
  }
  return Cached;
}

/// Return a base pointer for this value if known.  Otherwise, return it's
/// base defining value.
static Value *findBaseOrBDV(Value *I, DefiningValueMapTy &cache) {
  Value *def = findBaseDefiningValueCached(I, cache);
  auto Found = cache.find(def);
  if (Found != cache.end()) {
    // Either a base-of relation, or a self reference.  Caller must check.
    return Found->second;
  }
  // Only a BDV available
  return def;
}

/// Given the result of a call to findBaseDefiningValue, or findBaseOrBDV,
/// is it known to be a base pointer?  Or do we need to continue searching.
static bool isKnownBaseResult(Value *v) {
  if (!isa<PHINode>(v) && !isa<SelectInst>(v)) {
    // no recursion possible
    return true;
  }
  if (cast<Instruction>(v)->getMetadata("is_base_value")) {
    // This is a previously inserted base phi or select.  We know
    // that this is a base value.
    return true;
  }

  // We need to keep searching
  return false;
}

// TODO: find a better name for this
namespace {
class PhiState {
public:
  enum Status { Unknown, Base, Conflict };

  PhiState(Status s, Value *b = nullptr) : status(s), base(b) {
    assert(status != Base || b);
  }
  PhiState(Value *b) : status(Base), base(b) {}
  PhiState() : status(Unknown), base(nullptr) {}

  Status getStatus() const { return status; }
  Value *getBase() const { return base; }

  bool isBase() const { return getStatus() == Base; }
  bool isUnknown() const { return getStatus() == Unknown; }
  bool isConflict() const { return getStatus() == Conflict; }

  bool operator==(const PhiState &other) const {
    return base == other.base && status == other.status;
  }

  bool operator!=(const PhiState &other) const { return !(*this == other); }

  void dump() {
    errs() << status << " (" << base << " - "
           << (base ? base->getName() : "nullptr") << "): ";
  }

private:
  Status status;
  Value *base; // non null only if status == base
};

typedef DenseMap<Value *, PhiState> ConflictStateMapTy;
// Values of type PhiState form a lattice, and this is a helper
// class that implementes the meet operation.  The meat of the meet
// operation is implemented in MeetPhiStates::pureMeet
class MeetPhiStates {
public:
  // phiStates is a mapping from PHINodes and SelectInst's to PhiStates.
  explicit MeetPhiStates(const ConflictStateMapTy &phiStates)
      : phiStates(phiStates) {}

  // Destructively meet the current result with the base V.  V can
  // either be a merge instruction (SelectInst / PHINode), in which
  // case its status is looked up in the phiStates map; or a regular
  // SSA value, in which case it is assumed to be a base.
  void meetWith(Value *V) {
    PhiState otherState = getStateForBDV(V);
    assert((MeetPhiStates::pureMeet(otherState, currentResult) ==
            MeetPhiStates::pureMeet(currentResult, otherState)) &&
           "math is wrong: meet does not commute!");
    currentResult = MeetPhiStates::pureMeet(otherState, currentResult);
  }

  PhiState getResult() const { return currentResult; }

private:
  const ConflictStateMapTy &phiStates;
  PhiState currentResult;

  /// Return a phi state for a base defining value.  We'll generate a new
  /// base state for known bases and expect to find a cached state otherwise
  PhiState getStateForBDV(Value *baseValue) {
    if (isKnownBaseResult(baseValue)) {
      return PhiState(baseValue);
    } else {
      return lookupFromMap(baseValue);
    }
  }

  PhiState lookupFromMap(Value *V) {
    auto I = phiStates.find(V);
    assert(I != phiStates.end() && "lookup failed!");
    return I->second;
  }

  static PhiState pureMeet(const PhiState &stateA, const PhiState &stateB) {
    switch (stateA.getStatus()) {
    case PhiState::Unknown:
      return stateB;

    case PhiState::Base:
      assert(stateA.getBase() && "can't be null");
      if (stateB.isUnknown())
        return stateA;

      if (stateB.isBase()) {
        if (stateA.getBase() == stateB.getBase()) {
          assert(stateA == stateB && "equality broken!");
          return stateA;
        }
        return PhiState(PhiState::Conflict);
      }
      assert(stateB.isConflict() && "only three states!");
      return PhiState(PhiState::Conflict);

    case PhiState::Conflict:
      return stateA;
    }
    llvm_unreachable("only three states!");
  }
};
}
/// For a given value or instruction, figure out what base ptr it's derived
/// from.  For gc objects, this is simply itself.  On success, returns a value
/// which is the base pointer.  (This is reliable and can be used for
/// relocation.)  On failure, returns nullptr.
static Value *findBasePointer(Value *I, DefiningValueMapTy &cache,
                              DenseSet<llvm::Value *> &NewInsertedDefs) {
  Value *def = findBaseOrBDV(I, cache);

  if (isKnownBaseResult(def)) {
    return def;
  }

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
  // analougous to pessimistic data flow and would likely lead to an
  // overall worse solution.

  ConflictStateMapTy states;
  states[def] = PhiState();
  // Recursively fill in all phis & selects reachable from the initial one
  // for which we don't already know a definite base value for
  // TODO: This should be rewritten with a worklist
  bool done = false;
  while (!done) {
    done = true;
    // Since we're adding elements to 'states' as we run, we can't keep
    // iterators into the set.
    SmallVector<Value*, 16> Keys;
    Keys.reserve(states.size());
    for (auto Pair : states) {
      Value *V = Pair.first;
      Keys.push_back(V);
    }
    for (Value *v : Keys) {
      assert(!isKnownBaseResult(v) && "why did it get added?");
      if (PHINode *phi = dyn_cast<PHINode>(v)) {
        assert(phi->getNumIncomingValues() > 0 &&
               "zero input phis are illegal");
        for (Value *InVal : phi->incoming_values()) {
          Value *local = findBaseOrBDV(InVal, cache);
          if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
            states[local] = PhiState();
            done = false;
          }
        }
      } else if (SelectInst *sel = dyn_cast<SelectInst>(v)) {
        Value *local = findBaseOrBDV(sel->getTrueValue(), cache);
        if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
          states[local] = PhiState();
          done = false;
        }
        local = findBaseOrBDV(sel->getFalseValue(), cache);
        if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
          states[local] = PhiState();
          done = false;
        }
      }
    }
  }

  if (TraceLSP) {
    errs() << "States after initialization:\n";
    for (auto Pair : states) {
      Instruction *v = cast<Instruction>(Pair.first);
      PhiState state = Pair.second;
      state.dump();
      v->dump();
    }
  }

  // TODO: come back and revisit the state transitions around inputs which
  // have reached conflict state.  The current version seems too conservative.

  bool progress = true;
  while (progress) {
#ifndef NDEBUG
    size_t oldSize = states.size();
#endif
    progress = false;
    // We're only changing keys in this loop, thus safe to keep iterators
    for (auto Pair : states) {
      MeetPhiStates calculateMeet(states);
      Value *v = Pair.first;
      assert(!isKnownBaseResult(v) && "why did it get added?");
      if (SelectInst *select = dyn_cast<SelectInst>(v)) {
        calculateMeet.meetWith(findBaseOrBDV(select->getTrueValue(), cache));
        calculateMeet.meetWith(findBaseOrBDV(select->getFalseValue(), cache));
      } else
        for (Value *Val : cast<PHINode>(v)->incoming_values())
          calculateMeet.meetWith(findBaseOrBDV(Val, cache));

      PhiState oldState = states[v];
      PhiState newState = calculateMeet.getResult();
      if (oldState != newState) {
        progress = true;
        states[v] = newState;
      }
    }

    assert(oldSize <= states.size());
    assert(oldSize == states.size() || progress);
  }

  if (TraceLSP) {
    errs() << "States after meet iteration:\n";
    for (auto Pair : states) {
      Instruction *v = cast<Instruction>(Pair.first);
      PhiState state = Pair.second;
      state.dump();
      v->dump();
    }
  }

  // Insert Phis for all conflicts
  // We want to keep naming deterministic in the loop that follows, so
  // sort the keys before iteration.  This is useful in allowing us to
  // write stable tests. Note that there is no invalidation issue here.
  SmallVector<Value*, 16> Keys;
  Keys.reserve(states.size());
  for (auto Pair : states) {
    Value *V = Pair.first;
    Keys.push_back(V);
  }
  std::sort(Keys.begin(), Keys.end(), order_by_name);
  // TODO: adjust naming patterns to avoid this order of iteration dependency
  for (Value *V : Keys) {
    Instruction *v = cast<Instruction>(V);
    PhiState state = states[V];
    assert(!isKnownBaseResult(v) && "why did it get added?");
    assert(!state.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!state.isConflict())
      continue;
    
    if (isa<PHINode>(v)) {
      int num_preds =
          std::distance(pred_begin(v->getParent()), pred_end(v->getParent()));
      assert(num_preds > 0 && "how did we reach here");
      PHINode *phi = PHINode::Create(v->getType(), num_preds, "base_phi", v);
      NewInsertedDefs.insert(phi);
      // Add metadata marking this as a base value
      auto *const_1 = ConstantInt::get(
          Type::getInt32Ty(
              v->getParent()->getParent()->getParent()->getContext()),
          1);
      auto MDConst = ConstantAsMetadata::get(const_1);
      MDNode *md = MDNode::get(
          v->getParent()->getParent()->getParent()->getContext(), MDConst);
      phi->setMetadata("is_base_value", md);
      states[v] = PhiState(PhiState::Conflict, phi);
    } else {
      SelectInst *sel = cast<SelectInst>(v);
      // The undef will be replaced later
      UndefValue *undef = UndefValue::get(sel->getType());
      SelectInst *basesel = SelectInst::Create(sel->getCondition(), undef,
                                               undef, "base_select", sel);
      NewInsertedDefs.insert(basesel);
      // Add metadata marking this as a base value
      auto *const_1 = ConstantInt::get(
          Type::getInt32Ty(
              v->getParent()->getParent()->getParent()->getContext()),
          1);
      auto MDConst = ConstantAsMetadata::get(const_1);
      MDNode *md = MDNode::get(
          v->getParent()->getParent()->getParent()->getContext(), MDConst);
      basesel->setMetadata("is_base_value", md);
      states[v] = PhiState(PhiState::Conflict, basesel);
    }
  }

  // Fixup all the inputs of the new PHIs
  for (auto Pair : states) {
    Instruction *v = cast<Instruction>(Pair.first);
    PhiState state = Pair.second;

    assert(!isKnownBaseResult(v) && "why did it get added?");
    assert(!state.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!state.isConflict())
      continue;
    
    if (PHINode *basephi = dyn_cast<PHINode>(state.getBase())) {
      PHINode *phi = cast<PHINode>(v);
      unsigned NumPHIValues = phi->getNumIncomingValues();
      for (unsigned i = 0; i < NumPHIValues; i++) {
        Value *InVal = phi->getIncomingValue(i);
        BasicBlock *InBB = phi->getIncomingBlock(i);

        // If we've already seen InBB, add the same incoming value
        // we added for it earlier.  The IR verifier requires phi
        // nodes with multiple entries from the same basic block
        // to have the same incoming value for each of those
        // entries.  If we don't do this check here and basephi
        // has a different type than base, we'll end up adding two
        // bitcasts (and hence two distinct values) as incoming
        // values for the same basic block.

        int blockIndex = basephi->getBasicBlockIndex(InBB);
        if (blockIndex != -1) {
          Value *oldBase = basephi->getIncomingValue(blockIndex);
          basephi->addIncoming(oldBase, InBB);
#ifndef NDEBUG
          Value *base = findBaseOrBDV(InVal, cache);
          if (!isKnownBaseResult(base)) {
            // Either conflict or base.
            assert(states.count(base));
            base = states[base].getBase();
            assert(base != nullptr && "unknown PhiState!");
            assert(NewInsertedDefs.count(base) &&
                   "should have already added this in a prev. iteration!");
          }

          // In essense this assert states: the only way two
          // values incoming from the same basic block may be
          // different is by being different bitcasts of the same
          // value.  A cleanup that remains TODO is changing
          // findBaseOrBDV to return an llvm::Value of the correct
          // type (and still remain pure).  This will remove the
          // need to add bitcasts.
          assert(base->stripPointerCasts() == oldBase->stripPointerCasts() &&
                 "sanity -- findBaseOrBDV should be pure!");
#endif
          continue;
        }

        // Find either the defining value for the PHI or the normal base for
        // a non-phi node
        Value *base = findBaseOrBDV(InVal, cache);
        if (!isKnownBaseResult(base)) {
          // Either conflict or base.
          assert(states.count(base));
          base = states[base].getBase();
          assert(base != nullptr && "unknown PhiState!");
        }
        assert(base && "can't be null");
        // Must use original input BB since base may not be Instruction
        // The cast is needed since base traversal may strip away bitcasts
        if (base->getType() != basephi->getType()) {
          base = new BitCastInst(base, basephi->getType(), "cast",
                                 InBB->getTerminator());
          NewInsertedDefs.insert(base);
        }
        basephi->addIncoming(base, InBB);
      }
      assert(basephi->getNumIncomingValues() == NumPHIValues);
    } else {
      SelectInst *basesel = cast<SelectInst>(state.getBase());
      SelectInst *sel = cast<SelectInst>(v);
      // Operand 1 & 2 are true, false path respectively. TODO: refactor to
      // something more safe and less hacky.
      for (int i = 1; i <= 2; i++) {
        Value *InVal = sel->getOperand(i);
        // Find either the defining value for the PHI or the normal base for
        // a non-phi node
        Value *base = findBaseOrBDV(InVal, cache);
        if (!isKnownBaseResult(base)) {
          // Either conflict or base.
          assert(states.count(base));
          base = states[base].getBase();
          assert(base != nullptr && "unknown PhiState!");
        }
        assert(base && "can't be null");
        // Must use original input BB since base may not be Instruction
        // The cast is needed since base traversal may strip away bitcasts
        if (base->getType() != basesel->getType()) {
          base = new BitCastInst(base, basesel->getType(), "cast", basesel);
          NewInsertedDefs.insert(base);
        }
        basesel->setOperand(i, base);
      }
    }
  }

  // Cache all of our results so we can cheaply reuse them
  // NOTE: This is actually two caches: one of the base defining value
  // relation and one of the base pointer relation!  FIXME
  for (auto item : states) {
    Value *v = item.first;
    Value *base = item.second.getBase();
    assert(v && base);
    assert(!isKnownBaseResult(v) && "why did it get added?");

    if (TraceLSP) {
      std::string fromstr =
          cache.count(v) ? (cache[v]->hasName() ? cache[v]->getName() : "")
                         : "none";
      errs() << "Updating base value cache"
             << " for: " << (v->hasName() ? v->getName() : "")
             << " from: " << fromstr
             << " to: " << (base->hasName() ? base->getName() : "") << "\n";
    }

    assert(isKnownBaseResult(base) &&
           "must be something we 'know' is a base pointer");
    if (cache.count(v)) {
      // Once we transition from the BDV relation being store in the cache to
      // the base relation being stored, it must be stable
      assert((!isKnownBaseResult(cache[v]) || cache[v] == base) &&
             "base relation should be stable");
    }
    cache[v] = base;
  }
  assert(cache.find(def) != cache.end());
  return cache[def];
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
static void findBasePointers(const StatepointLiveSetTy &live,
                             DenseMap<llvm::Value *, llvm::Value *> &PointerToBase,
                             DominatorTree *DT, DefiningValueMapTy &DVCache,
                             DenseSet<llvm::Value *> &NewInsertedDefs) {
  // For the naming of values inserted to be deterministic - which makes for
  // much cleaner and more stable tests - we need to assign an order to the
  // live values.  DenseSets do not provide a deterministic order across runs.
  SmallVector<Value*, 64> Temp;
  Temp.insert(Temp.end(), live.begin(), live.end());
  std::sort(Temp.begin(), Temp.end(), order_by_name);
  for (Value *ptr : Temp) {
    Value *base = findBasePointer(ptr, DVCache, NewInsertedDefs);
    assert(base && "failed to find base pointer");
    PointerToBase[ptr] = base;
    assert((!isa<Instruction>(base) || !isa<Instruction>(ptr) ||
            DT->dominates(cast<Instruction>(base)->getParent(),
                          cast<Instruction>(ptr)->getParent())) &&
           "The base we found better dominate the derived pointer");

    // If you see this trip and like to live really dangerously, the code should
    // be correct, just with idioms the verifier can't handle.  You can try
    // disabling the verifier at your own substaintial risk.
    assert(!isNullConstant(base) && "the relocation code needs adjustment to "
                                    "handle the relocation of a null pointer "
                                    "constant without causing false positives "
                                    "in the safepoint ir verifier.");
  }
}

/// Find the required based pointers (and adjust the live set) for the given
/// parse point.
static void findBasePointers(DominatorTree &DT, DefiningValueMapTy &DVCache,
                             const CallSite &CS,
                             PartiallyConstructedSafepointRecord &result) {
  DenseMap<llvm::Value *, llvm::Value *> PointerToBase;
  DenseSet<llvm::Value *> NewInsertedDefs;
  findBasePointers(result.liveset, PointerToBase, &DT, DVCache, NewInsertedDefs);

  if (PrintBasePointers) {
    // Note: Need to print these in a stable order since this is checked in
    // some tests.
    errs() << "Base Pairs (w/o Relocation):\n";
    SmallVector<Value*, 64> Temp;
    Temp.reserve(PointerToBase.size());
    for (auto Pair : PointerToBase) {
      Temp.push_back(Pair.first);
    }
    std::sort(Temp.begin(), Temp.end(), order_by_name);
    for (Value *Ptr : Temp) {
      Value *Base = PointerToBase[Ptr];
      errs() << " derived %" << Ptr->getName() << " base %"
             << Base->getName() << "\n";
    }
  }

  result.PointerToBase = PointerToBase;
  result.NewInsertedDefs = NewInsertedDefs;
}

/// Check for liveness of items in the insert defs and add them to the live
/// and base pointer sets
static void fixupLiveness(DominatorTree &DT, const CallSite &CS,
                          const DenseSet<Value *> &allInsertedDefs,
                          PartiallyConstructedSafepointRecord &result) {
  Instruction *inst = CS.getInstruction();

  auto liveset = result.liveset;
  auto PointerToBase = result.PointerToBase;

  auto is_live_gc_reference =
      [&](Value &V) { return isLiveGCReferenceAt(V, inst, DT, nullptr); };

  // For each new definition, check to see if a) the definition dominates the
  // instruction we're interested in, and b) one of the uses of that definition
  // is edge-reachable from the instruction we're interested in.  This is the
  // same definition of liveness we used in the intial liveness analysis
  for (Value *newDef : allInsertedDefs) {
    if (liveset.count(newDef)) {
      // already live, no action needed
      continue;
    }

    // PERF: Use DT to check instruction domination might not be good for
    // compilation time, and we could change to optimal solution if this
    // turn to be a issue
    if (!DT.dominates(cast<Instruction>(newDef), inst)) {
      // can't possibly be live at inst
      continue;
    }

    if (is_live_gc_reference(*newDef)) {
      // Add the live new defs into liveset and PointerToBase
      liveset.insert(newDef);
      PointerToBase[newDef] = newDef;
    }
  }

  result.liveset = liveset;
  result.PointerToBase = PointerToBase;
}

static void fixupLiveReferences(
    Function &F, DominatorTree &DT, Pass *P,
    const DenseSet<llvm::Value *> &allInsertedDefs,
    ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    fixupLiveness(DT, CS, allInsertedDefs, info);
  }
}

// Normalize basic block to make it ready to be target of invoke statepoint.
// It means spliting it to have single predecessor. Return newly created BB
// ready to be successor of invoke statepoint.
static BasicBlock *normalizeBBForInvokeSafepoint(BasicBlock *BB,
                                                 BasicBlock *InvokeParent,
                                                 Pass *P) {
  BasicBlock *ret = BB;

  if (!BB->getUniquePredecessor()) {
    ret = SplitBlockPredecessors(BB, InvokeParent, "");
  }

  // Another requirement for such basic blocks is to not have any phi nodes.
  // Since we just ensured that new BB will have single predecessor,
  // all phi nodes in it will have one value. Here it would be naturall place
  // to
  // remove them all. But we can not do this because we are risking to remove
  // one of the values stored in liveset of another statepoint. We will do it
  // later after placing all safepoints.

  return ret;
}

static int find_index(ArrayRef<Value *> livevec, Value *val) {
  auto itr = std::find(livevec.begin(), livevec.end(), val);
  assert(livevec.end() != itr);
  size_t index = std::distance(livevec.begin(), itr);
  assert(index < livevec.size());
  return index;
}

// Create new attribute set containing only attributes which can be transfered
// from original call to the safepoint.
static AttributeSet legalizeCallAttributes(AttributeSet AS) {
  AttributeSet ret;

  for (unsigned Slot = 0; Slot < AS.getNumSlots(); Slot++) {
    unsigned index = AS.getSlotIndex(Slot);

    if (index == AttributeSet::ReturnIndex ||
        index == AttributeSet::FunctionIndex) {

      for (auto it = AS.begin(Slot), it_end = AS.end(Slot); it != it_end;
           ++it) {
        Attribute attr = *it;

        // Do not allow certain attributes - just skip them
        // Safepoint can not be read only or read none.
        if (attr.hasAttribute(Attribute::ReadNone) ||
            attr.hasAttribute(Attribute::ReadOnly))
          continue;

        ret = ret.addAttributes(
            AS.getContext(), index,
            AttributeSet::get(AS.getContext(), index, AttrBuilder(attr)));
      }
    }

    // Just skip parameter attributes for now
  }

  return ret;
}

/// Helper function to place all gc relocates necessary for the given
/// statepoint.
/// Inputs:
///   liveVariables - list of variables to be relocated.
///   liveStart - index of the first live variable.
///   basePtrs - base pointers.
///   statepointToken - statepoint instruction to which relocates should be
///   bound.
///   Builder - Llvm IR builder to be used to construct new calls.
void CreateGCRelocates(ArrayRef<llvm::Value *> liveVariables,
                       const int liveStart,
                       ArrayRef<llvm::Value *> basePtrs,
                       Instruction *statepointToken, IRBuilder<> Builder) {

  SmallVector<Instruction *, 64> NewDefs;
  NewDefs.reserve(liveVariables.size());

  Module *M = statepointToken->getParent()->getParent()->getParent();

  for (unsigned i = 0; i < liveVariables.size(); i++) {
    // We generate a (potentially) unique declaration for every pointer type
    // combination.  This results is some blow up the function declarations in
    // the IR, but removes the need for argument bitcasts which shrinks the IR
    // greatly and makes it much more readable.
    SmallVector<Type *, 1> types;                    // one per 'any' type
    types.push_back(liveVariables[i]->getType()); // result type
    Value *gc_relocate_decl = Intrinsic::getDeclaration(
        M, Intrinsic::experimental_gc_relocate, types);

    // Generate the gc.relocate call and save the result
    Value *baseIdx =
        ConstantInt::get(Type::getInt32Ty(M->getContext()),
                         liveStart + find_index(liveVariables, basePtrs[i]));
    Value *liveIdx = ConstantInt::get(
        Type::getInt32Ty(M->getContext()),
        liveStart + find_index(liveVariables, liveVariables[i]));

    // only specify a debug name if we can give a useful one
    Value *reloc = Builder.CreateCall3(
        gc_relocate_decl, statepointToken, baseIdx, liveIdx,
        liveVariables[i]->hasName() ? liveVariables[i]->getName() + ".relocated"
                                    : "");
    // Trick CodeGen into thinking there are lots of free registers at this
    // fake call.
    cast<CallInst>(reloc)->setCallingConv(CallingConv::Cold);

    NewDefs.push_back(cast<Instruction>(reloc));
  }
  assert(NewDefs.size() == liveVariables.size() &&
         "missing or extra redefinition at safepoint");
}

static void
makeStatepointExplicitImpl(const CallSite &CS, /* to replace */
                           const SmallVectorImpl<llvm::Value *> &basePtrs,
                           const SmallVectorImpl<llvm::Value *> &liveVariables,
                           Pass *P,
                           PartiallyConstructedSafepointRecord &result) {
  assert(basePtrs.size() == liveVariables.size());
  assert(isStatepoint(CS) &&
         "This method expects to be rewriting a statepoint");

  BasicBlock *BB = CS.getInstruction()->getParent();
  assert(BB);
  Function *F = BB->getParent();
  assert(F && "must be set");
  Module *M = F->getParent();
  (void)M;
  assert(M && "must be set");

  // We're not changing the function signature of the statepoint since the gc
  // arguments go into the var args section.
  Function *gc_statepoint_decl = CS.getCalledFunction();

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  Instruction *insertBefore = CS.getInstruction();
  IRBuilder<> Builder(insertBefore);
  // Copy all of the arguments from the original statepoint - this includes the
  // target, call args, and deopt args
  SmallVector<llvm::Value *, 64> args;
  args.insert(args.end(), CS.arg_begin(), CS.arg_end());
  // TODO: Clear the 'needs rewrite' flag

  // add all the pointers to be relocated (gc arguments)
  // Capture the start of the live variable list for use in the gc_relocates
  const int live_start = args.size();
  args.insert(args.end(), liveVariables.begin(), liveVariables.end());

  // Create the statepoint given all the arguments
  Instruction *token = nullptr;
  AttributeSet return_attributes;
  if (CS.isCall()) {
    CallInst *toReplace = cast<CallInst>(CS.getInstruction());
    CallInst *call =
        Builder.CreateCall(gc_statepoint_decl, args, "safepoint_token");
    call->setTailCall(toReplace->isTailCall());
    call->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of sttributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    call->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    BasicBlock::iterator next(toReplace);
    assert(BB->end() != next && "not a terminator, must have next");
    next++;
    Instruction *IP = &*(next);
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(IP->getDebugLoc());

  } else {
    InvokeInst *toReplace = cast<InvokeInst>(CS.getInstruction());

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    InvokeInst *invoke = InvokeInst::Create(
        gc_statepoint_decl, toReplace->getNormalDest(),
        toReplace->getUnwindDest(), args, "", toReplace->getParent());
    invoke->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of sttributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    invoke->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = invoke;

    // Generate gc relocates in exceptional path
    BasicBlock *unwindBlock = normalizeBBForInvokeSafepoint(
        toReplace->getUnwindDest(), invoke->getParent(), P);

    Instruction *IP = &*(unwindBlock->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(toReplace->getDebugLoc());

    // Extract second element from landingpad return value. We will attach
    // exceptional gc relocates to it.
    const unsigned idx = 1;
    Instruction *exceptional_token =
        cast<Instruction>(Builder.CreateExtractValue(
            unwindBlock->getLandingPadInst(), idx, "relocate_token"));
    result.UnwindToken = exceptional_token;

    // Just throw away return value. We will use the one we got for normal
    // block.
    (void)CreateGCRelocates(liveVariables, live_start, basePtrs,
                            exceptional_token, Builder);

    // Generate gc relocates and returns for normal block
    BasicBlock *normalDest = normalizeBBForInvokeSafepoint(
        toReplace->getNormalDest(), invoke->getParent(), P);

    IP = &*(normalDest->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);

    // gc relocates will be generated later as if it were regular call
    // statepoint
  }
  assert(token);

  // Take the name of the original value call if it had one.
  token->takeName(CS.getInstruction());

  // The GCResult is already inserted, we just need to find it
#ifndef NDEBUG
  Instruction *toReplace = CS.getInstruction();
  assert((toReplace->hasNUses(0) || toReplace->hasNUses(1)) &&
         "only valid use before rewrite is gc.result");
  assert(!toReplace->hasOneUse() ||
         isGCResult(cast<Instruction>(*toReplace->user_begin())));
#endif

  // Update the gc.result of the original statepoint (if any) to use the newly
  // inserted statepoint.  This is safe to do here since the token can't be
  // considered a live reference.
  CS.getInstruction()->replaceAllUsesWith(token);

  result.StatepointToken = token;

  // Second, create a gc.relocate for every live variable
  CreateGCRelocates(liveVariables, live_start, basePtrs, token, Builder);

}

namespace {
struct name_ordering {
  Value *base;
  Value *derived;
  bool operator()(name_ordering const &a, name_ordering const &b) {
    return -1 == a.derived->getName().compare(b.derived->getName());
  }
};
}
static void stablize_order(SmallVectorImpl<Value *> &basevec,
                           SmallVectorImpl<Value *> &livevec) {
  assert(basevec.size() == livevec.size());

  SmallVector<name_ordering, 64> temp;
  for (size_t i = 0; i < basevec.size(); i++) {
    name_ordering v;
    v.base = basevec[i];
    v.derived = livevec[i];
    temp.push_back(v);
  }
  std::sort(temp.begin(), temp.end(), name_ordering());
  for (size_t i = 0; i < basevec.size(); i++) {
    basevec[i] = temp[i].base;
    livevec[i] = temp[i].derived;
  }
}

// Replace an existing gc.statepoint with a new one and a set of gc.relocates
// which make the relocations happening at this safepoint explicit.
// 
// WARNING: Does not do any fixup to adjust users of the original live
// values.  That's the callers responsibility.
static void
makeStatepointExplicit(DominatorTree &DT, const CallSite &CS, Pass *P,
                       PartiallyConstructedSafepointRecord &result) {
  auto liveset = result.liveset;
  auto PointerToBase = result.PointerToBase;

  // Convert to vector for efficient cross referencing.
  SmallVector<Value *, 64> basevec, livevec;
  livevec.reserve(liveset.size());
  basevec.reserve(liveset.size());
  for (Value *L : liveset) {
    livevec.push_back(L);

    assert(PointerToBase.find(L) != PointerToBase.end());
    Value *base = PointerToBase[L];
    basevec.push_back(base);
  }
  assert(livevec.size() == basevec.size());

  // To make the output IR slightly more stable (for use in diffs), ensure a
  // fixed order of the values in the safepoint (by sorting the value name).
  // The order is otherwise meaningless.
  stablize_order(basevec, livevec);

  // Do the actual rewriting and delete the old statepoint
  makeStatepointExplicitImpl(CS, basevec, livevec, P, result);
  CS.getInstruction()->eraseFromParent();
}

// Helper function for the relocationViaAlloca.
// It receives iterator to the statepoint gc relocates and emits store to the
// assigned
// location (via allocaMap) for the each one of them.
// Add visited values into the visitedLiveValues set we will later use them
// for sanity check.
static void
insertRelocationStores(iterator_range<Value::user_iterator> gcRelocs,
                       DenseMap<Value *, Value *> &allocaMap,
                       DenseSet<Value *> &visitedLiveValues) {

  for (User *U : gcRelocs) {
    if (!isa<IntrinsicInst>(U))
      continue;

    IntrinsicInst *relocatedValue = cast<IntrinsicInst>(U);

    // We only care about relocates
    if (relocatedValue->getIntrinsicID() !=
        Intrinsic::experimental_gc_relocate) {
      continue;
    }

    GCRelocateOperands relocateOperands(relocatedValue);
    Value *originalValue = const_cast<Value *>(relocateOperands.derivedPtr());
    assert(allocaMap.count(originalValue));
    Value *alloca = allocaMap[originalValue];

    // Emit store into the related alloca
    StoreInst *store = new StoreInst(relocatedValue, alloca);
    store->insertAfter(relocatedValue);

#ifndef NDEBUG
    visitedLiveValues.insert(originalValue);
#endif
  }
}

/// do all the relocation update via allocas and mem2reg
static void relocationViaAlloca(
    Function &F, DominatorTree &DT, ArrayRef<Value *> live,
    ArrayRef<struct PartiallyConstructedSafepointRecord> records) {
#ifndef NDEBUG
  int initialAllocaNum = 0;

  // record initial number of allocas
  for (inst_iterator itr = inst_begin(F), end = inst_end(F); itr != end;
       itr++) {
    if (isa<AllocaInst>(*itr))
      initialAllocaNum++;
  }
#endif

  // TODO-PERF: change data structures, reserve
  DenseMap<Value *, Value *> allocaMap;
  SmallVector<AllocaInst *, 200> PromotableAllocas;
  PromotableAllocas.reserve(live.size());

  // emit alloca for each live gc pointer
  for (unsigned i = 0; i < live.size(); i++) {
    Value *liveValue = live[i];
    AllocaInst *alloca = new AllocaInst(liveValue->getType(), "",
                                        F.getEntryBlock().getFirstNonPHI());
    allocaMap[liveValue] = alloca;
    PromotableAllocas.push_back(alloca);
  }

  // The next two loops are part of the same conceptual operation.  We need to
  // insert a store to the alloca after the original def and at each
  // redefinition.  We need to insert a load before each use.  These are split
  // into distinct loops for performance reasons.

  // update gc pointer after each statepoint
  // either store a relocated value or null (if no relocated value found for
  // this gc pointer and it is not a gc_result)
  // this must happen before we update the statepoint with load of alloca
  // otherwise we lose the link between statepoint and old def
  for (size_t i = 0; i < records.size(); i++) {
    const struct PartiallyConstructedSafepointRecord &info = records[i];
    Value *Statepoint = info.StatepointToken;

    // This will be used for consistency check
    DenseSet<Value *> visitedLiveValues;

    // Insert stores for normal statepoint gc relocates
    insertRelocationStores(Statepoint->users(), allocaMap, visitedLiveValues);

    // In case if it was invoke statepoint
    // we will insert stores for exceptional path gc relocates.
    if (isa<InvokeInst>(Statepoint)) {
      insertRelocationStores(info.UnwindToken->users(),
                             allocaMap, visitedLiveValues);
    }

#ifndef NDEBUG
    // As a debuging aid, pretend that an unrelocated pointer becomes null at
    // the gc.statepoint.  This will turn some subtle GC problems into slightly
    // easier to debug SEGVs
    SmallVector<AllocaInst *, 64> ToClobber;
    for (auto Pair : allocaMap) {
      Value *Def = Pair.first;
      AllocaInst *Alloca = cast<AllocaInst>(Pair.second);

      // This value was relocated
      if (visitedLiveValues.count(Def)) {
        continue;
      }
      ToClobber.push_back(Alloca);
    }

    auto InsertClobbersAt = [&](Instruction *IP) {
      for (auto *AI : ToClobber) {
        auto AIType = cast<PointerType>(AI->getType());
        auto PT = cast<PointerType>(AIType->getElementType());
        Constant *CPN = ConstantPointerNull::get(PT);
        StoreInst *store = new StoreInst(CPN, AI);
        store->insertBefore(IP);
      }
    };

    // Insert the clobbering stores.  These may get intermixed with the
    // gc.results and gc.relocates, but that's fine.  
    if (auto II = dyn_cast<InvokeInst>(Statepoint)) {
      InsertClobbersAt(II->getNormalDest()->getFirstInsertionPt());
      InsertClobbersAt(II->getUnwindDest()->getFirstInsertionPt());
    } else {
      BasicBlock::iterator Next(cast<CallInst>(Statepoint));
      Next++;
      InsertClobbersAt(Next);
    }
#endif
  }
  // update use with load allocas and add store for gc_relocated
  for (auto Pair : allocaMap) {
    Value *def = Pair.first;
    Value *alloca = Pair.second;

    // we pre-record the uses of allocas so that we dont have to worry about
    // later update
    // that change the user information.
    SmallVector<Instruction *, 20> uses;
    // PERF: trade a linear scan for repeated reallocation
    uses.reserve(std::distance(def->user_begin(), def->user_end()));
    for (User *U : def->users()) {
      if (!isa<ConstantExpr>(U)) {
        // If the def has a ConstantExpr use, then the def is either a
        // ConstantExpr use itself or null.  In either case
        // (recursively in the first, directly in the second), the oop
        // it is ultimately dependent on is null and this particular
        // use does not need to be fixed up.
        uses.push_back(cast<Instruction>(U));
      }
    }

    std::sort(uses.begin(), uses.end());
    auto last = std::unique(uses.begin(), uses.end());
    uses.erase(last, uses.end());

    for (Instruction *use : uses) {
      if (isa<PHINode>(use)) {
        PHINode *phi = cast<PHINode>(use);
        for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
          if (def == phi->getIncomingValue(i)) {
            LoadInst *load = new LoadInst(
                alloca, "", phi->getIncomingBlock(i)->getTerminator());
            phi->setIncomingValue(i, load);
          }
        }
      } else {
        LoadInst *load = new LoadInst(alloca, "", use);
        use->replaceUsesOfWith(def, load);
      }
    }

    // emit store for the initial gc value
    // store must be inserted after load, otherwise store will be in alloca's
    // use list and an extra load will be inserted before it
    StoreInst *store = new StoreInst(def, alloca);
    if (Instruction *inst = dyn_cast<Instruction>(def)) {
      if (InvokeInst *invoke = dyn_cast<InvokeInst>(inst)) {
        // InvokeInst is a TerminatorInst so the store need to be inserted
        // into its normal destination block.
        BasicBlock *normalDest = invoke->getNormalDest();
        store->insertBefore(normalDest->getFirstNonPHI());
      } else {
        assert(!inst->isTerminator() &&
               "The only TerminatorInst that can produce a value is "
               "InvokeInst which is handled above.");
         store->insertAfter(inst);
      }
    } else {
      assert((isa<Argument>(def) || isa<GlobalVariable>(def) ||
              (isa<Constant>(def) && cast<Constant>(def)->isNullValue())) &&
             "Must be argument or global");
      store->insertAfter(cast<Instruction>(alloca));
    }
  }

  assert(PromotableAllocas.size() == live.size() &&
         "we must have the same allocas with lives");
  if (!PromotableAllocas.empty()) {
    // apply mem2reg to promote alloca to SSA
    PromoteMemToReg(PromotableAllocas, DT);
  }

#ifndef NDEBUG
  for (inst_iterator itr = inst_begin(F), end = inst_end(F); itr != end;
       itr++) {
    if (isa<AllocaInst>(*itr))
      initialAllocaNum--;
  }
  assert(initialAllocaNum == 0 && "We must not introduce any extra allocas");
#endif
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(SmallVectorImpl<T> &Vec) {
  DenseSet<T> Seen;
  SmallVector<T, 128> TempVec;
  TempVec.reserve(Vec.size());
  for (auto Element : Vec)
    TempVec.push_back(Element);
  Vec.clear();
  for (auto V : TempVec) {
    if (Seen.insert(V).second) {
      Vec.push_back(V);
    }
  }
}

static Function *getUseHolder(Module &M) {
  FunctionType *ftype =
      FunctionType::get(Type::getVoidTy(M.getContext()), true);
  Function *Func = cast<Function>(M.getOrInsertFunction("__tmp_use", ftype));
  return Func;
}

/// Insert holders so that each Value is obviously live through the entire
/// liftetime of the call.
static void insertUseHolderAfter(CallSite &CS, const ArrayRef<Value *> Values,
                                 SmallVectorImpl<CallInst *> &holders) {
  Module *M = CS.getInstruction()->getParent()->getParent()->getParent();
  Function *Func = getUseHolder(*M);
  if (CS.isCall()) {
    // For call safepoints insert dummy calls right after safepoint
    BasicBlock::iterator next(CS.getInstruction());
    next++;
    CallInst *base_holder = CallInst::Create(Func, Values, "", next);
    holders.push_back(base_holder);
  } else if (CS.isInvoke()) {
    // For invoke safepooints insert dummy calls both in normal and
    // exceptional destination blocks
    InvokeInst *invoke = cast<InvokeInst>(CS.getInstruction());
    CallInst *normal_holder = CallInst::Create(
        Func, Values, "", invoke->getNormalDest()->getFirstInsertionPt());
    CallInst *unwind_holder = CallInst::Create(
        Func, Values, "", invoke->getUnwindDest()->getFirstInsertionPt());
    holders.push_back(normal_holder);
    holders.push_back(unwind_holder);
  } else
    llvm_unreachable("unsupported call type");
}

static void findLiveReferences(
    Function &F, DominatorTree &DT, Pass *P, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    analyzeParsePointLiveness(DT, CS, info);
  }
}

static void addBasesAsLiveValues(StatepointLiveSetTy &liveset,
                                 DenseMap<Value *, Value *> &PointerToBase) {
  // Identify any base pointers which are used in this safepoint, but not
  // themselves relocated.  We need to relocate them so that later inserted
  // safepoints can get the properly relocated base register.
  DenseSet<Value *> missing;
  for (Value *L : liveset) {
    assert(PointerToBase.find(L) != PointerToBase.end());
    Value *base = PointerToBase[L];
    assert(base);
    if (liveset.find(base) == liveset.end()) {
      assert(PointerToBase.find(base) == PointerToBase.end());
      // uniqued by set insert
      missing.insert(base);
    }
  }

  // Note that we want these at the end of the list, otherwise
  // register placement gets screwed up once we lower to STATEPOINT
  // instructions.  This is an utter hack, but there doesn't seem to be a
  // better one.
  for (Value *base : missing) {
    assert(base);
    liveset.insert(base);
    PointerToBase[base] = base;
  }
  assert(liveset.size() == PointerToBase.size());
}

static bool insertParsePoints(Function &F, DominatorTree &DT, Pass *P,
                              SmallVectorImpl<CallSite> &toUpdate) {
#ifndef NDEBUG
  // sanity check the input
  std::set<CallSite> uniqued;
  uniqued.insert(toUpdate.begin(), toUpdate.end());
  assert(uniqued.size() == toUpdate.size() && "no duplicates please!");

  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    assert(CS.getInstruction()->getParent()->getParent() == &F);
    assert(isStatepoint(CS) && "expected to already be a deopt statepoint");
  }
#endif

  // A list of dummy calls added to the IR to keep various values obviously
  // live in the IR.  We'll remove all of these when done.
  SmallVector<CallInst *, 64> holders;

  // Insert a dummy call with all of the arguments to the vm_state we'll need
  // for the actual safepoint insertion.  This ensures reference arguments in
  // the deopt argument list are considered live through the safepoint (and
  // thus makes sure they get relocated.)
  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    Statepoint StatepointCS(CS);

    SmallVector<Value *, 64> DeoptValues;
    for (Use &U : StatepointCS.vm_state_args()) {
      Value *Arg = cast<Value>(&U);
      if (isGCPointerType(Arg->getType()))
        DeoptValues.push_back(Arg);
    }
    insertUseHolderAfter(CS, DeoptValues, holders);
  }

  SmallVector<struct PartiallyConstructedSafepointRecord, 64> records;
  records.reserve(toUpdate.size());
  for (size_t i = 0; i < toUpdate.size(); i++) {
    struct PartiallyConstructedSafepointRecord info;
    records.push_back(info);
  }
  assert(records.size() == toUpdate.size());

  // A) Identify all gc pointers which are staticly live at the given call
  // site.
  findLiveReferences(F, DT, P, toUpdate, records);

  // B) Find the base pointers for each live pointer
  /* scope for caching */ {
    // Cache the 'defining value' relation used in the computation and
    // insertion of base phis and selects.  This ensures that we don't insert
    // large numbers of duplicate base_phis.
    DefiningValueMapTy DVCache;

    for (size_t i = 0; i < records.size(); i++) {
      struct PartiallyConstructedSafepointRecord &info = records[i];
      CallSite &CS = toUpdate[i];
      findBasePointers(DT, DVCache, CS, info);
    }
  } // end of cache scope

  // The base phi insertion logic (for any safepoint) may have inserted new
  // instructions which are now live at some safepoint.  The simplest such
  // example is:
  // loop:
  //   phi a  <-- will be a new base_phi here
  //   safepoint 1 <-- that needs to be live here
  //   gep a + 1
  //   safepoint 2
  //   br loop
  DenseSet<llvm::Value *> allInsertedDefs;
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    allInsertedDefs.insert(info.NewInsertedDefs.begin(),
                           info.NewInsertedDefs.end());
  }

  // We insert some dummy calls after each safepoint to definitely hold live
  // the base pointers which were identified for that safepoint.  We'll then
  // ask liveness for _every_ base inserted to see what is now live.  Then we
  // remove the dummy calls.
  holders.reserve(holders.size() + records.size());
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];

    SmallVector<Value *, 128> Bases;
    for (auto Pair : info.PointerToBase) {
      Bases.push_back(Pair.second);
    }
    insertUseHolderAfter(CS, Bases, holders);
  }

  // Add the bases explicitly to the live vector set.  This may result in a few
  // extra relocations, but the base has to be available whenever a pointer
  // derived from it is used.  Thus, we need it to be part of the statepoint's
  // gc arguments list.  TODO: Introduce an explicit notion (in the following
  // code) of the GC argument list as seperate from the live Values at a
  // given statepoint.
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    addBasesAsLiveValues(info.liveset, info.PointerToBase);
  }

  // If we inserted any new values, we need to adjust our notion of what is
  // live at a particular safepoint.
  if (!allInsertedDefs.empty()) {
    fixupLiveReferences(F, DT, P, allInsertedDefs, toUpdate, records);
  }
  if (PrintBasePointers) {
    for (size_t i = 0; i < records.size(); i++) {
      struct PartiallyConstructedSafepointRecord &info = records[i];
      errs() << "Base Pairs: (w/Relocation)\n";
      for (auto Pair : info.PointerToBase) {
        errs() << " derived %" << Pair.first->getName() << " base %"
               << Pair.second->getName() << "\n";
      }
    }
  }
  for (size_t i = 0; i < holders.size(); i++) {
    holders[i]->eraseFromParent();
    holders[i] = nullptr;
  }
  holders.clear();

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];
    makeStatepointExplicit(DT, CS, P, info);
  }
  toUpdate.clear(); // prevent accident use of invalid CallSites

  // In case if we inserted relocates in a different basic block than the
  // original safepoint (this can happen for invokes). We need to be sure that
  // original values were not used in any of the phi nodes at the
  // beginning of basic block containing them. Because we know that all such
  // blocks will have single predecessor we can safely assume that all phi
  // nodes have single entry (because of normalizeBBForInvokeSafepoint).
  // Just remove them all here.
  for (size_t i = 0; i < records.size(); i++) {
    Instruction *I = records[i].StatepointToken;

    if (InvokeInst *invoke = dyn_cast<InvokeInst>(I)) {
      FoldSingleEntryPHINodes(invoke->getNormalDest());
      assert(!isa<PHINode>(invoke->getNormalDest()->begin()));

      FoldSingleEntryPHINodes(invoke->getUnwindDest());
      assert(!isa<PHINode>(invoke->getUnwindDest()->begin()));
    }
  }

  // Do all the fixups of the original live variables to their relocated selves
  SmallVector<Value *, 128> live;
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    // We can't simply save the live set from the original insertion.  One of
    // the live values might be the result of a call which needs a safepoint.
    // That Value* no longer exists and we need to use the new gc_result.
    // Thankfully, the liveset is embedded in the statepoint (and updated), so
    // we just grab that.
    Statepoint statepoint(info.StatepointToken);
    live.insert(live.end(), statepoint.gc_args_begin(),
                statepoint.gc_args_end());
  }
  unique_unsorted(live);

#ifndef NDEBUG
  // sanity check
  for (auto ptr : live) {
    assert(isGCPointerType(ptr->getType()) && "must be a gc pointer type");
  }
#endif

  relocationViaAlloca(F, DT, live, records);
  return !records.empty();
}

/// Returns true if this function should be rewritten by this pass.  The main
/// point of this function is as an extension point for custom logic.
static bool shouldRewriteStatepointsIn(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const std::string StatepointExampleName("statepoint-example");
    return StatepointExampleName == F.getGC();
  } else
    return false;
}

bool RewriteStatepointsForGC::runOnFunction(Function &F) {
  // Nothing to do for declarations.
  if (F.isDeclaration() || F.empty())
    return false;

  // Policy choice says not to rewrite - the most common reason is that we're
  // compiling code without a GCStrategy.
  if (!shouldRewriteStatepointsIn(F))
    return false;

  // Gather all the statepoints which need rewritten.
  SmallVector<CallSite, 64> ParsePointNeeded;
  for (Instruction &I : inst_range(F)) {
    // TODO: only the ones with the flag set!
    if (isStatepoint(I))
      ParsePointNeeded.push_back(CallSite(&I));
  }

  // Return early if no work to do.
  if (ParsePointNeeded.empty())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return insertParsePoints(F, DT, this, ParsePointNeeded);
}
