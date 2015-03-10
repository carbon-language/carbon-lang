//===- LoopAccessAnalysis.cpp - Loop Access Analysis Implementation --------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The implementation for the loop memory dependence that was originally
// developed for the loop vectorizer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/VectorUtils.h"
using namespace llvm;

#define DEBUG_TYPE "loop-accesses"

static cl::opt<unsigned, true>
VectorizationFactor("force-vector-width", cl::Hidden,
                    cl::desc("Sets the SIMD width. Zero is autoselect."),
                    cl::location(VectorizerParams::VectorizationFactor));
unsigned VectorizerParams::VectorizationFactor;

static cl::opt<unsigned, true>
VectorizationInterleave("force-vector-interleave", cl::Hidden,
                        cl::desc("Sets the vectorization interleave count. "
                                 "Zero is autoselect."),
                        cl::location(
                            VectorizerParams::VectorizationInterleave));
unsigned VectorizerParams::VectorizationInterleave;

static cl::opt<unsigned, true> RuntimeMemoryCheckThreshold(
    "runtime-memory-check-threshold", cl::Hidden,
    cl::desc("When performing memory disambiguation checks at runtime do not "
             "generate more than this number of comparisons (default = 8)."),
    cl::location(VectorizerParams::RuntimeMemoryCheckThreshold), cl::init(8));
unsigned VectorizerParams::RuntimeMemoryCheckThreshold;

/// Maximum SIMD width.
const unsigned VectorizerParams::MaxVectorWidth = 64;

/// \brief We collect interesting dependences up to this threshold.
static cl::opt<unsigned> MaxInterestingDependence(
    "max-interesting-dependences", cl::Hidden,
    cl::desc("Maximum number of interesting dependences collected by "
             "loop-access analysis (default = 100)"),
    cl::init(100));

bool VectorizerParams::isInterleaveForced() {
  return ::VectorizationInterleave.getNumOccurrences() > 0;
}

void LoopAccessReport::emitAnalysis(const LoopAccessReport &Message,
                                    const Function *TheFunction,
                                    const Loop *TheLoop,
                                    const char *PassName) {
  DebugLoc DL = TheLoop->getStartLoc();
  if (const Instruction *I = Message.getInstr())
    DL = I->getDebugLoc();
  emitOptimizationRemarkAnalysis(TheFunction->getContext(), PassName,
                                 *TheFunction, DL, Message.str());
}

Value *llvm::stripIntegerCast(Value *V) {
  if (CastInst *CI = dyn_cast<CastInst>(V))
    if (CI->getOperand(0)->getType()->isIntegerTy())
      return CI->getOperand(0);
  return V;
}

const SCEV *llvm::replaceSymbolicStrideSCEV(ScalarEvolution *SE,
                                            const ValueToValueMap &PtrToStride,
                                            Value *Ptr, Value *OrigPtr) {

  const SCEV *OrigSCEV = SE->getSCEV(Ptr);

  // If there is an entry in the map return the SCEV of the pointer with the
  // symbolic stride replaced by one.
  ValueToValueMap::const_iterator SI =
      PtrToStride.find(OrigPtr ? OrigPtr : Ptr);
  if (SI != PtrToStride.end()) {
    Value *StrideVal = SI->second;

    // Strip casts.
    StrideVal = stripIntegerCast(StrideVal);

    // Replace symbolic stride by one.
    Value *One = ConstantInt::get(StrideVal->getType(), 1);
    ValueToValueMap RewriteMap;
    RewriteMap[StrideVal] = One;

    const SCEV *ByOne =
        SCEVParameterRewriter::rewrite(OrigSCEV, *SE, RewriteMap, true);
    DEBUG(dbgs() << "LAA: Replacing SCEV: " << *OrigSCEV << " by: " << *ByOne
                 << "\n");
    return ByOne;
  }

  // Otherwise, just return the SCEV of the original pointer.
  return SE->getSCEV(Ptr);
}

void LoopAccessInfo::RuntimePointerCheck::insert(
    ScalarEvolution *SE, Loop *Lp, Value *Ptr, bool WritePtr, unsigned DepSetId,
    unsigned ASId, const ValueToValueMap &Strides) {
  // Get the stride replaced scev.
  const SCEV *Sc = replaceSymbolicStrideSCEV(SE, Strides, Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Sc);
  assert(AR && "Invalid addrec expression");
  const SCEV *Ex = SE->getBackedgeTakenCount(Lp);
  const SCEV *ScEnd = AR->evaluateAtIteration(Ex, *SE);
  Pointers.push_back(Ptr);
  Starts.push_back(AR->getStart());
  Ends.push_back(ScEnd);
  IsWritePtr.push_back(WritePtr);
  DependencySetId.push_back(DepSetId);
  AliasSetId.push_back(ASId);
}

bool LoopAccessInfo::RuntimePointerCheck::needsChecking(
    unsigned I, unsigned J, const SmallVectorImpl<int> *PtrPartition) const {
  // No need to check if two readonly pointers intersect.
  if (!IsWritePtr[I] && !IsWritePtr[J])
    return false;

  // Only need to check pointers between two different dependency sets.
  if (DependencySetId[I] == DependencySetId[J])
    return false;

  // Only need to check pointers in the same alias set.
  if (AliasSetId[I] != AliasSetId[J])
    return false;

  // If PtrPartition is set omit checks between pointers of the same partition.
  // Partition number -1 means that the pointer is used in multiple partitions.
  // In this case we can't omit the check.
  if (PtrPartition && (*PtrPartition)[I] != -1 &&
      (*PtrPartition)[I] == (*PtrPartition)[J])
    return false;

  return true;
}

void LoopAccessInfo::RuntimePointerCheck::print(
    raw_ostream &OS, unsigned Depth,
    const SmallVectorImpl<int> *PtrPartition) const {
  unsigned NumPointers = Pointers.size();
  if (NumPointers == 0)
    return;

  OS.indent(Depth) << "Run-time memory checks:\n";
  unsigned N = 0;
  for (unsigned I = 0; I < NumPointers; ++I)
    for (unsigned J = I + 1; J < NumPointers; ++J)
      if (needsChecking(I, J, PtrPartition)) {
        OS.indent(Depth) << N++ << ":\n";
        OS.indent(Depth + 2) << *Pointers[I];
        if (PtrPartition)
          OS << " (Partition: " << (*PtrPartition)[I] << ")";
        OS << "\n";
        OS.indent(Depth + 2) << *Pointers[J];
        if (PtrPartition)
          OS << " (Partition: " << (*PtrPartition)[J] << ")";
        OS << "\n";
      }
}

namespace {
/// \brief Analyses memory accesses in a loop.
///
/// Checks whether run time pointer checks are needed and builds sets for data
/// dependence checking.
class AccessAnalysis {
public:
  /// \brief Read or write access location.
  typedef PointerIntPair<Value *, 1, bool> MemAccessInfo;
  typedef SmallPtrSet<MemAccessInfo, 8> MemAccessInfoSet;

  AccessAnalysis(const DataLayout &Dl, AliasAnalysis *AA,
                 MemoryDepChecker::DepCandidates &DA)
      : DL(Dl), AST(*AA), DepCands(DA), IsRTCheckNeeded(false) {}

  /// \brief Register a load  and whether it is only read from.
  void addLoad(AliasAnalysis::Location &Loc, bool IsReadOnly) {
    Value *Ptr = const_cast<Value*>(Loc.Ptr);
    AST.add(Ptr, AliasAnalysis::UnknownSize, Loc.AATags);
    Accesses.insert(MemAccessInfo(Ptr, false));
    if (IsReadOnly)
      ReadOnlyPtr.insert(Ptr);
  }

  /// \brief Register a store.
  void addStore(AliasAnalysis::Location &Loc) {
    Value *Ptr = const_cast<Value*>(Loc.Ptr);
    AST.add(Ptr, AliasAnalysis::UnknownSize, Loc.AATags);
    Accesses.insert(MemAccessInfo(Ptr, true));
  }

  /// \brief Check whether we can check the pointers at runtime for
  /// non-intersection.
  bool canCheckPtrAtRT(LoopAccessInfo::RuntimePointerCheck &RtCheck,
                       unsigned &NumComparisons, ScalarEvolution *SE,
                       Loop *TheLoop, const ValueToValueMap &Strides,
                       bool ShouldCheckStride = false);

  /// \brief Goes over all memory accesses, checks whether a RT check is needed
  /// and builds sets of dependent accesses.
  void buildDependenceSets() {
    processMemAccesses();
  }

  bool isRTCheckNeeded() { return IsRTCheckNeeded; }

  bool isDependencyCheckNeeded() { return !CheckDeps.empty(); }
  void resetDepChecks() { CheckDeps.clear(); }

  MemAccessInfoSet &getDependenciesToCheck() { return CheckDeps; }

private:
  typedef SetVector<MemAccessInfo> PtrAccessSet;

  /// \brief Go over all memory access and check whether runtime pointer checks
  /// are needed /// and build sets of dependency check candidates.
  void processMemAccesses();

  /// Set of all accesses.
  PtrAccessSet Accesses;

  const DataLayout &DL;

  /// Set of accesses that need a further dependence check.
  MemAccessInfoSet CheckDeps;

  /// Set of pointers that are read only.
  SmallPtrSet<Value*, 16> ReadOnlyPtr;

  /// An alias set tracker to partition the access set by underlying object and
  //intrinsic property (such as TBAA metadata).
  AliasSetTracker AST;

  /// Sets of potentially dependent accesses - members of one set share an
  /// underlying pointer. The set "CheckDeps" identfies which sets really need a
  /// dependence check.
  MemoryDepChecker::DepCandidates &DepCands;

  bool IsRTCheckNeeded;
};

} // end anonymous namespace

/// \brief Check whether a pointer can participate in a runtime bounds check.
static bool hasComputableBounds(ScalarEvolution *SE,
                                const ValueToValueMap &Strides, Value *Ptr) {
  const SCEV *PtrScev = replaceSymbolicStrideSCEV(SE, Strides, Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);
  if (!AR)
    return false;

  return AR->isAffine();
}

/// \brief Check the stride of the pointer and ensure that it does not wrap in
/// the address space.
static int isStridedPtr(ScalarEvolution *SE, Value *Ptr, const Loop *Lp,
                        const ValueToValueMap &StridesMap);

bool AccessAnalysis::canCheckPtrAtRT(
    LoopAccessInfo::RuntimePointerCheck &RtCheck, unsigned &NumComparisons,
    ScalarEvolution *SE, Loop *TheLoop, const ValueToValueMap &StridesMap,
    bool ShouldCheckStride) {
  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  bool CanDoRT = true;

  bool IsDepCheckNeeded = isDependencyCheckNeeded();
  NumComparisons = 0;

  // We assign a consecutive id to access from different alias sets.
  // Accesses between different groups doesn't need to be checked.
  unsigned ASId = 1;
  for (auto &AS : AST) {
    unsigned NumReadPtrChecks = 0;
    unsigned NumWritePtrChecks = 0;

    // We assign consecutive id to access from different dependence sets.
    // Accesses within the same set don't need a runtime check.
    unsigned RunningDepId = 1;
    DenseMap<Value *, unsigned> DepSetId;

    for (auto A : AS) {
      Value *Ptr = A.getValue();
      bool IsWrite = Accesses.count(MemAccessInfo(Ptr, true));
      MemAccessInfo Access(Ptr, IsWrite);

      if (IsWrite)
        ++NumWritePtrChecks;
      else
        ++NumReadPtrChecks;

      if (hasComputableBounds(SE, StridesMap, Ptr) &&
          // When we run after a failing dependency check we have to make sure
          // we don't have wrapping pointers.
          (!ShouldCheckStride ||
           isStridedPtr(SE, Ptr, TheLoop, StridesMap) == 1)) {
        // The id of the dependence set.
        unsigned DepId;

        if (IsDepCheckNeeded) {
          Value *Leader = DepCands.getLeaderValue(Access).getPointer();
          unsigned &LeaderId = DepSetId[Leader];
          if (!LeaderId)
            LeaderId = RunningDepId++;
          DepId = LeaderId;
        } else
          // Each access has its own dependence set.
          DepId = RunningDepId++;

        RtCheck.insert(SE, TheLoop, Ptr, IsWrite, DepId, ASId, StridesMap);

        DEBUG(dbgs() << "LAA: Found a runtime check ptr:" << *Ptr << '\n');
      } else {
        CanDoRT = false;
      }
    }

    if (IsDepCheckNeeded && CanDoRT && RunningDepId == 2)
      NumComparisons += 0; // Only one dependence set.
    else {
      NumComparisons += (NumWritePtrChecks * (NumReadPtrChecks +
                                              NumWritePtrChecks - 1));
    }

    ++ASId;
  }

  // If the pointers that we would use for the bounds comparison have different
  // address spaces, assume the values aren't directly comparable, so we can't
  // use them for the runtime check. We also have to assume they could
  // overlap. In the future there should be metadata for whether address spaces
  // are disjoint.
  unsigned NumPointers = RtCheck.Pointers.size();
  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i + 1; j < NumPointers; ++j) {
      // Only need to check pointers between two different dependency sets.
      if (RtCheck.DependencySetId[i] == RtCheck.DependencySetId[j])
       continue;
      // Only need to check pointers in the same alias set.
      if (RtCheck.AliasSetId[i] != RtCheck.AliasSetId[j])
        continue;

      Value *PtrI = RtCheck.Pointers[i];
      Value *PtrJ = RtCheck.Pointers[j];

      unsigned ASi = PtrI->getType()->getPointerAddressSpace();
      unsigned ASj = PtrJ->getType()->getPointerAddressSpace();
      if (ASi != ASj) {
        DEBUG(dbgs() << "LAA: Runtime check would require comparison between"
                       " different address spaces\n");
        return false;
      }
    }
  }

  return CanDoRT;
}

void AccessAnalysis::processMemAccesses() {
  // We process the set twice: first we process read-write pointers, last we
  // process read-only pointers. This allows us to skip dependence tests for
  // read-only pointers.

  DEBUG(dbgs() << "LAA: Processing memory accesses...\n");
  DEBUG(dbgs() << "  AST: "; AST.dump());
  DEBUG(dbgs() << "LAA:   Accesses(" << Accesses.size() << "):\n");
  DEBUG({
    for (auto A : Accesses)
      dbgs() << "\t" << *A.getPointer() << " (" <<
                (A.getInt() ? "write" : (ReadOnlyPtr.count(A.getPointer()) ?
                                         "read-only" : "read")) << ")\n";
  });

  // The AliasSetTracker has nicely partitioned our pointers by metadata
  // compatibility and potential for underlying-object overlap. As a result, we
  // only need to check for potential pointer dependencies within each alias
  // set.
  for (auto &AS : AST) {
    // Note that both the alias-set tracker and the alias sets themselves used
    // linked lists internally and so the iteration order here is deterministic
    // (matching the original instruction order within each set).

    bool SetHasWrite = false;

    // Map of pointers to last access encountered.
    typedef DenseMap<Value*, MemAccessInfo> UnderlyingObjToAccessMap;
    UnderlyingObjToAccessMap ObjToLastAccess;

    // Set of access to check after all writes have been processed.
    PtrAccessSet DeferredAccesses;

    // Iterate over each alias set twice, once to process read/write pointers,
    // and then to process read-only pointers.
    for (int SetIteration = 0; SetIteration < 2; ++SetIteration) {
      bool UseDeferred = SetIteration > 0;
      PtrAccessSet &S = UseDeferred ? DeferredAccesses : Accesses;

      for (auto AV : AS) {
        Value *Ptr = AV.getValue();

        // For a single memory access in AliasSetTracker, Accesses may contain
        // both read and write, and they both need to be handled for CheckDeps.
        for (auto AC : S) {
          if (AC.getPointer() != Ptr)
            continue;

          bool IsWrite = AC.getInt();

          // If we're using the deferred access set, then it contains only
          // reads.
          bool IsReadOnlyPtr = ReadOnlyPtr.count(Ptr) && !IsWrite;
          if (UseDeferred && !IsReadOnlyPtr)
            continue;
          // Otherwise, the pointer must be in the PtrAccessSet, either as a
          // read or a write.
          assert(((IsReadOnlyPtr && UseDeferred) || IsWrite ||
                  S.count(MemAccessInfo(Ptr, false))) &&
                 "Alias-set pointer not in the access set?");

          MemAccessInfo Access(Ptr, IsWrite);
          DepCands.insert(Access);

          // Memorize read-only pointers for later processing and skip them in
          // the first round (they need to be checked after we have seen all
          // write pointers). Note: we also mark pointer that are not
          // consecutive as "read-only" pointers (so that we check
          // "a[b[i]] +="). Hence, we need the second check for "!IsWrite".
          if (!UseDeferred && IsReadOnlyPtr) {
            DeferredAccesses.insert(Access);
            continue;
          }

          // If this is a write - check other reads and writes for conflicts. If
          // this is a read only check other writes for conflicts (but only if
          // there is no other write to the ptr - this is an optimization to
          // catch "a[i] = a[i] + " without having to do a dependence check).
          if ((IsWrite || IsReadOnlyPtr) && SetHasWrite) {
            CheckDeps.insert(Access);
            IsRTCheckNeeded = true;
          }

          if (IsWrite)
            SetHasWrite = true;

          // Create sets of pointers connected by a shared alias set and
          // underlying object.
          typedef SmallVector<Value *, 16> ValueVector;
          ValueVector TempObjects;
          GetUnderlyingObjects(Ptr, TempObjects, DL);
          for (Value *UnderlyingObj : TempObjects) {
            UnderlyingObjToAccessMap::iterator Prev =
                ObjToLastAccess.find(UnderlyingObj);
            if (Prev != ObjToLastAccess.end())
              DepCands.unionSets(Access, Prev->second);

            ObjToLastAccess[UnderlyingObj] = Access;
          }
        }
      }
    }
  }
}

static bool isInBoundsGep(Value *Ptr) {
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
    return GEP->isInBounds();
  return false;
}

/// \brief Check whether the access through \p Ptr has a constant stride.
static int isStridedPtr(ScalarEvolution *SE, Value *Ptr, const Loop *Lp,
                        const ValueToValueMap &StridesMap) {
  const Type *Ty = Ptr->getType();
  assert(Ty->isPointerTy() && "Unexpected non-ptr");

  // Make sure that the pointer does not point to aggregate types.
  const PointerType *PtrTy = cast<PointerType>(Ty);
  if (PtrTy->getElementType()->isAggregateType()) {
    DEBUG(dbgs() << "LAA: Bad stride - Not a pointer to a scalar type"
          << *Ptr << "\n");
    return 0;
  }

  const SCEV *PtrScev = replaceSymbolicStrideSCEV(SE, StridesMap, Ptr);

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);
  if (!AR) {
    DEBUG(dbgs() << "LAA: Bad stride - Not an AddRecExpr pointer "
          << *Ptr << " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  // The accesss function must stride over the innermost loop.
  if (Lp != AR->getLoop()) {
    DEBUG(dbgs() << "LAA: Bad stride - Not striding over innermost loop " <<
          *Ptr << " SCEV: " << *PtrScev << "\n");
  }

  // The address calculation must not wrap. Otherwise, a dependence could be
  // inverted.
  // An inbounds getelementptr that is a AddRec with a unit stride
  // cannot wrap per definition. The unit stride requirement is checked later.
  // An getelementptr without an inbounds attribute and unit stride would have
  // to access the pointer value "0" which is undefined behavior in address
  // space 0, therefore we can also vectorize this case.
  bool IsInBoundsGEP = isInBoundsGep(Ptr);
  bool IsNoWrapAddRec = AR->getNoWrapFlags(SCEV::NoWrapMask);
  bool IsInAddressSpaceZero = PtrTy->getAddressSpace() == 0;
  if (!IsNoWrapAddRec && !IsInBoundsGEP && !IsInAddressSpaceZero) {
    DEBUG(dbgs() << "LAA: Bad stride - Pointer may wrap in the address space "
          << *Ptr << " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  // Check the step is constant.
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C) {
    DEBUG(dbgs() << "LAA: Bad stride - Not a constant strided " << *Ptr <<
          " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  auto &DL = Lp->getHeader()->getModule()->getDataLayout();
  int64_t Size = DL.getTypeAllocSize(PtrTy->getElementType());
  const APInt &APStepVal = C->getValue()->getValue();

  // Huge step value - give up.
  if (APStepVal.getBitWidth() > 64)
    return 0;

  int64_t StepVal = APStepVal.getSExtValue();

  // Strided access.
  int64_t Stride = StepVal / Size;
  int64_t Rem = StepVal % Size;
  if (Rem)
    return 0;

  // If the SCEV could wrap but we have an inbounds gep with a unit stride we
  // know we can't "wrap around the address space". In case of address space
  // zero we know that this won't happen without triggering undefined behavior.
  if (!IsNoWrapAddRec && (IsInBoundsGEP || IsInAddressSpaceZero) &&
      Stride != 1 && Stride != -1)
    return 0;

  return Stride;
}

bool MemoryDepChecker::Dependence::isSafeForVectorization(DepType Type) {
  switch (Type) {
  case NoDep:
  case Forward:
  case BackwardVectorizable:
    return true;

  case Unknown:
  case ForwardButPreventsForwarding:
  case Backward:
  case BackwardVectorizableButPreventsForwarding:
    return false;
  }
}

bool MemoryDepChecker::Dependence::isInterestingDependence(DepType Type) {
  switch (Type) {
  case NoDep:
  case Forward:
    return false;

  case BackwardVectorizable:
  case Unknown:
  case ForwardButPreventsForwarding:
  case Backward:
  case BackwardVectorizableButPreventsForwarding:
    return true;
  }
}

bool MemoryDepChecker::Dependence::isPossiblyBackward() const {
  switch (Type) {
  case NoDep:
  case Forward:
  case ForwardButPreventsForwarding:
    return false;

  case Unknown:
  case BackwardVectorizable:
  case Backward:
  case BackwardVectorizableButPreventsForwarding:
    return true;
  }
}

bool MemoryDepChecker::couldPreventStoreLoadForward(unsigned Distance,
                                                    unsigned TypeByteSize) {
  // If loads occur at a distance that is not a multiple of a feasible vector
  // factor store-load forwarding does not take place.
  // Positive dependences might cause troubles because vectorizing them might
  // prevent store-load forwarding making vectorized code run a lot slower.
  //   a[i] = a[i-3] ^ a[i-8];
  //   The stores to a[i:i+1] don't align with the stores to a[i-3:i-2] and
  //   hence on your typical architecture store-load forwarding does not take
  //   place. Vectorizing in such cases does not make sense.
  // Store-load forwarding distance.
  const unsigned NumCyclesForStoreLoadThroughMemory = 8*TypeByteSize;
  // Maximum vector factor.
  unsigned MaxVFWithoutSLForwardIssues =
    VectorizerParams::MaxVectorWidth * TypeByteSize;
  if(MaxSafeDepDistBytes < MaxVFWithoutSLForwardIssues)
    MaxVFWithoutSLForwardIssues = MaxSafeDepDistBytes;

  for (unsigned vf = 2*TypeByteSize; vf <= MaxVFWithoutSLForwardIssues;
       vf *= 2) {
    if (Distance % vf && Distance / vf < NumCyclesForStoreLoadThroughMemory) {
      MaxVFWithoutSLForwardIssues = (vf >>=1);
      break;
    }
  }

  if (MaxVFWithoutSLForwardIssues< 2*TypeByteSize) {
    DEBUG(dbgs() << "LAA: Distance " << Distance <<
          " that could cause a store-load forwarding conflict\n");
    return true;
  }

  if (MaxVFWithoutSLForwardIssues < MaxSafeDepDistBytes &&
      MaxVFWithoutSLForwardIssues !=
      VectorizerParams::MaxVectorWidth * TypeByteSize)
    MaxSafeDepDistBytes = MaxVFWithoutSLForwardIssues;
  return false;
}

MemoryDepChecker::Dependence::DepType
MemoryDepChecker::isDependent(const MemAccessInfo &A, unsigned AIdx,
                              const MemAccessInfo &B, unsigned BIdx,
                              const ValueToValueMap &Strides) {
  assert (AIdx < BIdx && "Must pass arguments in program order");

  Value *APtr = A.getPointer();
  Value *BPtr = B.getPointer();
  bool AIsWrite = A.getInt();
  bool BIsWrite = B.getInt();

  // Two reads are independent.
  if (!AIsWrite && !BIsWrite)
    return Dependence::NoDep;

  // We cannot check pointers in different address spaces.
  if (APtr->getType()->getPointerAddressSpace() !=
      BPtr->getType()->getPointerAddressSpace())
    return Dependence::Unknown;

  const SCEV *AScev = replaceSymbolicStrideSCEV(SE, Strides, APtr);
  const SCEV *BScev = replaceSymbolicStrideSCEV(SE, Strides, BPtr);

  int StrideAPtr = isStridedPtr(SE, APtr, InnermostLoop, Strides);
  int StrideBPtr = isStridedPtr(SE, BPtr, InnermostLoop, Strides);

  const SCEV *Src = AScev;
  const SCEV *Sink = BScev;

  // If the induction step is negative we have to invert source and sink of the
  // dependence.
  if (StrideAPtr < 0) {
    //Src = BScev;
    //Sink = AScev;
    std::swap(APtr, BPtr);
    std::swap(Src, Sink);
    std::swap(AIsWrite, BIsWrite);
    std::swap(AIdx, BIdx);
    std::swap(StrideAPtr, StrideBPtr);
  }

  const SCEV *Dist = SE->getMinusSCEV(Sink, Src);

  DEBUG(dbgs() << "LAA: Src Scev: " << *Src << "Sink Scev: " << *Sink
        << "(Induction step: " << StrideAPtr <<  ")\n");
  DEBUG(dbgs() << "LAA: Distance for " << *InstMap[AIdx] << " to "
        << *InstMap[BIdx] << ": " << *Dist << "\n");

  // Need consecutive accesses. We don't want to vectorize
  // "A[B[i]] += ..." and similar code or pointer arithmetic that could wrap in
  // the address space.
  if (!StrideAPtr || !StrideBPtr || StrideAPtr != StrideBPtr){
    DEBUG(dbgs() << "Non-consecutive pointer access\n");
    return Dependence::Unknown;
  }

  const SCEVConstant *C = dyn_cast<SCEVConstant>(Dist);
  if (!C) {
    DEBUG(dbgs() << "LAA: Dependence because of non-constant distance\n");
    ShouldRetryWithRuntimeCheck = true;
    return Dependence::Unknown;
  }

  Type *ATy = APtr->getType()->getPointerElementType();
  Type *BTy = BPtr->getType()->getPointerElementType();
  auto &DL = InnermostLoop->getHeader()->getModule()->getDataLayout();
  unsigned TypeByteSize = DL.getTypeAllocSize(ATy);

  // Negative distances are not plausible dependencies.
  const APInt &Val = C->getValue()->getValue();
  if (Val.isNegative()) {
    bool IsTrueDataDependence = (AIsWrite && !BIsWrite);
    if (IsTrueDataDependence &&
        (couldPreventStoreLoadForward(Val.abs().getZExtValue(), TypeByteSize) ||
         ATy != BTy))
      return Dependence::ForwardButPreventsForwarding;

    DEBUG(dbgs() << "LAA: Dependence is negative: NoDep\n");
    return Dependence::Forward;
  }

  // Write to the same location with the same size.
  // Could be improved to assert type sizes are the same (i32 == float, etc).
  if (Val == 0) {
    if (ATy == BTy)
      return Dependence::NoDep;
    DEBUG(dbgs() << "LAA: Zero dependence difference but different types\n");
    return Dependence::Unknown;
  }

  assert(Val.isStrictlyPositive() && "Expect a positive value");

  if (ATy != BTy) {
    DEBUG(dbgs() <<
          "LAA: ReadWrite-Write positive dependency with different types\n");
    return Dependence::Unknown;
  }

  unsigned Distance = (unsigned) Val.getZExtValue();

  // Bail out early if passed-in parameters make vectorization not feasible.
  unsigned ForcedFactor = (VectorizerParams::VectorizationFactor ?
                           VectorizerParams::VectorizationFactor : 1);
  unsigned ForcedUnroll = (VectorizerParams::VectorizationInterleave ?
                           VectorizerParams::VectorizationInterleave : 1);

  // The distance must be bigger than the size needed for a vectorized version
  // of the operation and the size of the vectorized operation must not be
  // bigger than the currrent maximum size.
  if (Distance < 2*TypeByteSize ||
      2*TypeByteSize > MaxSafeDepDistBytes ||
      Distance < TypeByteSize * ForcedUnroll * ForcedFactor) {
    DEBUG(dbgs() << "LAA: Failure because of Positive distance "
        << Val.getSExtValue() << '\n');
    return Dependence::Backward;
  }

  // Positive distance bigger than max vectorization factor.
  MaxSafeDepDistBytes = Distance < MaxSafeDepDistBytes ?
    Distance : MaxSafeDepDistBytes;

  bool IsTrueDataDependence = (!AIsWrite && BIsWrite);
  if (IsTrueDataDependence &&
      couldPreventStoreLoadForward(Distance, TypeByteSize))
    return Dependence::BackwardVectorizableButPreventsForwarding;

  DEBUG(dbgs() << "LAA: Positive distance " << Val.getSExtValue() <<
        " with max VF = " << MaxSafeDepDistBytes / TypeByteSize << '\n');

  return Dependence::BackwardVectorizable;
}

bool MemoryDepChecker::areDepsSafe(DepCandidates &AccessSets,
                                   MemAccessInfoSet &CheckDeps,
                                   const ValueToValueMap &Strides) {

  MaxSafeDepDistBytes = -1U;
  while (!CheckDeps.empty()) {
    MemAccessInfo CurAccess = *CheckDeps.begin();

    // Get the relevant memory access set.
    EquivalenceClasses<MemAccessInfo>::iterator I =
      AccessSets.findValue(AccessSets.getLeaderValue(CurAccess));

    // Check accesses within this set.
    EquivalenceClasses<MemAccessInfo>::member_iterator AI, AE;
    AI = AccessSets.member_begin(I), AE = AccessSets.member_end();

    // Check every access pair.
    while (AI != AE) {
      CheckDeps.erase(*AI);
      EquivalenceClasses<MemAccessInfo>::member_iterator OI = std::next(AI);
      while (OI != AE) {
        // Check every accessing instruction pair in program order.
        for (std::vector<unsigned>::iterator I1 = Accesses[*AI].begin(),
             I1E = Accesses[*AI].end(); I1 != I1E; ++I1)
          for (std::vector<unsigned>::iterator I2 = Accesses[*OI].begin(),
               I2E = Accesses[*OI].end(); I2 != I2E; ++I2) {
            auto A = std::make_pair(&*AI, *I1);
            auto B = std::make_pair(&*OI, *I2);

            assert(*I1 != *I2);
            if (*I1 > *I2)
              std::swap(A, B);

            Dependence::DepType Type =
                isDependent(*A.first, A.second, *B.first, B.second, Strides);
            SafeForVectorization &= Dependence::isSafeForVectorization(Type);

            // Gather dependences unless we accumulated MaxInterestingDependence
            // dependences.  In that case return as soon as we find the first
            // unsafe dependence.  This puts a limit on this quadratic
            // algorithm.
            if (RecordInterestingDependences) {
              if (Dependence::isInterestingDependence(Type))
                InterestingDependences.push_back(
                    Dependence(A.second, B.second, Type));

              if (InterestingDependences.size() >= MaxInterestingDependence) {
                RecordInterestingDependences = false;
                InterestingDependences.clear();
                DEBUG(dbgs() << "Too many dependences, stopped recording\n");
              }
            }
            if (!RecordInterestingDependences && !SafeForVectorization)
              return false;
          }
        ++OI;
      }
      AI++;
    }
  }

  DEBUG(dbgs() << "Total Interesting Dependences: "
               << InterestingDependences.size() << "\n");
  return SafeForVectorization;
}

SmallVector<Instruction *, 4>
MemoryDepChecker::getInstructionsForAccess(Value *Ptr, bool isWrite) const {
  MemAccessInfo Access(Ptr, isWrite);
  auto &IndexVector = Accesses.find(Access)->second;

  SmallVector<Instruction *, 4> Insts;
  std::transform(IndexVector.begin(), IndexVector.end(),
                 std::back_inserter(Insts),
                 [&](unsigned Idx) { return this->InstMap[Idx]; });
  return Insts;
}

const char *MemoryDepChecker::Dependence::DepName[] = {
    "NoDep", "Unknown", "Forward", "ForwardButPreventsForwarding", "Backward",
    "BackwardVectorizable", "BackwardVectorizableButPreventsForwarding"};

void MemoryDepChecker::Dependence::print(
    raw_ostream &OS, unsigned Depth,
    const SmallVectorImpl<Instruction *> &Instrs) const {
  OS.indent(Depth) << DepName[Type] << ":\n";
  OS.indent(Depth + 2) << *Instrs[Source] << " -> \n";
  OS.indent(Depth + 2) << *Instrs[Destination] << "\n";
}

bool LoopAccessInfo::canAnalyzeLoop() {
    // We can only analyze innermost loops.
  if (!TheLoop->empty()) {
    emitAnalysis(LoopAccessReport() << "loop is not the innermost loop");
    return false;
  }

  // We must have a single backedge.
  if (TheLoop->getNumBackEdges() != 1) {
    emitAnalysis(
        LoopAccessReport() <<
        "loop control flow is not understood by analyzer");
    return false;
  }

  // We must have a single exiting block.
  if (!TheLoop->getExitingBlock()) {
    emitAnalysis(
        LoopAccessReport() <<
        "loop control flow is not understood by analyzer");
    return false;
  }

  // We only handle bottom-tested loops, i.e. loop in which the condition is
  // checked at the end of each iteration. With that we can assume that all
  // instructions in the loop are executed the same number of times.
  if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch()) {
    emitAnalysis(
        LoopAccessReport() <<
        "loop control flow is not understood by analyzer");
    return false;
  }

  // We need to have a loop header.
  DEBUG(dbgs() << "LAA: Found a loop: " <<
        TheLoop->getHeader()->getName() << '\n');

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = SE->getBackedgeTakenCount(TheLoop);
  if (ExitCount == SE->getCouldNotCompute()) {
    emitAnalysis(LoopAccessReport() <<
                 "could not determine number of loop iterations");
    DEBUG(dbgs() << "LAA: SCEV could not compute the loop exit count.\n");
    return false;
  }

  return true;
}

void LoopAccessInfo::analyzeLoop(const ValueToValueMap &Strides) {

  typedef SmallVector<Value*, 16> ValueVector;
  typedef SmallPtrSet<Value*, 16> ValueSet;

  // Holds the Load and Store *instructions*.
  ValueVector Loads;
  ValueVector Stores;

  // Holds all the different accesses in the loop.
  unsigned NumReads = 0;
  unsigned NumReadWrites = 0;

  PtrRtCheck.Pointers.clear();
  PtrRtCheck.Need = false;

  const bool IsAnnotatedParallel = TheLoop->isAnnotatedParallel();

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {

    // Scan the BB and collect legal loads and stores.
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {

      // If this is a load, save it. If this instruction can read from memory
      // but is not a load, then we quit. Notice that we don't handle function
      // calls that read or write.
      if (it->mayReadFromMemory()) {
        // Many math library functions read the rounding mode. We will only
        // vectorize a loop if it contains known function calls that don't set
        // the flag. Therefore, it is safe to ignore this read from memory.
        CallInst *Call = dyn_cast<CallInst>(it);
        if (Call && getIntrinsicIDForCall(Call, TLI))
          continue;

        LoadInst *Ld = dyn_cast<LoadInst>(it);
        if (!Ld || (!Ld->isSimple() && !IsAnnotatedParallel)) {
          emitAnalysis(LoopAccessReport(Ld)
                       << "read with atomic ordering or volatile read");
          DEBUG(dbgs() << "LAA: Found a non-simple load.\n");
          CanVecMem = false;
          return;
        }
        NumLoads++;
        Loads.push_back(Ld);
        DepChecker.addAccess(Ld);
        continue;
      }

      // Save 'store' instructions. Abort if other instructions write to memory.
      if (it->mayWriteToMemory()) {
        StoreInst *St = dyn_cast<StoreInst>(it);
        if (!St) {
          emitAnalysis(LoopAccessReport(it) <<
                       "instruction cannot be vectorized");
          CanVecMem = false;
          return;
        }
        if (!St->isSimple() && !IsAnnotatedParallel) {
          emitAnalysis(LoopAccessReport(St)
                       << "write with atomic ordering or volatile write");
          DEBUG(dbgs() << "LAA: Found a non-simple store.\n");
          CanVecMem = false;
          return;
        }
        NumStores++;
        Stores.push_back(St);
        DepChecker.addAccess(St);
      }
    } // Next instr.
  } // Next block.

  // Now we have two lists that hold the loads and the stores.
  // Next, we find the pointers that they use.

  // Check if we see any stores. If there are no stores, then we don't
  // care if the pointers are *restrict*.
  if (!Stores.size()) {
    DEBUG(dbgs() << "LAA: Found a read-only loop!\n");
    CanVecMem = true;
    return;
  }

  MemoryDepChecker::DepCandidates DependentAccesses;
  AccessAnalysis Accesses(TheLoop->getHeader()->getModule()->getDataLayout(),
                          AA, DependentAccesses);

  // Holds the analyzed pointers. We don't want to call GetUnderlyingObjects
  // multiple times on the same object. If the ptr is accessed twice, once
  // for read and once for write, it will only appear once (on the write
  // list). This is okay, since we are going to check for conflicts between
  // writes and between reads and writes, but not between reads and reads.
  ValueSet Seen;

  ValueVector::iterator I, IE;
  for (I = Stores.begin(), IE = Stores.end(); I != IE; ++I) {
    StoreInst *ST = cast<StoreInst>(*I);
    Value* Ptr = ST->getPointerOperand();

    if (isUniform(Ptr)) {
      emitAnalysis(
          LoopAccessReport(ST)
          << "write to a loop invariant address could not be vectorized");
      DEBUG(dbgs() << "LAA: We don't allow storing to uniform addresses\n");
      CanVecMem = false;
      return;
    }

    // If we did *not* see this pointer before, insert it to  the read-write
    // list. At this phase it is only a 'write' list.
    if (Seen.insert(Ptr).second) {
      ++NumReadWrites;

      AliasAnalysis::Location Loc = AA->getLocation(ST);
      // The TBAA metadata could have a control dependency on the predication
      // condition, so we cannot rely on it when determining whether or not we
      // need runtime pointer checks.
      if (blockNeedsPredication(ST->getParent(), TheLoop, DT))
        Loc.AATags.TBAA = nullptr;

      Accesses.addStore(Loc);
    }
  }

  if (IsAnnotatedParallel) {
    DEBUG(dbgs()
          << "LAA: A loop annotated parallel, ignore memory dependency "
          << "checks.\n");
    CanVecMem = true;
    return;
  }

  for (I = Loads.begin(), IE = Loads.end(); I != IE; ++I) {
    LoadInst *LD = cast<LoadInst>(*I);
    Value* Ptr = LD->getPointerOperand();
    // If we did *not* see this pointer before, insert it to the
    // read list. If we *did* see it before, then it is already in
    // the read-write list. This allows us to vectorize expressions
    // such as A[i] += x;  Because the address of A[i] is a read-write
    // pointer. This only works if the index of A[i] is consecutive.
    // If the address of i is unknown (for example A[B[i]]) then we may
    // read a few words, modify, and write a few words, and some of the
    // words may be written to the same address.
    bool IsReadOnlyPtr = false;
    if (Seen.insert(Ptr).second || !isStridedPtr(SE, Ptr, TheLoop, Strides)) {
      ++NumReads;
      IsReadOnlyPtr = true;
    }

    AliasAnalysis::Location Loc = AA->getLocation(LD);
    // The TBAA metadata could have a control dependency on the predication
    // condition, so we cannot rely on it when determining whether or not we
    // need runtime pointer checks.
    if (blockNeedsPredication(LD->getParent(), TheLoop, DT))
      Loc.AATags.TBAA = nullptr;

    Accesses.addLoad(Loc, IsReadOnlyPtr);
  }

  // If we write (or read-write) to a single destination and there are no
  // other reads in this loop then is it safe to vectorize.
  if (NumReadWrites == 1 && NumReads == 0) {
    DEBUG(dbgs() << "LAA: Found a write-only loop!\n");
    CanVecMem = true;
    return;
  }

  // Build dependence sets and check whether we need a runtime pointer bounds
  // check.
  Accesses.buildDependenceSets();
  bool NeedRTCheck = Accesses.isRTCheckNeeded();

  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  bool CanDoRT = false;
  if (NeedRTCheck)
    CanDoRT = Accesses.canCheckPtrAtRT(PtrRtCheck, NumComparisons, SE, TheLoop,
                                       Strides);

  DEBUG(dbgs() << "LAA: We need to do " << NumComparisons <<
        " pointer comparisons.\n");

  // If we only have one set of dependences to check pointers among we don't
  // need a runtime check.
  if (NumComparisons == 0 && NeedRTCheck)
    NeedRTCheck = false;

  // Check that we did not find an unsizeable pointer.
  if (CanDoRT)
    DEBUG(dbgs() << "LAA: We can perform a memory runtime check if needed.\n");
  else if (NeedRTCheck) {
    emitAnalysis(LoopAccessReport() << "cannot identify array bounds");
    DEBUG(dbgs() << "LAA: We can't vectorize because we can't find " <<
          "the array bounds.\n");
    PtrRtCheck.reset();
    CanVecMem = false;
    return;
  }

  PtrRtCheck.Need = NeedRTCheck;

  CanVecMem = true;
  if (Accesses.isDependencyCheckNeeded()) {
    DEBUG(dbgs() << "LAA: Checking memory dependencies\n");
    CanVecMem = DepChecker.areDepsSafe(
        DependentAccesses, Accesses.getDependenciesToCheck(), Strides);
    MaxSafeDepDistBytes = DepChecker.getMaxSafeDepDistBytes();

    if (!CanVecMem && DepChecker.shouldRetryWithRuntimeCheck()) {
      DEBUG(dbgs() << "LAA: Retrying with memory checks\n");
      NeedRTCheck = true;

      // Clear the dependency checks. We assume they are not needed.
      Accesses.resetDepChecks();

      PtrRtCheck.reset();
      PtrRtCheck.Need = true;

      CanDoRT = Accesses.canCheckPtrAtRT(PtrRtCheck, NumComparisons, SE,
                                         TheLoop, Strides, true);
      // Check that we didn't find an unsizeable pointer.
      if (!CanDoRT && NumComparisons > 0) {
        emitAnalysis(LoopAccessReport()
                     << "cannot check memory dependencies at runtime");
        DEBUG(dbgs() << "LAA: Can't vectorize with memory checks\n");
        PtrRtCheck.reset();
        CanVecMem = false;
        return;
      }

      CanVecMem = true;
    }
  }

  if (!CanVecMem)
    emitAnalysis(LoopAccessReport() <<
                 "unsafe dependent memory operations in loop");

  DEBUG(dbgs() << "LAA: We" << (NeedRTCheck ? "" : " don't") <<
        " need a runtime memory check.\n");
}

bool LoopAccessInfo::blockNeedsPredication(BasicBlock *BB, Loop *TheLoop,
                                           DominatorTree *DT)  {
  assert(TheLoop->contains(BB) && "Unknown block used");

  // Blocks that do not dominate the latch need predication.
  BasicBlock* Latch = TheLoop->getLoopLatch();
  return !DT->dominates(BB, Latch);
}

void LoopAccessInfo::emitAnalysis(LoopAccessReport &Message) {
  assert(!Report && "Multiple reports generated");
  Report = Message;
}

bool LoopAccessInfo::isUniform(Value *V) const {
  return (SE->isLoopInvariant(SE->getSCEV(V), TheLoop));
}

// FIXME: this function is currently a duplicate of the one in
// LoopVectorize.cpp.
static Instruction *getFirstInst(Instruction *FirstInst, Value *V,
                                 Instruction *Loc) {
  if (FirstInst)
    return FirstInst;
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() == Loc->getParent() ? I : nullptr;
  return nullptr;
}

std::pair<Instruction *, Instruction *> LoopAccessInfo::addRuntimeCheck(
    Instruction *Loc, const SmallVectorImpl<int> *PtrPartition) const {
  Instruction *tnullptr = nullptr;
  if (!PtrRtCheck.Need)
    return std::pair<Instruction *, Instruction *>(tnullptr, tnullptr);

  unsigned NumPointers = PtrRtCheck.Pointers.size();
  SmallVector<TrackingVH<Value> , 2> Starts;
  SmallVector<TrackingVH<Value> , 2> Ends;

  LLVMContext &Ctx = Loc->getContext();
  SCEVExpander Exp(*SE, DL, "induction");
  Instruction *FirstInst = nullptr;

  for (unsigned i = 0; i < NumPointers; ++i) {
    Value *Ptr = PtrRtCheck.Pointers[i];
    const SCEV *Sc = SE->getSCEV(Ptr);

    if (SE->isLoopInvariant(Sc, TheLoop)) {
      DEBUG(dbgs() << "LAA: Adding RT check for a loop invariant ptr:" <<
            *Ptr <<"\n");
      Starts.push_back(Ptr);
      Ends.push_back(Ptr);
    } else {
      DEBUG(dbgs() << "LAA: Adding RT check for range:" << *Ptr << '\n');
      unsigned AS = Ptr->getType()->getPointerAddressSpace();

      // Use this type for pointer arithmetic.
      Type *PtrArithTy = Type::getInt8PtrTy(Ctx, AS);

      Value *Start = Exp.expandCodeFor(PtrRtCheck.Starts[i], PtrArithTy, Loc);
      Value *End = Exp.expandCodeFor(PtrRtCheck.Ends[i], PtrArithTy, Loc);
      Starts.push_back(Start);
      Ends.push_back(End);
    }
  }

  IRBuilder<> ChkBuilder(Loc);
  // Our instructions might fold to a constant.
  Value *MemoryRuntimeCheck = nullptr;
  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i+1; j < NumPointers; ++j) {
      if (!PtrRtCheck.needsChecking(i, j, PtrPartition))
        continue;

      unsigned AS0 = Starts[i]->getType()->getPointerAddressSpace();
      unsigned AS1 = Starts[j]->getType()->getPointerAddressSpace();

      assert((AS0 == Ends[j]->getType()->getPointerAddressSpace()) &&
             (AS1 == Ends[i]->getType()->getPointerAddressSpace()) &&
             "Trying to bounds check pointers with different address spaces");

      Type *PtrArithTy0 = Type::getInt8PtrTy(Ctx, AS0);
      Type *PtrArithTy1 = Type::getInt8PtrTy(Ctx, AS1);

      Value *Start0 = ChkBuilder.CreateBitCast(Starts[i], PtrArithTy0, "bc");
      Value *Start1 = ChkBuilder.CreateBitCast(Starts[j], PtrArithTy1, "bc");
      Value *End0 =   ChkBuilder.CreateBitCast(Ends[i],   PtrArithTy1, "bc");
      Value *End1 =   ChkBuilder.CreateBitCast(Ends[j],   PtrArithTy0, "bc");

      Value *Cmp0 = ChkBuilder.CreateICmpULE(Start0, End1, "bound0");
      FirstInst = getFirstInst(FirstInst, Cmp0, Loc);
      Value *Cmp1 = ChkBuilder.CreateICmpULE(Start1, End0, "bound1");
      FirstInst = getFirstInst(FirstInst, Cmp1, Loc);
      Value *IsConflict = ChkBuilder.CreateAnd(Cmp0, Cmp1, "found.conflict");
      FirstInst = getFirstInst(FirstInst, IsConflict, Loc);
      if (MemoryRuntimeCheck) {
        IsConflict = ChkBuilder.CreateOr(MemoryRuntimeCheck, IsConflict,
                                         "conflict.rdx");
        FirstInst = getFirstInst(FirstInst, IsConflict, Loc);
      }
      MemoryRuntimeCheck = IsConflict;
    }
  }

  // We have to do this trickery because the IRBuilder might fold the check to a
  // constant expression in which case there is no Instruction anchored in a
  // the block.
  Instruction *Check = BinaryOperator::CreateAnd(MemoryRuntimeCheck,
                                                 ConstantInt::getTrue(Ctx));
  ChkBuilder.Insert(Check, "memcheck.conflict");
  FirstInst = getFirstInst(FirstInst, Check, Loc);
  return std::make_pair(FirstInst, Check);
}

LoopAccessInfo::LoopAccessInfo(Loop *L, ScalarEvolution *SE,
                               const DataLayout &DL,
                               const TargetLibraryInfo *TLI, AliasAnalysis *AA,
                               DominatorTree *DT,
                               const ValueToValueMap &Strides)
    : DepChecker(SE, L), NumComparisons(0), TheLoop(L), SE(SE), DL(DL),
      TLI(TLI), AA(AA), DT(DT), NumLoads(0), NumStores(0),
      MaxSafeDepDistBytes(-1U), CanVecMem(false) {
  if (canAnalyzeLoop())
    analyzeLoop(Strides);
}

void LoopAccessInfo::print(raw_ostream &OS, unsigned Depth) const {
  if (CanVecMem) {
    if (PtrRtCheck.empty())
      OS.indent(Depth) << "Memory dependences are safe\n";
    else
      OS.indent(Depth) << "Memory dependences are safe with run-time checks\n";
  }

  if (Report)
    OS.indent(Depth) << "Report: " << Report->str() << "\n";

  if (auto *InterestingDependences = DepChecker.getInterestingDependences()) {
    OS.indent(Depth) << "Interesting Dependences:\n";
    for (auto &Dep : *InterestingDependences) {
      Dep.print(OS, Depth + 2, DepChecker.getMemoryInstructions());
      OS << "\n";
    }
  } else
    OS.indent(Depth) << "Too many interesting dependences, not recorded\n";

  // List the pair of accesses need run-time checks to prove independence.
  PtrRtCheck.print(OS, Depth);
  OS << "\n";
}

const LoopAccessInfo &
LoopAccessAnalysis::getInfo(Loop *L, const ValueToValueMap &Strides) {
  auto &LAI = LoopAccessInfoMap[L];

#ifndef NDEBUG
  assert((!LAI || LAI->NumSymbolicStrides == Strides.size()) &&
         "Symbolic strides changed for loop");
#endif

  if (!LAI) {
    const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
    LAI = llvm::make_unique<LoopAccessInfo>(L, SE, DL, TLI, AA, DT, Strides);
#ifndef NDEBUG
    LAI->NumSymbolicStrides = Strides.size();
#endif
  }
  return *LAI.get();
}

void LoopAccessAnalysis::print(raw_ostream &OS, const Module *M) const {
  LoopAccessAnalysis &LAA = *const_cast<LoopAccessAnalysis *>(this);

  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ValueToValueMap NoSymbolicStrides;

  for (Loop *TopLevelLoop : *LI)
    for (Loop *L : depth_first(TopLevelLoop)) {
      OS.indent(2) << L->getHeader()->getName() << ":\n";
      auto &LAI = LAA.getInfo(L, NoSymbolicStrides);
      LAI.print(OS, 4);
    }
}

bool LoopAccessAnalysis::runOnFunction(Function &F) {
  SE = &getAnalysis<ScalarEvolution>();
  auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  TLI = TLIP ? &TLIP->getTLI() : nullptr;
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  return false;
}

void LoopAccessAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.setPreservesAll();
}

char LoopAccessAnalysis::ID = 0;
static const char laa_name[] = "Loop Access Analysis";
#define LAA_NAME "loop-accesses"

INITIALIZE_PASS_BEGIN(LoopAccessAnalysis, LAA_NAME, laa_name, false, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopAccessAnalysis, LAA_NAME, laa_name, false, true)

namespace llvm {
  Pass *createLAAPass() {
    return new LoopAccessAnalysis();
  }
}
