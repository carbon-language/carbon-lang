//===- BBVectorize.cpp - A Basic-Block Vectorizer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a basic-block vectorization pass. The algorithm was
// inspired by that used by the Vienna MAP Vectorizor by Franchetti and Kral,
// et al. It works by looking for chains of pairable operations and then
// pairing them.
//
//===----------------------------------------------------------------------===//

#define BBV_NAME "bb-vectorize"
#define DEBUG_TYPE BBV_NAME
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Vectorize.h"
#include <algorithm>
#include <map>
using namespace llvm;

static cl::opt<unsigned>
ReqChainDepth("bb-vectorize-req-chain-depth", cl::init(6), cl::Hidden,
  cl::desc("The required chain depth for vectorization"));

static cl::opt<unsigned>
SearchLimit("bb-vectorize-search-limit", cl::init(400), cl::Hidden,
  cl::desc("The maximum search distance for instruction pairs"));

static cl::opt<bool>
SplatBreaksChain("bb-vectorize-splat-breaks-chain", cl::init(false), cl::Hidden,
  cl::desc("Replicating one element to a pair breaks the chain"));

static cl::opt<unsigned>
VectorBits("bb-vectorize-vector-bits", cl::init(128), cl::Hidden,
  cl::desc("The size of the native vector registers"));

static cl::opt<unsigned>
MaxIter("bb-vectorize-max-iter", cl::init(0), cl::Hidden,
  cl::desc("The maximum number of pairing iterations"));

static cl::opt<unsigned>
MaxInsts("bb-vectorize-max-instr-per-group", cl::init(500), cl::Hidden,
  cl::desc("The maximum number of pairable instructions per group"));

static cl::opt<unsigned>
MaxCandPairsForCycleCheck("bb-vectorize-max-cycle-check-pairs", cl::init(200),
  cl::Hidden, cl::desc("The maximum number of candidate pairs with which to use"
                       " a full cycle check"));

static cl::opt<bool>
NoInts("bb-vectorize-no-ints", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize integer values"));

static cl::opt<bool>
NoFloats("bb-vectorize-no-floats", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point values"));

static cl::opt<bool>
NoPointers("bb-vectorize-no-pointers", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize pointer values"));

static cl::opt<bool>
NoCasts("bb-vectorize-no-casts", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize casting (conversion) operations"));

static cl::opt<bool>
NoMath("bb-vectorize-no-math", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point math intrinsics"));

static cl::opt<bool>
NoFMA("bb-vectorize-no-fma", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize the fused-multiply-add intrinsic"));

static cl::opt<bool>
NoSelect("bb-vectorize-no-select", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize select instructions"));

static cl::opt<bool>
NoGEP("bb-vectorize-no-gep", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize getelementptr instructions"));

static cl::opt<bool>
NoMemOps("bb-vectorize-no-mem-ops", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize loads and stores"));

static cl::opt<bool>
AlignedOnly("bb-vectorize-aligned-only", cl::init(false), cl::Hidden,
  cl::desc("Only generate aligned loads and stores"));

static cl::opt<bool>
NoMemOpBoost("bb-vectorize-no-mem-op-boost",
  cl::init(false), cl::Hidden,
  cl::desc("Don't boost the chain-depth contribution of loads and stores"));

static cl::opt<bool>
FastDep("bb-vectorize-fast-dep", cl::init(false), cl::Hidden,
  cl::desc("Use a fast instruction dependency analysis"));

#ifndef NDEBUG
static cl::opt<bool>
DebugInstructionExamination("bb-vectorize-debug-instruction-examination",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " instruction-examination process"));
static cl::opt<bool>
DebugCandidateSelection("bb-vectorize-debug-candidate-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " candidate-selection process"));
static cl::opt<bool>
DebugPairSelection("bb-vectorize-debug-pair-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " pair-selection process"));
static cl::opt<bool>
DebugCycleCheck("bb-vectorize-debug-cycle-check",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " cycle-checking process"));
#endif

STATISTIC(NumFusedOps, "Number of operations fused by bb-vectorize");

namespace {
  struct BBVectorize : public BasicBlockPass {
    static char ID; // Pass identification, replacement for typeid

    const VectorizeConfig Config;

    BBVectorize(const VectorizeConfig &C = VectorizeConfig())
      : BasicBlockPass(ID), Config(C) {
      initializeBBVectorizePass(*PassRegistry::getPassRegistry());
    }

    BBVectorize(Pass *P, const VectorizeConfig &C)
      : BasicBlockPass(ID), Config(C) {
      AA = &P->getAnalysis<AliasAnalysis>();
      SE = &P->getAnalysis<ScalarEvolution>();
      TD = P->getAnalysisIfAvailable<TargetData>();
    }

    typedef std::pair<Value *, Value *> ValuePair;
    typedef std::pair<ValuePair, size_t> ValuePairWithDepth;
    typedef std::pair<ValuePair, ValuePair> VPPair; // A ValuePair pair
    typedef std::pair<std::multimap<Value *, Value *>::iterator,
              std::multimap<Value *, Value *>::iterator> VPIteratorPair;
    typedef std::pair<std::multimap<ValuePair, ValuePair>::iterator,
              std::multimap<ValuePair, ValuePair>::iterator>
                VPPIteratorPair;

    AliasAnalysis *AA;
    ScalarEvolution *SE;
    TargetData *TD;

    // FIXME: const correct?

    bool vectorizePairs(BasicBlock &BB);

    bool getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts);

    void computeConnectedPairs(std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts,
                       std::multimap<ValuePair, ValuePair> &ConnectedPairs);

    void buildDepMap(BasicBlock &BB,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts,
                       DenseSet<ValuePair> &PairableInstUsers);

    void choosePairs(std::multimap<Value *, Value *> &CandidatePairs,
                        std::vector<Value *> &PairableInsts,
                        std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                        DenseSet<ValuePair> &PairableInstUsers,
                        DenseMap<Value *, Value *>& ChosenPairs);

    void fuseChosenPairs(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *>& ChosenPairs);

    bool isInstVectorizable(Instruction *I, bool &IsSimpleLoadStore);

    bool areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore);

    bool trackUsesOfI(DenseSet<Value *> &Users,
                      AliasSetTracker &WriteSet, Instruction *I,
                      Instruction *J, bool UpdateUsers = true,
                      std::multimap<Value *, Value *> *LoadMoveSet = 0);

    void computePairsConnectedTo(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      ValuePair P);

    bool pairsConflict(ValuePair P, ValuePair Q,
                 DenseSet<ValuePair> &PairableInstUsers,
                 std::multimap<ValuePair, ValuePair> *PairableInstUserMap = 0);

    bool pairWillFormCycle(ValuePair P,
                       std::multimap<ValuePair, ValuePair> &PairableInstUsers,
                       DenseSet<ValuePair> &CurrentPairs);

    void pruneTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree,
                      DenseSet<ValuePair> &PrunedTree, ValuePair J,
                      bool UseCycleCheck);

    void buildInitialTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree, ValuePair J);

    void findBestTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseSet<ValuePair> &BestTree, size_t &BestMaxDepth,
                      size_t &BestEffSize, VPIteratorPair ChoiceRange,
                      bool UseCycleCheck);

    Value *getReplacementPointerInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool &FlipMemInputs);

    void fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned NumElem, unsigned MaskOffset, unsigned NumInElem,
                     unsigned IdxOffset, std::vector<Constant*> &Mask);

    Value *getReplacementShuffleMask(LLVMContext& Context, Instruction *I,
                     Instruction *J);

    Value *getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool FlipMemInputs);

    void getReplacementInputsForPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, SmallVector<Value *, 3> &ReplacedOperands,
                     bool &FlipMemInputs);

    void replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt, Instruction *&K1,
                     Instruction *&K2, bool &FlipMemInputs);

    void collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I);

    void collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet);

    bool canMoveUsesOfIAfterJ(BasicBlock &BB,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I, Instruction *J);

    void moveUsesOfIAfterJ(BasicBlock &BB,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J);

    bool vectorizeBB(BasicBlock &BB) {
      bool changed = false;
      // Iterate a sufficient number of times to merge types of size 1 bit,
      // then 2 bits, then 4, etc. up to half of the target vector width of the
      // target vector register.
      for (unsigned v = 2, n = 1;
           v <= Config.VectorBits && (!Config.MaxIter || n <= Config.MaxIter);
           v *= 2, ++n) {
        DEBUG(dbgs() << "BBV: fusing loop #" << n <<
              " for " << BB.getName() << " in " <<
              BB.getParent()->getName() << "...\n");
        if (vectorizePairs(BB))
          changed = true;
        else
          break;
      }

      DEBUG(dbgs() << "BBV: done!\n");
      return changed;
    }

    virtual bool runOnBasicBlock(BasicBlock &BB) {
      AA = &getAnalysis<AliasAnalysis>();
      SE = &getAnalysis<ScalarEvolution>();
      TD = getAnalysisIfAvailable<TargetData>();

      return vectorizeBB(BB);
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      BasicBlockPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.setPreservesCFG();
    }

    // This returns the vector type that holds a pair of the provided type.
    // If the provided type is already a vector, then its length is doubled.
    static inline VectorType *getVecTypeForPair(Type *ElemTy) {
      if (VectorType *VTy = dyn_cast<VectorType>(ElemTy)) {
        unsigned numElem = VTy->getNumElements();
        return VectorType::get(ElemTy->getScalarType(), numElem*2);
      }

      return VectorType::get(ElemTy, 2);
    }

    // Returns the weight associated with the provided value. A chain of
    // candidate pairs has a length given by the sum of the weights of its
    // members (one weight per pair; the weight of each member of the pair
    // is assumed to be the same). This length is then compared to the
    // chain-length threshold to determine if a given chain is significant
    // enough to be vectorized. The length is also used in comparing
    // candidate chains where longer chains are considered to be better.
    // Note: when this function returns 0, the resulting instructions are
    // not actually fused.
    inline size_t getDepthFactor(Value *V) {
      // InsertElement and ExtractElement have a depth factor of zero. This is
      // for two reasons: First, they cannot be usefully fused. Second, because
      // the pass generates a lot of these, they can confuse the simple metric
      // used to compare the trees in the next iteration. Thus, giving them a
      // weight of zero allows the pass to essentially ignore them in
      // subsequent iterations when looking for vectorization opportunities
      // while still tracking dependency chains that flow through those
      // instructions.
      if (isa<InsertElementInst>(V) || isa<ExtractElementInst>(V))
        return 0;

      // Give a load or store half of the required depth so that load/store
      // pairs will vectorize.
      if (!Config.NoMemOpBoost && (isa<LoadInst>(V) || isa<StoreInst>(V)))
        return Config.ReqChainDepth/2;

      return 1;
    }

    // This determines the relative offset of two loads or stores, returning
    // true if the offset could be determined to be some constant value.
    // For example, if OffsetInElmts == 1, then J accesses the memory directly
    // after I; if OffsetInElmts == -1 then I accesses the memory
    // directly after J. This function assumes that both instructions
    // have the same type.
    bool getPairPtrInfo(Instruction *I, Instruction *J,
        Value *&IPtr, Value *&JPtr, unsigned &IAlignment, unsigned &JAlignment,
        int64_t &OffsetInElmts) {
      OffsetInElmts = 0;
      if (isa<LoadInst>(I)) {
        IPtr = cast<LoadInst>(I)->getPointerOperand();
        JPtr = cast<LoadInst>(J)->getPointerOperand();
        IAlignment = cast<LoadInst>(I)->getAlignment();
        JAlignment = cast<LoadInst>(J)->getAlignment();
      } else {
        IPtr = cast<StoreInst>(I)->getPointerOperand();
        JPtr = cast<StoreInst>(J)->getPointerOperand();
        IAlignment = cast<StoreInst>(I)->getAlignment();
        JAlignment = cast<StoreInst>(J)->getAlignment();
      }

      const SCEV *IPtrSCEV = SE->getSCEV(IPtr);
      const SCEV *JPtrSCEV = SE->getSCEV(JPtr);

      // If this is a trivial offset, then we'll get something like
      // 1*sizeof(type). With target data, which we need anyway, this will get
      // constant folded into a number.
      const SCEV *OffsetSCEV = SE->getMinusSCEV(JPtrSCEV, IPtrSCEV);
      if (const SCEVConstant *ConstOffSCEV =
            dyn_cast<SCEVConstant>(OffsetSCEV)) {
        ConstantInt *IntOff = ConstOffSCEV->getValue();
        int64_t Offset = IntOff->getSExtValue();

        Type *VTy = cast<PointerType>(IPtr->getType())->getElementType();
        int64_t VTyTSS = (int64_t) TD->getTypeStoreSize(VTy);

        assert(VTy == cast<PointerType>(JPtr->getType())->getElementType());

        OffsetInElmts = Offset/VTyTSS;
        return (abs64(Offset) % VTyTSS) == 0;
      }

      return false;
    }

    // Returns true if the provided CallInst represents an intrinsic that can
    // be vectorized.
    bool isVectorizableIntrinsic(CallInst* I) {
      Function *F = I->getCalledFunction();
      if (!F) return false;

      unsigned IID = F->getIntrinsicID();
      if (!IID) return false;

      switch(IID) {
      default:
        return false;
      case Intrinsic::sqrt:
      case Intrinsic::powi:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::pow:
        return Config.VectorizeMath;
      case Intrinsic::fma:
        return Config.VectorizeFMA;
      }
    }

    // Returns true if J is the second element in some pair referenced by
    // some multimap pair iterator pair.
    template <typename V>
    bool isSecondInIteratorPair(V J, std::pair<
           typename std::multimap<V, V>::iterator,
           typename std::multimap<V, V>::iterator> PairRange) {
      for (typename std::multimap<V, V>::iterator K = PairRange.first;
           K != PairRange.second; ++K)
        if (K->second == J) return true;

      return false;
    }
  };

  // This function implements one vectorization iteration on the provided
  // basic block. It returns true if the block is changed.
  bool BBVectorize::vectorizePairs(BasicBlock &BB) {
    bool ShouldContinue;
    BasicBlock::iterator Start = BB.getFirstInsertionPt();

    std::vector<Value *> AllPairableInsts;
    DenseMap<Value *, Value *> AllChosenPairs;

    do {
      std::vector<Value *> PairableInsts;
      std::multimap<Value *, Value *> CandidatePairs;
      ShouldContinue = getCandidatePairs(BB, Start, CandidatePairs,
                                         PairableInsts);
      if (PairableInsts.empty()) continue;

      // Now we have a map of all of the pairable instructions and we need to
      // select the best possible pairing. A good pairing is one such that the
      // users of the pair are also paired. This defines a (directed) forest
      // over the pairs such that two pairs are connected iff the second pair
      // uses the first.

      // Note that it only matters that both members of the second pair use some
      // element of the first pair (to allow for splatting).

      std::multimap<ValuePair, ValuePair> ConnectedPairs;
      computeConnectedPairs(CandidatePairs, PairableInsts, ConnectedPairs);
      if (ConnectedPairs.empty()) continue;

      // Build the pairable-instruction dependency map
      DenseSet<ValuePair> PairableInstUsers;
      buildDepMap(BB, CandidatePairs, PairableInsts, PairableInstUsers);

      // There is now a graph of the connected pairs. For each variable, pick
      // the pairing with the largest tree meeting the depth requirement on at
      // least one branch. Then select all pairings that are part of that tree
      // and remove them from the list of available pairings and pairable
      // variables.

      DenseMap<Value *, Value *> ChosenPairs;
      choosePairs(CandidatePairs, PairableInsts, ConnectedPairs,
        PairableInstUsers, ChosenPairs);

      if (ChosenPairs.empty()) continue;
      AllPairableInsts.insert(AllPairableInsts.end(), PairableInsts.begin(),
                              PairableInsts.end());
      AllChosenPairs.insert(ChosenPairs.begin(), ChosenPairs.end());
    } while (ShouldContinue);

    if (AllChosenPairs.empty()) return false;
    NumFusedOps += AllChosenPairs.size();

    // A set of pairs has now been selected. It is now necessary to replace the
    // paired instructions with vector instructions. For this procedure each
    // operand must be replaced with a vector operand. This vector is formed
    // by using build_vector on the old operands. The replaced values are then
    // replaced with a vector_extract on the result.  Subsequent optimization
    // passes should coalesce the build/extract combinations.

    fuseChosenPairs(BB, AllPairableInsts, AllChosenPairs);
    return true;
  }

  // This function returns true if the provided instruction is capable of being
  // fused into a vector instruction. This determination is based only on the
  // type and other attributes of the instruction.
  bool BBVectorize::isInstVectorizable(Instruction *I,
                                         bool &IsSimpleLoadStore) {
    IsSimpleLoadStore = false;

    if (CallInst *C = dyn_cast<CallInst>(I)) {
      if (!isVectorizableIntrinsic(C))
        return false;
    } else if (LoadInst *L = dyn_cast<LoadInst>(I)) {
      // Vectorize simple loads if possbile:
      IsSimpleLoadStore = L->isSimple();
      if (!IsSimpleLoadStore || !Config.VectorizeMemOps)
        return false;
    } else if (StoreInst *S = dyn_cast<StoreInst>(I)) {
      // Vectorize simple stores if possbile:
      IsSimpleLoadStore = S->isSimple();
      if (!IsSimpleLoadStore || !Config.VectorizeMemOps)
        return false;
    } else if (CastInst *C = dyn_cast<CastInst>(I)) {
      // We can vectorize casts, but not casts of pointer types, etc.
      if (!Config.VectorizeCasts)
        return false;

      Type *SrcTy = C->getSrcTy();
      if (!SrcTy->isSingleValueType())
        return false;

      Type *DestTy = C->getDestTy();
      if (!DestTy->isSingleValueType())
        return false;
    } else if (isa<SelectInst>(I)) {
      if (!Config.VectorizeSelect)
        return false;
    } else if (GetElementPtrInst *G = dyn_cast<GetElementPtrInst>(I)) {
      if (!Config.VectorizeGEP)
        return false;

      // Currently, vector GEPs exist only with one index.
      if (G->getNumIndices() != 1)
        return false;
    } else if (!(I->isBinaryOp() || isa<ShuffleVectorInst>(I) ||
        isa<ExtractElementInst>(I) || isa<InsertElementInst>(I))) {
      return false;
    }

    // We can't vectorize memory operations without target data
    if (TD == 0 && IsSimpleLoadStore)
      return false;

    Type *T1, *T2;
    if (isa<StoreInst>(I)) {
      // For stores, it is the value type, not the pointer type that matters
      // because the value is what will come from a vector register.

      Value *IVal = cast<StoreInst>(I)->getValueOperand();
      T1 = IVal->getType();
    } else {
      T1 = I->getType();
    }

    if (I->isCast())
      T2 = cast<CastInst>(I)->getSrcTy();
    else
      T2 = T1;

    // Not every type can be vectorized...
    if (!(VectorType::isValidElementType(T1) || T1->isVectorTy()) ||
        !(VectorType::isValidElementType(T2) || T2->isVectorTy()))
      return false;

    if (!Config.VectorizeInts
        && (T1->isIntOrIntVectorTy() || T2->isIntOrIntVectorTy()))
      return false;

    if (!Config.VectorizeFloats
        && (T1->isFPOrFPVectorTy() || T2->isFPOrFPVectorTy()))
      return false;

    if ((!Config.VectorizePointers || TD == 0)
        && ((T1->isPointerTy() ||
              (T1->isVectorTy() && T1->getScalarType()->isPointerTy())) ||
            (T2->isPointerTy() ||
              (T2->isVectorTy() && T2->getScalarType()->isPointerTy()))
           ))
      return false;

    if (T1->getPrimitiveSizeInBits() > Config.VectorBits/2 ||
        T2->getPrimitiveSizeInBits() > Config.VectorBits/2)
      return false;

    return true;
  }

  // This function returns true if the two provided instructions are compatible
  // (meaning that they can be fused into a vector instruction). This assumes
  // that I has already been determined to be vectorizable and that J is not
  // in the use tree of I.
  bool BBVectorize::areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore) {
    DEBUG(if (DebugInstructionExamination) dbgs() << "BBV: looking at " << *I <<
                     " <-> " << *J << "\n");

    // Loads and stores can be merged if they have different alignments,
    // but are otherwise the same.
    LoadInst *LI, *LJ;
    StoreInst *SI, *SJ;
    if ((LI = dyn_cast<LoadInst>(I)) && (LJ = dyn_cast<LoadInst>(J))) {
      if (I->getType() != J->getType())
        return false;

      if (LI->getPointerOperand()->getType() !=
            LJ->getPointerOperand()->getType() ||
          LI->isVolatile() != LJ->isVolatile() ||
          LI->getOrdering() != LJ->getOrdering() ||
          LI->getSynchScope() != LJ->getSynchScope())
        return false;
    } else if ((SI = dyn_cast<StoreInst>(I)) && (SJ = dyn_cast<StoreInst>(J))) {
      if (SI->getValueOperand()->getType() !=
            SJ->getValueOperand()->getType() ||
          SI->getPointerOperand()->getType() !=
            SJ->getPointerOperand()->getType() ||
          SI->isVolatile() != SJ->isVolatile() ||
          SI->getOrdering() != SJ->getOrdering() ||
          SI->getSynchScope() != SJ->getSynchScope())
        return false;
    } else if (!J->isSameOperationAs(I)) {
      return false;
    }
    // FIXME: handle addsub-type operations!

    if (IsSimpleLoadStore) {
      Value *IPtr, *JPtr;
      unsigned IAlignment, JAlignment;
      int64_t OffsetInElmts = 0;
      if (getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment,
            OffsetInElmts) && abs64(OffsetInElmts) == 1) {
        if (Config.AlignedOnly) {
          Type *aType = isa<StoreInst>(I) ?
            cast<StoreInst>(I)->getValueOperand()->getType() : I->getType();
          // An aligned load or store is possible only if the instruction
          // with the lower offset has an alignment suitable for the
          // vector type.

          unsigned BottomAlignment = IAlignment;
          if (OffsetInElmts < 0) BottomAlignment = JAlignment;

          Type *VType = getVecTypeForPair(aType);
          unsigned VecAlignment = TD->getPrefTypeAlignment(VType);
          if (BottomAlignment < VecAlignment)
            return false;
        }
      } else {
        return false;
      }
    } else if (isa<ShuffleVectorInst>(I)) {
      // Only merge two shuffles if they're both constant
      return isa<Constant>(I->getOperand(2)) &&
             isa<Constant>(J->getOperand(2));
      // FIXME: We may want to vectorize non-constant shuffles also.
    }

    // The powi intrinsic is special because only the first argument is
    // vectorized, the second arguments must be equal.
    CallInst *CI = dyn_cast<CallInst>(I);
    Function *FI;
    if (CI && (FI = CI->getCalledFunction()) &&
        FI->getIntrinsicID() == Intrinsic::powi) {

      Value *A1I = CI->getArgOperand(1),
            *A1J = cast<CallInst>(J)->getArgOperand(1);
      const SCEV *A1ISCEV = SE->getSCEV(A1I),
                 *A1JSCEV = SE->getSCEV(A1J);
      return (A1ISCEV == A1JSCEV);
    }

    return true;
  }

  // Figure out whether or not J uses I and update the users and write-set
  // structures associated with I. Specifically, Users represents the set of
  // instructions that depend on I. WriteSet represents the set
  // of memory locations that are dependent on I. If UpdateUsers is true,
  // and J uses I, then Users is updated to contain J and WriteSet is updated
  // to contain any memory locations to which J writes. The function returns
  // true if J uses I. By default, alias analysis is used to determine
  // whether J reads from memory that overlaps with a location in WriteSet.
  // If LoadMoveSet is not null, then it is a previously-computed multimap
  // where the key is the memory-based user instruction and the value is
  // the instruction to be compared with I. So, if LoadMoveSet is provided,
  // then the alias analysis is not used. This is necessary because this
  // function is called during the process of moving instructions during
  // vectorization and the results of the alias analysis are not stable during
  // that process.
  bool BBVectorize::trackUsesOfI(DenseSet<Value *> &Users,
                       AliasSetTracker &WriteSet, Instruction *I,
                       Instruction *J, bool UpdateUsers,
                       std::multimap<Value *, Value *> *LoadMoveSet) {
    bool UsesI = false;

    // This instruction may already be marked as a user due, for example, to
    // being a member of a selected pair.
    if (Users.count(J))
      UsesI = true;

    if (!UsesI)
      for (User::op_iterator JU = J->op_begin(), JE = J->op_end();
           JU != JE; ++JU) {
        Value *V = *JU;
        if (I == V || Users.count(V)) {
          UsesI = true;
          break;
        }
      }
    if (!UsesI && J->mayReadFromMemory()) {
      if (LoadMoveSet) {
        VPIteratorPair JPairRange = LoadMoveSet->equal_range(J);
        UsesI = isSecondInIteratorPair<Value*>(I, JPairRange);
      } else {
        for (AliasSetTracker::iterator W = WriteSet.begin(),
             WE = WriteSet.end(); W != WE; ++W) {
          if (W->aliasesUnknownInst(J, *AA)) {
            UsesI = true;
            break;
          }
        }
      }
    }

    if (UsesI && UpdateUsers) {
      if (J->mayWriteToMemory()) WriteSet.add(J);
      Users.insert(J);
    }

    return UsesI;
  }

  // This function iterates over all instruction pairs in the provided
  // basic block and collects all candidate pairs for vectorization.
  bool BBVectorize::getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts) {
    BasicBlock::iterator E = BB.end();
    if (Start == E) return false;

    bool ShouldContinue = false, IAfterStart = false;
    for (BasicBlock::iterator I = Start++; I != E; ++I) {
      if (I == Start) IAfterStart = true;

      bool IsSimpleLoadStore;
      if (!isInstVectorizable(I, IsSimpleLoadStore)) continue;

      // Look for an instruction with which to pair instruction *I...
      DenseSet<Value *> Users;
      AliasSetTracker WriteSet(*AA);
      bool JAfterStart = IAfterStart;
      BasicBlock::iterator J = llvm::next(I);
      for (unsigned ss = 0; J != E && ss <= Config.SearchLimit; ++J, ++ss) {
        if (J == Start) JAfterStart = true;

        // Determine if J uses I, if so, exit the loop.
        bool UsesI = trackUsesOfI(Users, WriteSet, I, J, !Config.FastDep);
        if (Config.FastDep) {
          // Note: For this heuristic to be effective, independent operations
          // must tend to be intermixed. This is likely to be true from some
          // kinds of grouped loop unrolling (but not the generic LLVM pass),
          // but otherwise may require some kind of reordering pass.

          // When using fast dependency analysis,
          // stop searching after first use:
          if (UsesI) break;
        } else {
          if (UsesI) continue;
        }

        // J does not use I, and comes before the first use of I, so it can be
        // merged with I if the instructions are compatible.
        if (!areInstsCompatible(I, J, IsSimpleLoadStore)) continue;

        // J is a candidate for merging with I.
        if (!PairableInsts.size() ||
             PairableInsts[PairableInsts.size()-1] != I) {
          PairableInsts.push_back(I);
        }

        CandidatePairs.insert(ValuePair(I, J));

        // The next call to this function must start after the last instruction
        // selected during this invocation.
        if (JAfterStart) {
          Start = llvm::next(J);
          IAfterStart = JAfterStart = false;
        }

        DEBUG(if (DebugCandidateSelection) dbgs() << "BBV: candidate pair "
                     << *I << " <-> " << *J << "\n");

        // If we have already found too many pairs, break here and this function
        // will be called again starting after the last instruction selected
        // during this invocation.
        if (PairableInsts.size() >= Config.MaxInsts) {
          ShouldContinue = true;
          break;
        }
      }

      if (ShouldContinue)
        break;
    }

    DEBUG(dbgs() << "BBV: found " << PairableInsts.size()
           << " instructions with candidate pairs\n");

    return ShouldContinue;
  }

  // Finds candidate pairs connected to the pair P = <PI, PJ>. This means that
  // it looks for pairs such that both members have an input which is an
  // output of PI or PJ.
  void BBVectorize::computePairsConnectedTo(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      ValuePair P) {
    StoreInst *SI, *SJ;

    // For each possible pairing for this variable, look at the uses of
    // the first value...
    for (Value::use_iterator I = P.first->use_begin(),
         E = P.first->use_end(); I != E; ++I) {
      if (isa<LoadInst>(*I)) {
        // A pair cannot be connected to a load because the load only takes one
        // operand (the address) and it is a scalar even after vectorization.
        continue;
      } else if ((SI = dyn_cast<StoreInst>(*I)) &&
                 P.first == SI->getPointerOperand()) {
        // Similarly, a pair cannot be connected to a store through its
        // pointer operand.
        continue;
      }

      VPIteratorPair IPairRange = CandidatePairs.equal_range(*I);

      // For each use of the first variable, look for uses of the second
      // variable...
      for (Value::use_iterator J = P.second->use_begin(),
           E2 = P.second->use_end(); J != E2; ++J) {
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.second == SJ->getPointerOperand())
          continue;

        VPIteratorPair JPairRange = CandidatePairs.equal_range(*J);

        // Look for <I, J>:
        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));

        // Look for <J, I>:
        if (isSecondInIteratorPair<Value*>(*I, JPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*J, *I)));
      }

      if (Config.SplatBreaksChain) continue;
      // Look for cases where just the first value in the pair is used by
      // both members of another pair (splatting).
      for (Value::use_iterator J = P.first->use_begin(); J != E; ++J) {
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.first == SJ->getPointerOperand())
          continue;

        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));
      }
    }

    if (Config.SplatBreaksChain) return;
    // Look for cases where just the second value in the pair is used by
    // both members of another pair (splatting).
    for (Value::use_iterator I = P.second->use_begin(),
         E = P.second->use_end(); I != E; ++I) {
      if (isa<LoadInst>(*I))
        continue;
      else if ((SI = dyn_cast<StoreInst>(*I)) &&
               P.second == SI->getPointerOperand())
        continue;

      VPIteratorPair IPairRange = CandidatePairs.equal_range(*I);

      for (Value::use_iterator J = P.second->use_begin(); J != E; ++J) {
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.second == SJ->getPointerOperand())
          continue;

        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));
      }
    }
  }

  // This function figures out which pairs are connected.  Two pairs are
  // connected if some output of the first pair forms an input to both members
  // of the second pair.
  void BBVectorize::computeConnectedPairs(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs) {

    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PE = PairableInsts.end(); PI != PE; ++PI) {
      VPIteratorPair choiceRange = CandidatePairs.equal_range(*PI);

      for (std::multimap<Value *, Value *>::iterator P = choiceRange.first;
           P != choiceRange.second; ++P)
        computePairsConnectedTo(CandidatePairs, PairableInsts,
                                ConnectedPairs, *P);
    }

    DEBUG(dbgs() << "BBV: found " << ConnectedPairs.size()
                 << " pair connections.\n");
  }

  // This function builds a set of use tuples such that <A, B> is in the set
  // if B is in the use tree of A. If B is in the use tree of A, then B
  // depends on the output of A.
  void BBVectorize::buildDepMap(
                      BasicBlock &BB,
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      DenseSet<ValuePair> &PairableInstUsers) {
    DenseSet<Value *> IsInPair;
    for (std::multimap<Value *, Value *>::iterator C = CandidatePairs.begin(),
         E = CandidatePairs.end(); C != E; ++C) {
      IsInPair.insert(C->first);
      IsInPair.insert(C->second);
    }

    // Iterate through the basic block, recording all Users of each
    // pairable instruction.

    BasicBlock::iterator E = BB.end();
    for (BasicBlock::iterator I = BB.getFirstInsertionPt(); I != E; ++I) {
      if (IsInPair.find(I) == IsInPair.end()) continue;

      DenseSet<Value *> Users;
      AliasSetTracker WriteSet(*AA);
      for (BasicBlock::iterator J = llvm::next(I); J != E; ++J)
        (void) trackUsesOfI(Users, WriteSet, I, J);

      for (DenseSet<Value *>::iterator U = Users.begin(), E = Users.end();
           U != E; ++U)
        PairableInstUsers.insert(ValuePair(I, *U));
    }
  }

  // Returns true if an input to pair P is an output of pair Q and also an
  // input of pair Q is an output of pair P. If this is the case, then these
  // two pairs cannot be simultaneously fused.
  bool BBVectorize::pairsConflict(ValuePair P, ValuePair Q,
                     DenseSet<ValuePair> &PairableInstUsers,
                     std::multimap<ValuePair, ValuePair> *PairableInstUserMap) {
    // Two pairs are in conflict if they are mutual Users of eachother.
    bool QUsesP = PairableInstUsers.count(ValuePair(P.first,  Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.first,  Q.second)) ||
                  PairableInstUsers.count(ValuePair(P.second, Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.second, Q.second));
    bool PUsesQ = PairableInstUsers.count(ValuePair(Q.first,  P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.first,  P.second)) ||
                  PairableInstUsers.count(ValuePair(Q.second, P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.second, P.second));
    if (PairableInstUserMap) {
      // FIXME: The expensive part of the cycle check is not so much the cycle
      // check itself but this edge insertion procedure. This needs some
      // profiling and probably a different data structure (same is true of
      // most uses of std::multimap).
      if (PUsesQ) {
        VPPIteratorPair QPairRange = PairableInstUserMap->equal_range(Q);
        if (!isSecondInIteratorPair(P, QPairRange))
          PairableInstUserMap->insert(VPPair(Q, P));
      }
      if (QUsesP) {
        VPPIteratorPair PPairRange = PairableInstUserMap->equal_range(P);
        if (!isSecondInIteratorPair(Q, PPairRange))
          PairableInstUserMap->insert(VPPair(P, Q));
      }
    }

    return (QUsesP && PUsesQ);
  }

  // This function walks the use graph of current pairs to see if, starting
  // from P, the walk returns to P.
  bool BBVectorize::pairWillFormCycle(ValuePair P,
                       std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                       DenseSet<ValuePair> &CurrentPairs) {
    DEBUG(if (DebugCycleCheck)
            dbgs() << "BBV: starting cycle check for : " << *P.first << " <-> "
                   << *P.second << "\n");
    // A lookup table of visisted pairs is kept because the PairableInstUserMap
    // contains non-direct associations.
    DenseSet<ValuePair> Visited;
    SmallVector<ValuePair, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(P);
    do {
      ValuePair QTop = Q.pop_back_val();
      Visited.insert(QTop);

      DEBUG(if (DebugCycleCheck)
              dbgs() << "BBV: cycle check visiting: " << *QTop.first << " <-> "
                     << *QTop.second << "\n");
      VPPIteratorPair QPairRange = PairableInstUserMap.equal_range(QTop);
      for (std::multimap<ValuePair, ValuePair>::iterator C = QPairRange.first;
           C != QPairRange.second; ++C) {
        if (C->second == P) {
          DEBUG(dbgs()
                 << "BBV: rejected to prevent non-trivial cycle formation: "
                 << *C->first.first << " <-> " << *C->first.second << "\n");
          return true;
        }

        if (CurrentPairs.count(C->second) && !Visited.count(C->second))
          Q.push_back(C->second);
      }
    } while (!Q.empty());

    return false;
  }

  // This function builds the initial tree of connected pairs with the
  // pair J at the root.
  void BBVectorize::buildInitialTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree, ValuePair J) {
    // Each of these pairs is viewed as the root node of a Tree. The Tree
    // is then walked (depth-first). As this happens, we keep track of
    // the pairs that compose the Tree and the maximum depth of the Tree.
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.back();

      // Push each child onto the queue:
      bool MoreChildren = false;
      size_t MaxChildDepth = QTop.second;
      VPPIteratorPair qtRange = ConnectedPairs.equal_range(QTop.first);
      for (std::multimap<ValuePair, ValuePair>::iterator k = qtRange.first;
           k != qtRange.second; ++k) {
        // Make sure that this child pair is still a candidate:
        bool IsStillCand = false;
        VPIteratorPair checkRange =
          CandidatePairs.equal_range(k->second.first);
        for (std::multimap<Value *, Value *>::iterator m = checkRange.first;
             m != checkRange.second; ++m) {
          if (m->second == k->second.second) {
            IsStillCand = true;
            break;
          }
        }

        if (IsStillCand) {
          DenseMap<ValuePair, size_t>::iterator C = Tree.find(k->second);
          if (C == Tree.end()) {
            size_t d = getDepthFactor(k->second.first);
            Q.push_back(ValuePairWithDepth(k->second, QTop.second+d));
            MoreChildren = true;
          } else {
            MaxChildDepth = std::max(MaxChildDepth, C->second);
          }
        }
      }

      if (!MoreChildren) {
        // Record the current pair as part of the Tree:
        Tree.insert(ValuePairWithDepth(QTop.first, MaxChildDepth));
        Q.pop_back();
      }
    } while (!Q.empty());
  }

  // Given some initial tree, prune it by removing conflicting pairs (pairs
  // that cannot be simultaneously chosen for vectorization).
  void BBVectorize::pruneTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree,
                      DenseSet<ValuePair> &PrunedTree, ValuePair J,
                      bool UseCycleCheck) {
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.pop_back_val();
      PrunedTree.insert(QTop.first);

      // Visit each child, pruning as necessary...
      DenseMap<ValuePair, size_t> BestChildren;
      VPPIteratorPair QTopRange = ConnectedPairs.equal_range(QTop.first);
      for (std::multimap<ValuePair, ValuePair>::iterator K = QTopRange.first;
           K != QTopRange.second; ++K) {
        DenseMap<ValuePair, size_t>::iterator C = Tree.find(K->second);
        if (C == Tree.end()) continue;

        // This child is in the Tree, now we need to make sure it is the
        // best of any conflicting children. There could be multiple
        // conflicting children, so first, determine if we're keeping
        // this child, then delete conflicting children as necessary.

        // It is also necessary to guard against pairing-induced
        // dependencies. Consider instructions a .. x .. y .. b
        // such that (a,b) are to be fused and (x,y) are to be fused
        // but a is an input to x and b is an output from y. This
        // means that y cannot be moved after b but x must be moved
        // after b for (a,b) to be fused. In other words, after
        // fusing (a,b) we have y .. a/b .. x where y is an input
        // to a/b and x is an output to a/b: x and y can no longer
        // be legally fused. To prevent this condition, we must
        // make sure that a child pair added to the Tree is not
        // both an input and output of an already-selected pair.

        // Pairing-induced dependencies can also form from more complicated
        // cycles. The pair vs. pair conflicts are easy to check, and so
        // that is done explicitly for "fast rejection", and because for
        // child vs. child conflicts, we may prefer to keep the current
        // pair in preference to the already-selected child.
        DenseSet<ValuePair> CurrentPairs;

        bool CanAdd = true;
        for (DenseMap<ValuePair, size_t>::iterator C2
              = BestChildren.begin(), E2 = BestChildren.end();
             C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            if (C2->second >= C->second) {
              CanAdd = false;
              break;
            }

            CurrentPairs.insert(C2->first);
          }
        }
        if (!CanAdd) continue;

        // Even worse, this child could conflict with another node already
        // selected for the Tree. If that is the case, ignore this child.
        for (DenseSet<ValuePair>::iterator T = PrunedTree.begin(),
             E2 = PrunedTree.end(); T != E2; ++T) {
          if (T->first == C->first.first ||
              T->first == C->first.second ||
              T->second == C->first.first ||
              T->second == C->first.second ||
              pairsConflict(*T, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*T);
        }
        if (!CanAdd) continue;

        // And check the queue too...
        for (SmallVector<ValuePairWithDepth, 32>::iterator C2 = Q.begin(),
             E2 = Q.end(); C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(C2->first);
        }
        if (!CanAdd) continue;

        // Last but not least, check for a conflict with any of the
        // already-chosen pairs.
        for (DenseMap<Value *, Value *>::iterator C2 =
              ChosenPairs.begin(), E2 = ChosenPairs.end();
             C2 != E2; ++C2) {
          if (pairsConflict(*C2, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*C2);
        }
        if (!CanAdd) continue;

        // To check for non-trivial cycles formed by the addition of the
        // current pair we've formed a list of all relevant pairs, now use a
        // graph walk to check for a cycle. We start from the current pair and
        // walk the use tree to see if we again reach the current pair. If we
        // do, then the current pair is rejected.

        // FIXME: It may be more efficient to use a topological-ordering
        // algorithm to improve the cycle check. This should be investigated.
        if (UseCycleCheck &&
            pairWillFormCycle(C->first, PairableInstUserMap, CurrentPairs))
          continue;

        // This child can be added, but we may have chosen it in preference
        // to an already-selected child. Check for this here, and if a
        // conflict is found, then remove the previously-selected child
        // before adding this one in its place.
        for (DenseMap<ValuePair, size_t>::iterator C2
              = BestChildren.begin(); C2 != BestChildren.end();) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers))
            BestChildren.erase(C2++);
          else
            ++C2;
        }

        BestChildren.insert(ValuePairWithDepth(C->first, C->second));
      }

      for (DenseMap<ValuePair, size_t>::iterator C
            = BestChildren.begin(), E2 = BestChildren.end();
           C != E2; ++C) {
        size_t DepthF = getDepthFactor(C->first.first);
        Q.push_back(ValuePairWithDepth(C->first, QTop.second+DepthF));
      }
    } while (!Q.empty());
  }

  // This function finds the best tree of mututally-compatible connected
  // pairs, given the choice of root pairs as an iterator range.
  void BBVectorize::findBestTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseSet<ValuePair> &BestTree, size_t &BestMaxDepth,
                      size_t &BestEffSize, VPIteratorPair ChoiceRange,
                      bool UseCycleCheck) {
    for (std::multimap<Value *, Value *>::iterator J = ChoiceRange.first;
         J != ChoiceRange.second; ++J) {

      // Before going any further, make sure that this pair does not
      // conflict with any already-selected pairs (see comment below
      // near the Tree pruning for more details).
      DenseSet<ValuePair> ChosenPairSet;
      bool DoesConflict = false;
      for (DenseMap<Value *, Value *>::iterator C = ChosenPairs.begin(),
           E = ChosenPairs.end(); C != E; ++C) {
        if (pairsConflict(*C, *J, PairableInstUsers,
                          UseCycleCheck ? &PairableInstUserMap : 0)) {
          DoesConflict = true;
          break;
        }

        ChosenPairSet.insert(*C);
      }
      if (DoesConflict) continue;

      if (UseCycleCheck &&
          pairWillFormCycle(*J, PairableInstUserMap, ChosenPairSet))
        continue;

      DenseMap<ValuePair, size_t> Tree;
      buildInitialTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                          PairableInstUsers, ChosenPairs, Tree, *J);

      // Because we'll keep the child with the largest depth, the largest
      // depth is still the same in the unpruned Tree.
      size_t MaxDepth = Tree.lookup(*J);

      DEBUG(if (DebugPairSelection) dbgs() << "BBV: found Tree for pair {"
                   << *J->first << " <-> " << *J->second << "} of depth " <<
                   MaxDepth << " and size " << Tree.size() << "\n");

      // At this point the Tree has been constructed, but, may contain
      // contradictory children (meaning that different children of
      // some tree node may be attempting to fuse the same instruction).
      // So now we walk the tree again, in the case of a conflict,
      // keep only the child with the largest depth. To break a tie,
      // favor the first child.

      DenseSet<ValuePair> PrunedTree;
      pruneTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                   PairableInstUsers, PairableInstUserMap, ChosenPairs, Tree,
                   PrunedTree, *J, UseCycleCheck);

      size_t EffSize = 0;
      for (DenseSet<ValuePair>::iterator S = PrunedTree.begin(),
           E = PrunedTree.end(); S != E; ++S)
        EffSize += getDepthFactor(S->first);

      DEBUG(if (DebugPairSelection)
             dbgs() << "BBV: found pruned Tree for pair {"
             << *J->first << " <-> " << *J->second << "} of depth " <<
             MaxDepth << " and size " << PrunedTree.size() <<
            " (effective size: " << EffSize << ")\n");
      if (MaxDepth >= Config.ReqChainDepth && EffSize > BestEffSize) {
        BestMaxDepth = MaxDepth;
        BestEffSize = EffSize;
        BestTree = PrunedTree;
      }
    }
  }

  // Given the list of candidate pairs, this function selects those
  // that will be fused into vector instructions.
  void BBVectorize::choosePairs(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      DenseMap<Value *, Value *>& ChosenPairs) {
    bool UseCycleCheck =
     CandidatePairs.size() <= Config.MaxCandPairsForCycleCheck;
    std::multimap<ValuePair, ValuePair> PairableInstUserMap;
    for (std::vector<Value *>::iterator I = PairableInsts.begin(),
         E = PairableInsts.end(); I != E; ++I) {
      // The number of possible pairings for this variable:
      size_t NumChoices = CandidatePairs.count(*I);
      if (!NumChoices) continue;

      VPIteratorPair ChoiceRange = CandidatePairs.equal_range(*I);

      // The best pair to choose and its tree:
      size_t BestMaxDepth = 0, BestEffSize = 0;
      DenseSet<ValuePair> BestTree;
      findBestTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                      PairableInstUsers, PairableInstUserMap, ChosenPairs,
                      BestTree, BestMaxDepth, BestEffSize, ChoiceRange,
                      UseCycleCheck);

      // A tree has been chosen (or not) at this point. If no tree was
      // chosen, then this instruction, I, cannot be paired (and is no longer
      // considered).

      DEBUG(if (BestTree.size() > 0)
              dbgs() << "BBV: selected pairs in the best tree for: "
                     << *cast<Instruction>(*I) << "\n");

      for (DenseSet<ValuePair>::iterator S = BestTree.begin(),
           SE2 = BestTree.end(); S != SE2; ++S) {
        // Insert the members of this tree into the list of chosen pairs.
        ChosenPairs.insert(ValuePair(S->first, S->second));
        DEBUG(dbgs() << "BBV: selected pair: " << *S->first << " <-> " <<
               *S->second << "\n");

        // Remove all candidate pairs that have values in the chosen tree.
        for (std::multimap<Value *, Value *>::iterator K =
               CandidatePairs.begin(); K != CandidatePairs.end();) {
          if (K->first == S->first || K->second == S->first ||
              K->second == S->second || K->first == S->second) {
            // Don't remove the actual pair chosen so that it can be used
            // in subsequent tree selections.
            if (!(K->first == S->first && K->second == S->second))
              CandidatePairs.erase(K++);
            else
              ++K;
          } else {
            ++K;
          }
        }
      }
    }

    DEBUG(dbgs() << "BBV: selected " << ChosenPairs.size() << " pairs.\n");
  }

  std::string getReplacementName(Instruction *I, bool IsInput, unsigned o,
                     unsigned n = 0) {
    if (!I->hasName())
      return "";

    return (I->getName() + (IsInput ? ".v.i" : ".v.r") + utostr(o) +
             (n > 0 ? "." + utostr(n) : "")).str();
  }

  // Returns the value that is to be used as the pointer input to the vector
  // instruction that fuses I with J.
  Value *BBVectorize::getReplacementPointerInput(LLVMContext& Context,
                     Instruction *I, Instruction *J, unsigned o,
                     bool &FlipMemInputs) {
    Value *IPtr, *JPtr;
    unsigned IAlignment, JAlignment;
    int64_t OffsetInElmts;
    (void) getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment,
                          OffsetInElmts);

    // The pointer value is taken to be the one with the lowest offset.
    Value *VPtr;
    if (OffsetInElmts > 0) {
      VPtr = IPtr;
    } else {
      FlipMemInputs = true;
      VPtr = JPtr;
    }

    Type *ArgType = cast<PointerType>(IPtr->getType())->getElementType();
    Type *VArgType = getVecTypeForPair(ArgType);
    Type *VArgPtrType = PointerType::get(VArgType,
      cast<PointerType>(IPtr->getType())->getAddressSpace());
    return new BitCastInst(VPtr, VArgPtrType, getReplacementName(I, true, o),
                        /* insert before */ FlipMemInputs ? J : I);
  }

  void BBVectorize::fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned NumElem, unsigned MaskOffset, unsigned NumInElem,
                     unsigned IdxOffset, std::vector<Constant*> &Mask) {
    for (unsigned v = 0; v < NumElem/2; ++v) {
      int m = cast<ShuffleVectorInst>(J)->getMaskValue(v);
      if (m < 0) {
        Mask[v+MaskOffset] = UndefValue::get(Type::getInt32Ty(Context));
      } else {
        unsigned mm = m + (int) IdxOffset;
        if (m >= (int) NumInElem)
          mm += (int) NumInElem;

        Mask[v+MaskOffset] =
          ConstantInt::get(Type::getInt32Ty(Context), mm);
      }
    }
  }

  // Returns the value that is to be used as the vector-shuffle mask to the
  // vector instruction that fuses I with J.
  Value *BBVectorize::getReplacementShuffleMask(LLVMContext& Context,
                     Instruction *I, Instruction *J) {
    // This is the shuffle mask. We need to append the second
    // mask to the first, and the numbers need to be adjusted.

    Type *ArgType = I->getType();
    Type *VArgType = getVecTypeForPair(ArgType);

    // Get the total number of elements in the fused vector type.
    // By definition, this must equal the number of elements in
    // the final mask.
    unsigned NumElem = cast<VectorType>(VArgType)->getNumElements();
    std::vector<Constant*> Mask(NumElem);

    Type *OpType = I->getOperand(0)->getType();
    unsigned NumInElem = cast<VectorType>(OpType)->getNumElements();

    // For the mask from the first pair...
    fillNewShuffleMask(Context, I, NumElem, 0, NumInElem, 0, Mask);

    // For the mask from the second pair...
    fillNewShuffleMask(Context, J, NumElem, NumElem/2, NumInElem, NumInElem,
                       Mask);

    return ConstantVector::get(Mask);
  }

  // Returns the value to be used as the specified operand of the vector
  // instruction that fuses I with J.
  Value *BBVectorize::getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool FlipMemInputs) {
    Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), 1);

      // Compute the fused vector type for this operand
    Type *ArgType = I->getOperand(o)->getType();
    VectorType *VArgType = getVecTypeForPair(ArgType);

    Instruction *L = I, *H = J;
    if (FlipMemInputs) {
      L = J;
      H = I;
    }

    if (ArgType->isVectorTy()) {
      unsigned numElem = cast<VectorType>(VArgType)->getNumElements();
      std::vector<Constant*> Mask(numElem);
      for (unsigned v = 0; v < numElem; ++v)
        Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);

      Instruction *BV = new ShuffleVectorInst(L->getOperand(o),
                                              H->getOperand(o),
                                              ConstantVector::get(Mask),
                                              getReplacementName(I, true, o));
      BV->insertBefore(J);
      return BV;
    }

    // If these two inputs are the output of another vector instruction,
    // then we should use that output directly. It might be necessary to
    // permute it first. [When pairings are fused recursively, you can
    // end up with cases where a large vector is decomposed into scalars
    // using extractelement instructions, then built into size-2
    // vectors using insertelement and the into larger vectors using
    // shuffles. InstCombine does not simplify all of these cases well,
    // and so we make sure that shuffles are generated here when possible.
    ExtractElementInst *LEE
      = dyn_cast<ExtractElementInst>(L->getOperand(o));
    ExtractElementInst *HEE
      = dyn_cast<ExtractElementInst>(H->getOperand(o));

    if (LEE && HEE &&
        LEE->getOperand(0)->getType() == HEE->getOperand(0)->getType()) {
      VectorType *EEType = cast<VectorType>(LEE->getOperand(0)->getType());
      unsigned LowIndx = cast<ConstantInt>(LEE->getOperand(1))->getZExtValue();
      unsigned HighIndx = cast<ConstantInt>(HEE->getOperand(1))->getZExtValue();
      if (LEE->getOperand(0) == HEE->getOperand(0)) {
        if (LowIndx == 0 && HighIndx == 1)
          return LEE->getOperand(0);

        std::vector<Constant*> Mask(2);
        Mask[0] = ConstantInt::get(Type::getInt32Ty(Context), LowIndx);
        Mask[1] = ConstantInt::get(Type::getInt32Ty(Context), HighIndx);

        Instruction *BV = new ShuffleVectorInst(LEE->getOperand(0),
                                          UndefValue::get(EEType),
                                          ConstantVector::get(Mask),
                                          getReplacementName(I, true, o));
        BV->insertBefore(J);
        return BV;
      }

      std::vector<Constant*> Mask(2);
      HighIndx += EEType->getNumElements();
      Mask[0] = ConstantInt::get(Type::getInt32Ty(Context), LowIndx);
      Mask[1] = ConstantInt::get(Type::getInt32Ty(Context), HighIndx);

      Instruction *BV = new ShuffleVectorInst(LEE->getOperand(0),
                                          HEE->getOperand(0),
                                          ConstantVector::get(Mask),
                                          getReplacementName(I, true, o));
      BV->insertBefore(J);
      return BV;
    }

    Instruction *BV1 = InsertElementInst::Create(
                                          UndefValue::get(VArgType),
                                          L->getOperand(o), CV0,
                                          getReplacementName(I, true, o, 1));
    BV1->insertBefore(I);
    Instruction *BV2 = InsertElementInst::Create(BV1, H->getOperand(o),
                                          CV1,
                                          getReplacementName(I, true, o, 2));
    BV2->insertBefore(J);
    return BV2;
  }

  // This function creates an array of values that will be used as the inputs
  // to the vector instruction that fuses I with J.
  void BBVectorize::getReplacementInputsForPair(LLVMContext& Context,
                     Instruction *I, Instruction *J,
                     SmallVector<Value *, 3> &ReplacedOperands,
                     bool &FlipMemInputs) {
    FlipMemInputs = false;
    unsigned NumOperands = I->getNumOperands();

    for (unsigned p = 0, o = NumOperands-1; p < NumOperands; ++p, --o) {
      // Iterate backward so that we look at the store pointer
      // first and know whether or not we need to flip the inputs.

      if (isa<LoadInst>(I) || (o == 1 && isa<StoreInst>(I))) {
        // This is the pointer for a load/store instruction.
        ReplacedOperands[o] = getReplacementPointerInput(Context, I, J, o,
                                FlipMemInputs);
        continue;
      } else if (isa<CallInst>(I)) {
        Function *F = cast<CallInst>(I)->getCalledFunction();
        unsigned IID = F->getIntrinsicID();
        if (o == NumOperands-1) {
          BasicBlock &BB = *I->getParent();

          Module *M = BB.getParent()->getParent();
          Type *ArgType = I->getType();
          Type *VArgType = getVecTypeForPair(ArgType);

          // FIXME: is it safe to do this here?
          ReplacedOperands[o] = Intrinsic::getDeclaration(M,
            (Intrinsic::ID) IID, VArgType);
          continue;
        } else if (IID == Intrinsic::powi && o == 1) {
          // The second argument of powi is a single integer and we've already
          // checked that both arguments are equal. As a result, we just keep
          // I's second argument.
          ReplacedOperands[o] = I->getOperand(o);
          continue;
        }
      } else if (isa<ShuffleVectorInst>(I) && o == NumOperands-1) {
        ReplacedOperands[o] = getReplacementShuffleMask(Context, I, J);
        continue;
      }

      ReplacedOperands[o] =
        getReplacementInput(Context, I, J, o, FlipMemInputs);
    }
  }

  // This function creates two values that represent the outputs of the
  // original I and J instructions. These are generally vector shuffles
  // or extracts. In many cases, these will end up being unused and, thus,
  // eliminated by later passes.
  void BBVectorize::replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt,
                     Instruction *&K1, Instruction *&K2,
                     bool &FlipMemInputs) {
    Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), 1);

    if (isa<StoreInst>(I)) {
      AA->replaceWithNewValue(I, K);
      AA->replaceWithNewValue(J, K);
    } else {
      Type *IType = I->getType();
      Type *VType = getVecTypeForPair(IType);

      if (IType->isVectorTy()) {
          unsigned numElem = cast<VectorType>(IType)->getNumElements();
          std::vector<Constant*> Mask1(numElem), Mask2(numElem);
          for (unsigned v = 0; v < numElem; ++v) {
            Mask1[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
            Mask2[v] = ConstantInt::get(Type::getInt32Ty(Context), numElem+v);
          }

          K1 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                       ConstantVector::get(
                                         FlipMemInputs ? Mask2 : Mask1),
                                       getReplacementName(K, false, 1));
          K2 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                       ConstantVector::get(
                                         FlipMemInputs ? Mask1 : Mask2),
                                       getReplacementName(K, false, 2));
      } else {
        K1 = ExtractElementInst::Create(K, FlipMemInputs ? CV1 : CV0,
                                          getReplacementName(K, false, 1));
        K2 = ExtractElementInst::Create(K, FlipMemInputs ? CV0 : CV1,
                                          getReplacementName(K, false, 2));
      }

      K1->insertAfter(K);
      K2->insertAfter(K1);
      InsertionPt = K2;
    }
  }

  // Move all uses of the function I (including pairing-induced uses) after J.
  bool BBVectorize::canMoveUsesOfIAfterJ(BasicBlock &BB,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I, Instruction *J) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = llvm::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    for (; cast<Instruction>(L) != J; ++L)
      (void) trackUsesOfI(Users, WriteSet, I, L, true, &LoadMoveSet);

    assert(cast<Instruction>(L) == J &&
      "Tracking has not proceeded far enough to check for dependencies");
    // If J is now in the use set of I, then trackUsesOfI will return true
    // and we have a dependency cycle (and the fusing operation must abort).
    return !trackUsesOfI(Users, WriteSet, I, J, true, &LoadMoveSet);
  }

  // Move all uses of the function I (including pairing-induced uses) after J.
  void BBVectorize::moveUsesOfIAfterJ(BasicBlock &BB,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = llvm::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    for (; cast<Instruction>(L) != J;) {
      if (trackUsesOfI(Users, WriteSet, I, L, true, &LoadMoveSet)) {
        // Move this instruction
        Instruction *InstToMove = L; ++L;

        DEBUG(dbgs() << "BBV: moving: " << *InstToMove <<
                        " to after " << *InsertionPt << "\n");
        InstToMove->removeFromParent();
        InstToMove->insertAfter(InsertionPt);
        InsertionPt = InstToMove;
      } else {
        ++L;
      }
    }
  }

  // Collect all load instruction that are in the move set of a given first
  // pair member.  These loads depend on the first instruction, I, and so need
  // to be moved after J (the second instruction) when the pair is fused.
  void BBVectorize::collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = llvm::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);

    // Note: We cannot end the loop when we reach J because J could be moved
    // farther down the use chain by another instruction pairing. Also, J
    // could be before I if this is an inverted input.
    for (BasicBlock::iterator E = BB.end(); cast<Instruction>(L) != E; ++L) {
      if (trackUsesOfI(Users, WriteSet, I, L)) {
        if (L->mayReadFromMemory())
          LoadMoveSet.insert(ValuePair(L, I));
      }
    }
  }

  // In cases where both load/stores and the computation of their pointers
  // are chosen for vectorization, we can end up in a situation where the
  // aliasing analysis starts returning different query results as the
  // process of fusing instruction pairs continues. Because the algorithm
  // relies on finding the same use trees here as were found earlier, we'll
  // need to precompute the necessary aliasing information here and then
  // manually update it during the fusion process.
  void BBVectorize::collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet) {
    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PIE = PairableInsts.end(); PI != PIE; ++PI) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(*PI);
      if (P == ChosenPairs.end()) continue;

      Instruction *I = cast<Instruction>(P->first);
      collectPairLoadMoveSet(BB, ChosenPairs, LoadMoveSet, I);
    }
  }

  // This function fuses the chosen instruction pairs into vector instructions,
  // taking care preserve any needed scalar outputs and, then, it reorders the
  // remaining instructions as needed (users of the first member of the pair
  // need to be moved to after the location of the second member of the pair
  // because the vector instruction is inserted in the location of the pair's
  // second member).
  void BBVectorize::fuseChosenPairs(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs) {
    LLVMContext& Context = BB.getContext();

    // During the vectorization process, the order of the pairs to be fused
    // could be flipped. So we'll add each pair, flipped, into the ChosenPairs
    // list. After a pair is fused, the flipped pair is removed from the list.
    std::vector<ValuePair> FlippedPairs;
    FlippedPairs.reserve(ChosenPairs.size());
    for (DenseMap<Value *, Value *>::iterator P = ChosenPairs.begin(),
         E = ChosenPairs.end(); P != E; ++P)
      FlippedPairs.push_back(ValuePair(P->second, P->first));
    for (std::vector<ValuePair>::iterator P = FlippedPairs.begin(),
         E = FlippedPairs.end(); P != E; ++P)
      ChosenPairs.insert(*P);

    std::multimap<Value *, Value *> LoadMoveSet;
    collectLoadMoveSet(BB, PairableInsts, ChosenPairs, LoadMoveSet);

    DEBUG(dbgs() << "BBV: initial: \n" << BB << "\n");

    for (BasicBlock::iterator PI = BB.getFirstInsertionPt(); PI != BB.end();) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(PI);
      if (P == ChosenPairs.end()) {
        ++PI;
        continue;
      }

      if (getDepthFactor(P->first) == 0) {
        // These instructions are not really fused, but are tracked as though
        // they are. Any case in which it would be interesting to fuse them
        // will be taken care of by InstCombine.
        --NumFusedOps;
        ++PI;
        continue;
      }

      Instruction *I = cast<Instruction>(P->first),
        *J = cast<Instruction>(P->second);

      DEBUG(dbgs() << "BBV: fusing: " << *I <<
             " <-> " << *J << "\n");

      // Remove the pair and flipped pair from the list.
      DenseMap<Value *, Value *>::iterator FP = ChosenPairs.find(P->second);
      assert(FP != ChosenPairs.end() && "Flipped pair not found in list");
      ChosenPairs.erase(FP);
      ChosenPairs.erase(P);

      if (!canMoveUsesOfIAfterJ(BB, LoadMoveSet, I, J)) {
        DEBUG(dbgs() << "BBV: fusion of: " << *I <<
               " <-> " << *J <<
               " aborted because of non-trivial dependency cycle\n");
        --NumFusedOps;
        ++PI;
        continue;
      }

      bool FlipMemInputs;
      unsigned NumOperands = I->getNumOperands();
      SmallVector<Value *, 3> ReplacedOperands(NumOperands);
      getReplacementInputsForPair(Context, I, J, ReplacedOperands,
        FlipMemInputs);

      // Make a copy of the original operation, change its type to the vector
      // type and replace its operands with the vector operands.
      Instruction *K = I->clone();
      if (I->hasName()) K->takeName(I);

      if (!isa<StoreInst>(K))
        K->mutateType(getVecTypeForPair(I->getType()));

      for (unsigned o = 0; o < NumOperands; ++o)
        K->setOperand(o, ReplacedOperands[o]);

      // If we've flipped the memory inputs, make sure that we take the correct
      // alignment.
      if (FlipMemInputs) {
        if (isa<StoreInst>(K))
          cast<StoreInst>(K)->setAlignment(cast<StoreInst>(J)->getAlignment());
        else
          cast<LoadInst>(K)->setAlignment(cast<LoadInst>(J)->getAlignment());
      }

      K->insertAfter(J);

      // Instruction insertion point:
      Instruction *InsertionPt = K;
      Instruction *K1 = 0, *K2 = 0;
      replaceOutputsOfPair(Context, I, J, K, InsertionPt, K1, K2,
        FlipMemInputs);

      // The use tree of the first original instruction must be moved to after
      // the location of the second instruction. The entire use tree of the
      // first instruction is disjoint from the input tree of the second
      // (by definition), and so commutes with it.

      moveUsesOfIAfterJ(BB, LoadMoveSet, InsertionPt, I, J);

      if (!isa<StoreInst>(I)) {
        I->replaceAllUsesWith(K1);
        J->replaceAllUsesWith(K2);
        AA->replaceWithNewValue(I, K1);
        AA->replaceWithNewValue(J, K2);
      }

      // Instructions that may read from memory may be in the load move set.
      // Once an instruction is fused, we no longer need its move set, and so
      // the values of the map never need to be updated. However, when a load
      // is fused, we need to merge the entries from both instructions in the
      // pair in case those instructions were in the move set of some other
      // yet-to-be-fused pair. The loads in question are the keys of the map.
      if (I->mayReadFromMemory()) {
        std::vector<ValuePair> NewSetMembers;
        VPIteratorPair IPairRange = LoadMoveSet.equal_range(I);
        VPIteratorPair JPairRange = LoadMoveSet.equal_range(J);
        for (std::multimap<Value *, Value *>::iterator N = IPairRange.first;
             N != IPairRange.second; ++N)
          NewSetMembers.push_back(ValuePair(K, N->second));
        for (std::multimap<Value *, Value *>::iterator N = JPairRange.first;
             N != JPairRange.second; ++N)
          NewSetMembers.push_back(ValuePair(K, N->second));
        for (std::vector<ValuePair>::iterator A = NewSetMembers.begin(),
             AE = NewSetMembers.end(); A != AE; ++A)
          LoadMoveSet.insert(*A);
      }

      // Before removing I, set the iterator to the next instruction.
      PI = llvm::next(BasicBlock::iterator(I));
      if (cast<Instruction>(PI) == J)
        ++PI;

      SE->forgetValue(I);
      SE->forgetValue(J);
      I->eraseFromParent();
      J->eraseFromParent();
    }

    DEBUG(dbgs() << "BBV: final: \n" << BB << "\n");
  }
}

char BBVectorize::ID = 0;
static const char bb_vectorize_name[] = "Basic-Block Vectorization";
INITIALIZE_PASS_BEGIN(BBVectorize, BBV_NAME, bb_vectorize_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(BBVectorize, BBV_NAME, bb_vectorize_name, false, false)

BasicBlockPass *llvm::createBBVectorizePass(const VectorizeConfig &C) {
  return new BBVectorize(C);
}

bool
llvm::vectorizeBasicBlock(Pass *P, BasicBlock &BB, const VectorizeConfig &C) {
  BBVectorize BBVectorizer(P, C);
  return BBVectorizer.vectorizeBB(BB);
}

//===----------------------------------------------------------------------===//
VectorizeConfig::VectorizeConfig() {
  VectorBits = ::VectorBits;
  VectorizeInts = !::NoInts;
  VectorizeFloats = !::NoFloats;
  VectorizePointers = !::NoPointers;
  VectorizeCasts = !::NoCasts;
  VectorizeMath = !::NoMath;
  VectorizeFMA = !::NoFMA;
  VectorizeSelect = !::NoSelect;
  VectorizeGEP = !::NoGEP;
  VectorizeMemOps = !::NoMemOps;
  AlignedOnly = ::AlignedOnly;
  ReqChainDepth= ::ReqChainDepth;
  SearchLimit = ::SearchLimit;
  MaxCandPairsForCycleCheck = ::MaxCandPairsForCycleCheck;
  SplatBreaksChain = ::SplatBreaksChain;
  MaxInsts = ::MaxInsts;
  MaxIter = ::MaxIter;
  NoMemOpBoost = ::NoMemOpBoost;
  FastDep = ::FastDep;
}
