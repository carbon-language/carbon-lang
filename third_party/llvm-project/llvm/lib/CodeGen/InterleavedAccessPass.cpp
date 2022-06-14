//===- InterleavedAccessPass.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Interleaved Access pass, which identifies
// interleaved memory accesses and transforms them into target specific
// intrinsics.
//
// An interleaved load reads data from memory into several vectors, with
// DE-interleaving the data on a factor. An interleaved store writes several
// vectors to memory with RE-interleaving the data on a factor.
//
// As interleaved accesses are difficult to identified in CodeGen (mainly
// because the VECTOR_SHUFFLE DAG node is quite different from the shufflevector
// IR), we identify and transform them to intrinsics in this pass so the
// intrinsics can be easily matched into target specific instructions later in
// CodeGen.
//
// E.g. An interleaved load (Factor = 2):
//        %wide.vec = load <8 x i32>, <8 x i32>* %ptr
//        %v0 = shuffle <8 x i32> %wide.vec, <8 x i32> poison, <0, 2, 4, 6>
//        %v1 = shuffle <8 x i32> %wide.vec, <8 x i32> poison, <1, 3, 5, 7>
//
// It could be transformed into a ld2 intrinsic in AArch64 backend or a vld2
// intrinsic in ARM backend.
//
// In X86, this can be further optimized into a set of target
// specific loads followed by an optimized sequence of shuffles.
//
// E.g. An interleaved store (Factor = 3):
//        %i.vec = shuffle <8 x i32> %v0, <8 x i32> %v1,
//                                    <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>
//        store <12 x i32> %i.vec, <12 x i32>* %ptr
//
// It could be transformed into a st3 intrinsic in AArch64 backend or a vst3
// intrinsic in ARM backend.
//
// Similarly, a set of interleaved stores can be transformed into an optimized
// sequence of shuffles followed by a set of target specific stores for X86.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "interleaved-access"

static cl::opt<bool> LowerInterleavedAccesses(
    "lower-interleaved-accesses",
    cl::desc("Enable lowering interleaved accesses to intrinsics"),
    cl::init(true), cl::Hidden);

namespace {

class InterleavedAccess : public FunctionPass {
public:
  static char ID;

  InterleavedAccess() : FunctionPass(ID) {
    initializeInterleavedAccessPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Interleaved Access Pass"; }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  DominatorTree *DT = nullptr;
  const TargetLowering *TLI = nullptr;

  /// The maximum supported interleave factor.
  unsigned MaxFactor;

  /// Transform an interleaved load into target specific intrinsics.
  bool lowerInterleavedLoad(LoadInst *LI,
                            SmallVector<Instruction *, 32> &DeadInsts);

  /// Transform an interleaved store into target specific intrinsics.
  bool lowerInterleavedStore(StoreInst *SI,
                             SmallVector<Instruction *, 32> &DeadInsts);

  /// Returns true if the uses of an interleaved load by the
  /// extractelement instructions in \p Extracts can be replaced by uses of the
  /// shufflevector instructions in \p Shuffles instead. If so, the necessary
  /// replacements are also performed.
  bool tryReplaceExtracts(ArrayRef<ExtractElementInst *> Extracts,
                          ArrayRef<ShuffleVectorInst *> Shuffles);

  /// Given a number of shuffles of the form shuffle(binop(x,y)), convert them
  /// to binop(shuffle(x), shuffle(y)) to allow the formation of an
  /// interleaving load. Any newly created shuffles that operate on \p LI will
  /// be added to \p Shuffles. Returns true, if any changes to the IR have been
  /// made.
  bool replaceBinOpShuffles(ArrayRef<ShuffleVectorInst *> BinOpShuffles,
                            SmallVectorImpl<ShuffleVectorInst *> &Shuffles,
                            LoadInst *LI);
};

} // end anonymous namespace.

char InterleavedAccess::ID = 0;

INITIALIZE_PASS_BEGIN(InterleavedAccess, DEBUG_TYPE,
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(InterleavedAccess, DEBUG_TYPE,
    "Lower interleaved memory accesses to target specific intrinsics", false,
    false)

FunctionPass *llvm::createInterleavedAccessPass() {
  return new InterleavedAccess();
}

/// Check if the mask is a DE-interleave mask of the given factor
/// \p Factor like:
///     <Index, Index+Factor, ..., Index+(NumElts-1)*Factor>
static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor,
                                       unsigned &Index) {
  // Check all potential start indices from 0 to (Factor - 1).
  for (Index = 0; Index < Factor; Index++) {
    unsigned i = 0;

    // Check that elements are in ascending order by Factor. Ignore undef
    // elements.
    for (; i < Mask.size(); i++)
      if (Mask[i] >= 0 && static_cast<unsigned>(Mask[i]) != Index + i * Factor)
        break;

    if (i == Mask.size())
      return true;
  }

  return false;
}

/// Check if the mask is a DE-interleave mask for an interleaved load.
///
/// E.g. DE-interleave masks (Factor = 2) could be:
///     <0, 2, 4, 6>    (mask of index 0 to extract even elements)
///     <1, 3, 5, 7>    (mask of index 1 to extract odd elements)
static bool isDeInterleaveMask(ArrayRef<int> Mask, unsigned &Factor,
                               unsigned &Index, unsigned MaxFactor,
                               unsigned NumLoadElements) {
  if (Mask.size() < 2)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++) {
    // Make sure we don't produce a load wider than the input load.
    if (Mask.size() * Factor > NumLoadElements)
      return false;
    if (isDeInterleaveMaskOfFactor(Mask, Factor, Index))
      return true;
  }

  return false;
}

/// Check if the mask can be used in an interleaved store.
//
/// It checks for a more general pattern than the RE-interleave mask.
/// I.e. <x, y, ... z, x+1, y+1, ...z+1, x+2, y+2, ...z+2, ...>
/// E.g. For a Factor of 2 (LaneLen=4): <4, 32, 5, 33, 6, 34, 7, 35>
/// E.g. For a Factor of 3 (LaneLen=4): <4, 32, 16, 5, 33, 17, 6, 34, 18, 7, 35, 19>
/// E.g. For a Factor of 4 (LaneLen=2): <8, 2, 12, 4, 9, 3, 13, 5>
///
/// The particular case of an RE-interleave mask is:
/// I.e. <0, LaneLen, ... , LaneLen*(Factor - 1), 1, LaneLen + 1, ...>
/// E.g. For a Factor of 2 (LaneLen=4): <0, 4, 1, 5, 2, 6, 3, 7>
static bool isReInterleaveMask(ArrayRef<int> Mask, unsigned &Factor,
                               unsigned MaxFactor, unsigned OpNumElts) {
  unsigned NumElts = Mask.size();
  if (NumElts < 4)
    return false;

  // Check potential Factors.
  for (Factor = 2; Factor <= MaxFactor; Factor++) {
    if (NumElts % Factor)
      continue;

    unsigned LaneLen = NumElts / Factor;
    if (!isPowerOf2_32(LaneLen))
      continue;

    // Check whether each element matches the general interleaved rule.
    // Ignore undef elements, as long as the defined elements match the rule.
    // Outer loop processes all factors (x, y, z in the above example)
    unsigned I = 0, J;
    for (; I < Factor; I++) {
      unsigned SavedLaneValue;
      unsigned SavedNoUndefs = 0;

      // Inner loop processes consecutive accesses (x, x+1... in the example)
      for (J = 0; J < LaneLen - 1; J++) {
        // Lane computes x's position in the Mask
        unsigned Lane = J * Factor + I;
        unsigned NextLane = Lane + Factor;
        int LaneValue = Mask[Lane];
        int NextLaneValue = Mask[NextLane];

        // If both are defined, values must be sequential
        if (LaneValue >= 0 && NextLaneValue >= 0 &&
            LaneValue + 1 != NextLaneValue)
          break;

        // If the next value is undef, save the current one as reference
        if (LaneValue >= 0 && NextLaneValue < 0) {
          SavedLaneValue = LaneValue;
          SavedNoUndefs = 1;
        }

        // Undefs are allowed, but defined elements must still be consecutive:
        // i.e.: x,..., undef,..., x + 2,..., undef,..., undef,..., x + 5, ....
        // Verify this by storing the last non-undef followed by an undef
        // Check that following non-undef masks are incremented with the
        // corresponding distance.
        if (SavedNoUndefs > 0 && LaneValue < 0) {
          SavedNoUndefs++;
          if (NextLaneValue >= 0 &&
              SavedLaneValue + SavedNoUndefs != (unsigned)NextLaneValue)
            break;
        }
      }

      if (J < LaneLen - 1)
        break;

      int StartMask = 0;
      if (Mask[I] >= 0) {
        // Check that the start of the I range (J=0) is greater than 0
        StartMask = Mask[I];
      } else if (Mask[(LaneLen - 1) * Factor + I] >= 0) {
        // StartMask defined by the last value in lane
        StartMask = Mask[(LaneLen - 1) * Factor + I] - J;
      } else if (SavedNoUndefs > 0) {
        // StartMask defined by some non-zero value in the j loop
        StartMask = SavedLaneValue - (LaneLen - 1 - SavedNoUndefs);
      }
      // else StartMask remains set to 0, i.e. all elements are undefs

      if (StartMask < 0)
        break;
      // We must stay within the vectors; This case can happen with undefs.
      if (StartMask + LaneLen > OpNumElts*2)
        break;
    }

    // Found an interleaved mask of current factor.
    if (I == Factor)
      return true;
  }

  return false;
}

bool InterleavedAccess::lowerInterleavedLoad(
    LoadInst *LI, SmallVector<Instruction *, 32> &DeadInsts) {
  if (!LI->isSimple() || isa<ScalableVectorType>(LI->getType()))
    return false;

  // Check if all users of this load are shufflevectors. If we encounter any
  // users that are extractelement instructions or binary operators, we save
  // them to later check if they can be modified to extract from one of the
  // shufflevectors instead of the load.

  SmallVector<ShuffleVectorInst *, 4> Shuffles;
  SmallVector<ExtractElementInst *, 4> Extracts;
  // BinOpShuffles need to be handled a single time in case both operands of the
  // binop are the same load.
  SmallSetVector<ShuffleVectorInst *, 4> BinOpShuffles;

  for (auto *User : LI->users()) {
    auto *Extract = dyn_cast<ExtractElementInst>(User);
    if (Extract && isa<ConstantInt>(Extract->getIndexOperand())) {
      Extracts.push_back(Extract);
      continue;
    }
    auto *BI = dyn_cast<BinaryOperator>(User);
    if (BI && BI->hasOneUse()) {
      if (auto *SVI = dyn_cast<ShuffleVectorInst>(*BI->user_begin())) {
        BinOpShuffles.insert(SVI);
        continue;
      }
    }
    auto *SVI = dyn_cast<ShuffleVectorInst>(User);
    if (!SVI || !isa<UndefValue>(SVI->getOperand(1)))
      return false;

    Shuffles.push_back(SVI);
  }

  if (Shuffles.empty() && BinOpShuffles.empty())
    return false;

  unsigned Factor, Index;

  unsigned NumLoadElements =
      cast<FixedVectorType>(LI->getType())->getNumElements();
  auto *FirstSVI = Shuffles.size() > 0 ? Shuffles[0] : BinOpShuffles[0];
  // Check if the first shufflevector is DE-interleave shuffle.
  if (!isDeInterleaveMask(FirstSVI->getShuffleMask(), Factor, Index, MaxFactor,
                          NumLoadElements))
    return false;

  // Holds the corresponding index for each DE-interleave shuffle.
  SmallVector<unsigned, 4> Indices;

  Type *VecTy = FirstSVI->getType();

  // Check if other shufflevectors are also DE-interleaved of the same type
  // and factor as the first shufflevector.
  for (auto *Shuffle : Shuffles) {
    if (Shuffle->getType() != VecTy)
      return false;
    if (!isDeInterleaveMaskOfFactor(Shuffle->getShuffleMask(), Factor,
                                    Index))
      return false;

    assert(Shuffle->getShuffleMask().size() <= NumLoadElements);
    Indices.push_back(Index);
  }
  for (auto *Shuffle : BinOpShuffles) {
    if (Shuffle->getType() != VecTy)
      return false;
    if (!isDeInterleaveMaskOfFactor(Shuffle->getShuffleMask(), Factor,
                                    Index))
      return false;

    assert(Shuffle->getShuffleMask().size() <= NumLoadElements);

    if (cast<Instruction>(Shuffle->getOperand(0))->getOperand(0) == LI)
      Indices.push_back(Index);
    if (cast<Instruction>(Shuffle->getOperand(0))->getOperand(1) == LI)
      Indices.push_back(Index);
  }

  // Try and modify users of the load that are extractelement instructions to
  // use the shufflevector instructions instead of the load.
  if (!tryReplaceExtracts(Extracts, Shuffles))
    return false;

  bool BinOpShuffleChanged =
      replaceBinOpShuffles(BinOpShuffles.getArrayRef(), Shuffles, LI);

  LLVM_DEBUG(dbgs() << "IA: Found an interleaved load: " << *LI << "\n");

  // Try to create target specific intrinsics to replace the load and shuffles.
  if (!TLI->lowerInterleavedLoad(LI, Shuffles, Indices, Factor)) {
    // If Extracts is not empty, tryReplaceExtracts made changes earlier.
    return !Extracts.empty() || BinOpShuffleChanged;
  }

  append_range(DeadInsts, Shuffles);

  DeadInsts.push_back(LI);
  return true;
}

bool InterleavedAccess::replaceBinOpShuffles(
    ArrayRef<ShuffleVectorInst *> BinOpShuffles,
    SmallVectorImpl<ShuffleVectorInst *> &Shuffles, LoadInst *LI) {
  for (auto *SVI : BinOpShuffles) {
    BinaryOperator *BI = cast<BinaryOperator>(SVI->getOperand(0));
    Type *BIOp0Ty = BI->getOperand(0)->getType();
    ArrayRef<int> Mask = SVI->getShuffleMask();
    assert(all_of(Mask, [&](int Idx) {
      return Idx < (int)cast<FixedVectorType>(BIOp0Ty)->getNumElements();
    }));

    auto *NewSVI1 =
        new ShuffleVectorInst(BI->getOperand(0), PoisonValue::get(BIOp0Ty),
                              Mask, SVI->getName(), SVI);
    auto *NewSVI2 = new ShuffleVectorInst(
        BI->getOperand(1), PoisonValue::get(BI->getOperand(1)->getType()), Mask,
        SVI->getName(), SVI);
    BinaryOperator *NewBI = BinaryOperator::CreateWithCopiedFlags(
        BI->getOpcode(), NewSVI1, NewSVI2, BI, BI->getName(), SVI);
    SVI->replaceAllUsesWith(NewBI);
    LLVM_DEBUG(dbgs() << "  Replaced: " << *BI << "\n    And   : " << *SVI
                      << "\n  With    : " << *NewSVI1 << "\n    And   : "
                      << *NewSVI2 << "\n    And   : " << *NewBI << "\n");
    RecursivelyDeleteTriviallyDeadInstructions(SVI);
    if (NewSVI1->getOperand(0) == LI)
      Shuffles.push_back(NewSVI1);
    if (NewSVI2->getOperand(0) == LI)
      Shuffles.push_back(NewSVI2);
  }

  return !BinOpShuffles.empty();
}

bool InterleavedAccess::tryReplaceExtracts(
    ArrayRef<ExtractElementInst *> Extracts,
    ArrayRef<ShuffleVectorInst *> Shuffles) {
  // If there aren't any extractelement instructions to modify, there's nothing
  // to do.
  if (Extracts.empty())
    return true;

  // Maps extractelement instructions to vector-index pairs. The extractlement
  // instructions will be modified to use the new vector and index operands.
  DenseMap<ExtractElementInst *, std::pair<Value *, int>> ReplacementMap;

  for (auto *Extract : Extracts) {
    // The vector index that is extracted.
    auto *IndexOperand = cast<ConstantInt>(Extract->getIndexOperand());
    auto Index = IndexOperand->getSExtValue();

    // Look for a suitable shufflevector instruction. The goal is to modify the
    // extractelement instruction (which uses an interleaved load) to use one
    // of the shufflevector instructions instead of the load.
    for (auto *Shuffle : Shuffles) {
      // If the shufflevector instruction doesn't dominate the extract, we
      // can't create a use of it.
      if (!DT->dominates(Shuffle, Extract))
        continue;

      // Inspect the indices of the shufflevector instruction. If the shuffle
      // selects the same index that is extracted, we can modify the
      // extractelement instruction.
      SmallVector<int, 4> Indices;
      Shuffle->getShuffleMask(Indices);
      for (unsigned I = 0; I < Indices.size(); ++I)
        if (Indices[I] == Index) {
          assert(Extract->getOperand(0) == Shuffle->getOperand(0) &&
                 "Vector operations do not match");
          ReplacementMap[Extract] = std::make_pair(Shuffle, I);
          break;
        }

      // If we found a suitable shufflevector instruction, stop looking.
      if (ReplacementMap.count(Extract))
        break;
    }

    // If we did not find a suitable shufflevector instruction, the
    // extractelement instruction cannot be modified, so we must give up.
    if (!ReplacementMap.count(Extract))
      return false;
  }

  // Finally, perform the replacements.
  IRBuilder<> Builder(Extracts[0]->getContext());
  for (auto &Replacement : ReplacementMap) {
    auto *Extract = Replacement.first;
    auto *Vector = Replacement.second.first;
    auto Index = Replacement.second.second;
    Builder.SetInsertPoint(Extract);
    Extract->replaceAllUsesWith(Builder.CreateExtractElement(Vector, Index));
    Extract->eraseFromParent();
  }

  return true;
}

bool InterleavedAccess::lowerInterleavedStore(
    StoreInst *SI, SmallVector<Instruction *, 32> &DeadInsts) {
  if (!SI->isSimple())
    return false;

  auto *SVI = dyn_cast<ShuffleVectorInst>(SI->getValueOperand());
  if (!SVI || !SVI->hasOneUse() || isa<ScalableVectorType>(SVI->getType()))
    return false;

  // Check if the shufflevector is RE-interleave shuffle.
  unsigned Factor;
  unsigned OpNumElts =
      cast<FixedVectorType>(SVI->getOperand(0)->getType())->getNumElements();
  if (!isReInterleaveMask(SVI->getShuffleMask(), Factor, MaxFactor, OpNumElts))
    return false;

  LLVM_DEBUG(dbgs() << "IA: Found an interleaved store: " << *SI << "\n");

  // Try to create target specific intrinsics to replace the store and shuffle.
  if (!TLI->lowerInterleavedStore(SI, SVI, Factor))
    return false;

  // Already have a new target specific interleaved store. Erase the old store.
  DeadInsts.push_back(SI);
  DeadInsts.push_back(SVI);
  return true;
}

bool InterleavedAccess::runOnFunction(Function &F) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC || !LowerInterleavedAccesses)
    return false;

  LLVM_DEBUG(dbgs() << "*** " << getPassName() << ": " << F.getName() << "\n");

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &TM = TPC->getTM<TargetMachine>();
  TLI = TM.getSubtargetImpl(F)->getTargetLowering();
  MaxFactor = TLI->getMaxSupportedInterleaveFactor();

  // Holds dead instructions that will be erased later.
  SmallVector<Instruction *, 32> DeadInsts;
  bool Changed = false;

  for (auto &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I))
      Changed |= lowerInterleavedLoad(LI, DeadInsts);

    if (auto *SI = dyn_cast<StoreInst>(&I))
      Changed |= lowerInterleavedStore(SI, DeadInsts);
  }

  for (auto I : DeadInsts)
    I->eraseFromParent();

  return Changed;
}
