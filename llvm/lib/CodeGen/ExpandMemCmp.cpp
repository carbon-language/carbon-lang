//===--- ExpandMemCmp.cpp - Expand memcmp() to load/stores ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to expand memcmp() calls into optimally-sized loads and
// compares for the target.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SizeOpts.h"

using namespace llvm;

#define DEBUG_TYPE "expandmemcmp"

STATISTIC(NumMemCmpCalls, "Number of memcmp calls");
STATISTIC(NumMemCmpNotConstant, "Number of memcmp calls without constant size");
STATISTIC(NumMemCmpGreaterThanMax,
          "Number of memcmp calls with size greater than max size");
STATISTIC(NumMemCmpInlined, "Number of inlined memcmp calls");

static cl::opt<unsigned> MemCmpEqZeroNumLoadsPerBlock(
    "memcmp-num-loads-per-block", cl::Hidden, cl::init(1),
    cl::desc("The number of loads per basic block for inline expansion of "
             "memcmp that is only being compared against zero."));

static cl::opt<unsigned> MaxLoadsPerMemcmp(
    "max-loads-per-memcmp", cl::Hidden,
    cl::desc("Set maximum number of loads used in expanded memcmp"));

static cl::opt<unsigned> MaxLoadsPerMemcmpOptSize(
    "max-loads-per-memcmp-opt-size", cl::Hidden,
    cl::desc("Set maximum number of loads used in expanded memcmp for -Os/Oz"));

namespace {


// This class provides helper functions to expand a memcmp library call into an
// inline expansion.
class MemCmpExpansion {
  struct ResultBlock {
    BasicBlock *BB = nullptr;
    PHINode *PhiSrc1 = nullptr;
    PHINode *PhiSrc2 = nullptr;

    ResultBlock() = default;
  };

  CallInst *const CI;
  ResultBlock ResBlock;
  const uint64_t Size;
  unsigned MaxLoadSize;
  uint64_t NumLoadsNonOneByte;
  const uint64_t NumLoadsPerBlockForZeroCmp;
  std::vector<BasicBlock *> LoadCmpBlocks;
  BasicBlock *EndBlock;
  PHINode *PhiRes;
  const bool IsUsedForZeroCmp;
  const DataLayout &DL;
  DomTreeUpdater *DTU;
  IRBuilder<> Builder;
  // Represents the decomposition in blocks of the expansion. For example,
  // comparing 33 bytes on X86+sse can be done with 2x16-byte loads and
  // 1x1-byte load, which would be represented as [{16, 0}, {16, 16}, {1, 32}.
  struct LoadEntry {
    LoadEntry(unsigned LoadSize, uint64_t Offset)
        : LoadSize(LoadSize), Offset(Offset) {
    }

    // The size of the load for this block, in bytes.
    unsigned LoadSize;
    // The offset of this load from the base pointer, in bytes.
    uint64_t Offset;
  };
  using LoadEntryVector = SmallVector<LoadEntry, 8>;
  LoadEntryVector LoadSequence;

  void createLoadCmpBlocks();
  void createResultBlock();
  void setupResultBlockPHINodes();
  void setupEndBlockPHINodes();
  Value *getCompareLoadPairs(unsigned BlockIndex, unsigned &LoadIndex);
  void emitLoadCompareBlock(unsigned BlockIndex);
  void emitLoadCompareBlockMultipleLoads(unsigned BlockIndex,
                                         unsigned &LoadIndex);
  void emitLoadCompareByteBlock(unsigned BlockIndex, unsigned OffsetBytes);
  void emitMemCmpResultBlock();
  Value *getMemCmpExpansionZeroCase();
  Value *getMemCmpEqZeroOneBlock();
  Value *getMemCmpOneBlock();
  struct LoadPair {
    Value *Lhs = nullptr;
    Value *Rhs = nullptr;
  };
  LoadPair getLoadPair(Type *LoadSizeType, bool NeedsBSwap, Type *CmpSizeType,
                       unsigned OffsetBytes);

  static LoadEntryVector
  computeGreedyLoadSequence(uint64_t Size, llvm::ArrayRef<unsigned> LoadSizes,
                            unsigned MaxNumLoads, unsigned &NumLoadsNonOneByte);
  static LoadEntryVector
  computeOverlappingLoadSequence(uint64_t Size, unsigned MaxLoadSize,
                                 unsigned MaxNumLoads,
                                 unsigned &NumLoadsNonOneByte);

public:
  MemCmpExpansion(CallInst *CI, uint64_t Size,
                  const TargetTransformInfo::MemCmpExpansionOptions &Options,
                  const bool IsUsedForZeroCmp, const DataLayout &TheDataLayout,
                  DomTreeUpdater *DTU);

  unsigned getNumBlocks();
  uint64_t getNumLoads() const { return LoadSequence.size(); }

  Value *getMemCmpExpansion();
};

MemCmpExpansion::LoadEntryVector MemCmpExpansion::computeGreedyLoadSequence(
    uint64_t Size, llvm::ArrayRef<unsigned> LoadSizes,
    const unsigned MaxNumLoads, unsigned &NumLoadsNonOneByte) {
  NumLoadsNonOneByte = 0;
  LoadEntryVector LoadSequence;
  uint64_t Offset = 0;
  while (Size && !LoadSizes.empty()) {
    const unsigned LoadSize = LoadSizes.front();
    const uint64_t NumLoadsForThisSize = Size / LoadSize;
    if (LoadSequence.size() + NumLoadsForThisSize > MaxNumLoads) {
      // Do not expand if the total number of loads is larger than what the
      // target allows. Note that it's important that we exit before completing
      // the expansion to avoid using a ton of memory to store the expansion for
      // large sizes.
      return {};
    }
    if (NumLoadsForThisSize > 0) {
      for (uint64_t I = 0; I < NumLoadsForThisSize; ++I) {
        LoadSequence.push_back({LoadSize, Offset});
        Offset += LoadSize;
      }
      if (LoadSize > 1)
        ++NumLoadsNonOneByte;
      Size = Size % LoadSize;
    }
    LoadSizes = LoadSizes.drop_front();
  }
  return LoadSequence;
}

MemCmpExpansion::LoadEntryVector
MemCmpExpansion::computeOverlappingLoadSequence(uint64_t Size,
                                                const unsigned MaxLoadSize,
                                                const unsigned MaxNumLoads,
                                                unsigned &NumLoadsNonOneByte) {
  // These are already handled by the greedy approach.
  if (Size < 2 || MaxLoadSize < 2)
    return {};

  // We try to do as many non-overlapping loads as possible starting from the
  // beginning.
  const uint64_t NumNonOverlappingLoads = Size / MaxLoadSize;
  assert(NumNonOverlappingLoads && "there must be at least one load");
  // There remain 0 to (MaxLoadSize - 1) bytes to load, this will be done with
  // an overlapping load.
  Size = Size - NumNonOverlappingLoads * MaxLoadSize;
  // Bail if we do not need an overloapping store, this is already handled by
  // the greedy approach.
  if (Size == 0)
    return {};
  // Bail if the number of loads (non-overlapping + potential overlapping one)
  // is larger than the max allowed.
  if ((NumNonOverlappingLoads + 1) > MaxNumLoads)
    return {};

  // Add non-overlapping loads.
  LoadEntryVector LoadSequence;
  uint64_t Offset = 0;
  for (uint64_t I = 0; I < NumNonOverlappingLoads; ++I) {
    LoadSequence.push_back({MaxLoadSize, Offset});
    Offset += MaxLoadSize;
  }

  // Add the last overlapping load.
  assert(Size > 0 && Size < MaxLoadSize && "broken invariant");
  LoadSequence.push_back({MaxLoadSize, Offset - (MaxLoadSize - Size)});
  NumLoadsNonOneByte = 1;
  return LoadSequence;
}

// Initialize the basic block structure required for expansion of memcmp call
// with given maximum load size and memcmp size parameter.
// This structure includes:
// 1. A list of load compare blocks - LoadCmpBlocks.
// 2. An EndBlock, split from original instruction point, which is the block to
// return from.
// 3. ResultBlock, block to branch to for early exit when a
// LoadCmpBlock finds a difference.
MemCmpExpansion::MemCmpExpansion(
    CallInst *const CI, uint64_t Size,
    const TargetTransformInfo::MemCmpExpansionOptions &Options,
    const bool IsUsedForZeroCmp, const DataLayout &TheDataLayout,
    DomTreeUpdater *DTU)
    : CI(CI), Size(Size), MaxLoadSize(0), NumLoadsNonOneByte(0),
      NumLoadsPerBlockForZeroCmp(Options.NumLoadsPerBlock),
      IsUsedForZeroCmp(IsUsedForZeroCmp), DL(TheDataLayout), DTU(DTU),
      Builder(CI) {
  assert(Size > 0 && "zero blocks");
  // Scale the max size down if the target can load more bytes than we need.
  llvm::ArrayRef<unsigned> LoadSizes(Options.LoadSizes);
  while (!LoadSizes.empty() && LoadSizes.front() > Size) {
    LoadSizes = LoadSizes.drop_front();
  }
  assert(!LoadSizes.empty() && "cannot load Size bytes");
  MaxLoadSize = LoadSizes.front();
  // Compute the decomposition.
  unsigned GreedyNumLoadsNonOneByte = 0;
  LoadSequence = computeGreedyLoadSequence(Size, LoadSizes, Options.MaxNumLoads,
                                           GreedyNumLoadsNonOneByte);
  NumLoadsNonOneByte = GreedyNumLoadsNonOneByte;
  assert(LoadSequence.size() <= Options.MaxNumLoads && "broken invariant");
  // If we allow overlapping loads and the load sequence is not already optimal,
  // use overlapping loads.
  if (Options.AllowOverlappingLoads &&
      (LoadSequence.empty() || LoadSequence.size() > 2)) {
    unsigned OverlappingNumLoadsNonOneByte = 0;
    auto OverlappingLoads = computeOverlappingLoadSequence(
        Size, MaxLoadSize, Options.MaxNumLoads, OverlappingNumLoadsNonOneByte);
    if (!OverlappingLoads.empty() &&
        (LoadSequence.empty() ||
         OverlappingLoads.size() < LoadSequence.size())) {
      LoadSequence = OverlappingLoads;
      NumLoadsNonOneByte = OverlappingNumLoadsNonOneByte;
    }
  }
  assert(LoadSequence.size() <= Options.MaxNumLoads && "broken invariant");
}

unsigned MemCmpExpansion::getNumBlocks() {
  if (IsUsedForZeroCmp)
    return getNumLoads() / NumLoadsPerBlockForZeroCmp +
           (getNumLoads() % NumLoadsPerBlockForZeroCmp != 0 ? 1 : 0);
  return getNumLoads();
}

void MemCmpExpansion::createLoadCmpBlocks() {
  for (unsigned i = 0; i < getNumBlocks(); i++) {
    BasicBlock *BB = BasicBlock::Create(CI->getContext(), "loadbb",
                                        EndBlock->getParent(), EndBlock);
    LoadCmpBlocks.push_back(BB);
  }
}

void MemCmpExpansion::createResultBlock() {
  ResBlock.BB = BasicBlock::Create(CI->getContext(), "res_block",
                                   EndBlock->getParent(), EndBlock);
}

MemCmpExpansion::LoadPair MemCmpExpansion::getLoadPair(Type *LoadSizeType,
                                                       bool NeedsBSwap,
                                                       Type *CmpSizeType,
                                                       unsigned OffsetBytes) {
  // Get the memory source at offset `OffsetBytes`.
  Value *LhsSource = CI->getArgOperand(0);
  Value *RhsSource = CI->getArgOperand(1);
  Align LhsAlign = LhsSource->getPointerAlignment(DL);
  Align RhsAlign = RhsSource->getPointerAlignment(DL);
  if (OffsetBytes > 0) {
    auto *ByteType = Type::getInt8Ty(CI->getContext());
    LhsSource = Builder.CreateConstGEP1_64(
        ByteType, Builder.CreateBitCast(LhsSource, ByteType->getPointerTo()),
        OffsetBytes);
    RhsSource = Builder.CreateConstGEP1_64(
        ByteType, Builder.CreateBitCast(RhsSource, ByteType->getPointerTo()),
        OffsetBytes);
    LhsAlign = commonAlignment(LhsAlign, OffsetBytes);
    RhsAlign = commonAlignment(RhsAlign, OffsetBytes);
  }
  LhsSource = Builder.CreateBitCast(LhsSource, LoadSizeType->getPointerTo());
  RhsSource = Builder.CreateBitCast(RhsSource, LoadSizeType->getPointerTo());

  // Create a constant or a load from the source.
  Value *Lhs = nullptr;
  if (auto *C = dyn_cast<Constant>(LhsSource))
    Lhs = ConstantFoldLoadFromConstPtr(C, LoadSizeType, DL);
  if (!Lhs)
    Lhs = Builder.CreateAlignedLoad(LoadSizeType, LhsSource, LhsAlign);

  Value *Rhs = nullptr;
  if (auto *C = dyn_cast<Constant>(RhsSource))
    Rhs = ConstantFoldLoadFromConstPtr(C, LoadSizeType, DL);
  if (!Rhs)
    Rhs = Builder.CreateAlignedLoad(LoadSizeType, RhsSource, RhsAlign);

  // Swap bytes if required.
  if (NeedsBSwap) {
    Function *Bswap = Intrinsic::getDeclaration(CI->getModule(),
                                                Intrinsic::bswap, LoadSizeType);
    Lhs = Builder.CreateCall(Bswap, Lhs);
    Rhs = Builder.CreateCall(Bswap, Rhs);
  }

  // Zero extend if required.
  if (CmpSizeType != nullptr && CmpSizeType != LoadSizeType) {
    Lhs = Builder.CreateZExt(Lhs, CmpSizeType);
    Rhs = Builder.CreateZExt(Rhs, CmpSizeType);
  }
  return {Lhs, Rhs};
}

// This function creates the IR instructions for loading and comparing 1 byte.
// It loads 1 byte from each source of the memcmp parameters with the given
// GEPIndex. It then subtracts the two loaded values and adds this result to the
// final phi node for selecting the memcmp result.
void MemCmpExpansion::emitLoadCompareByteBlock(unsigned BlockIndex,
                                               unsigned OffsetBytes) {
  BasicBlock *BB = LoadCmpBlocks[BlockIndex];
  Builder.SetInsertPoint(BB);
  const LoadPair Loads =
      getLoadPair(Type::getInt8Ty(CI->getContext()), /*NeedsBSwap=*/false,
                  Type::getInt32Ty(CI->getContext()), OffsetBytes);
  Value *Diff = Builder.CreateSub(Loads.Lhs, Loads.Rhs);

  PhiRes->addIncoming(Diff, BB);

  if (BlockIndex < (LoadCmpBlocks.size() - 1)) {
    // Early exit branch if difference found to EndBlock. Otherwise, continue to
    // next LoadCmpBlock,
    Value *Cmp = Builder.CreateICmp(ICmpInst::ICMP_NE, Diff,
                                    ConstantInt::get(Diff->getType(), 0));
    BranchInst *CmpBr =
        BranchInst::Create(EndBlock, LoadCmpBlocks[BlockIndex + 1], Cmp);
    if (DTU)
      DTU->applyUpdates(
          {{DominatorTree::Insert, BB, EndBlock},
           {DominatorTree::Insert, BB, LoadCmpBlocks[BlockIndex + 1]}});
    Builder.Insert(CmpBr);
  } else {
    // The last block has an unconditional branch to EndBlock.
    BranchInst *CmpBr = BranchInst::Create(EndBlock);
    if (DTU)
      DTU->applyUpdates({{DominatorTree::Insert, BB, EndBlock}});
    Builder.Insert(CmpBr);
  }
}

/// Generate an equality comparison for one or more pairs of loaded values.
/// This is used in the case where the memcmp() call is compared equal or not
/// equal to zero.
Value *MemCmpExpansion::getCompareLoadPairs(unsigned BlockIndex,
                                            unsigned &LoadIndex) {
  assert(LoadIndex < getNumLoads() &&
         "getCompareLoadPairs() called with no remaining loads");
  std::vector<Value *> XorList, OrList;
  Value *Diff = nullptr;

  const unsigned NumLoads =
      std::min(getNumLoads() - LoadIndex, NumLoadsPerBlockForZeroCmp);

  // For a single-block expansion, start inserting before the memcmp call.
  if (LoadCmpBlocks.empty())
    Builder.SetInsertPoint(CI);
  else
    Builder.SetInsertPoint(LoadCmpBlocks[BlockIndex]);

  Value *Cmp = nullptr;
  // If we have multiple loads per block, we need to generate a composite
  // comparison using xor+or. The type for the combinations is the largest load
  // type.
  IntegerType *const MaxLoadType =
      NumLoads == 1 ? nullptr
                    : IntegerType::get(CI->getContext(), MaxLoadSize * 8);
  for (unsigned i = 0; i < NumLoads; ++i, ++LoadIndex) {
    const LoadEntry &CurLoadEntry = LoadSequence[LoadIndex];
    const LoadPair Loads = getLoadPair(
        IntegerType::get(CI->getContext(), CurLoadEntry.LoadSize * 8),
        /*NeedsBSwap=*/false, MaxLoadType, CurLoadEntry.Offset);

    if (NumLoads != 1) {
      // If we have multiple loads per block, we need to generate a composite
      // comparison using xor+or.
      Diff = Builder.CreateXor(Loads.Lhs, Loads.Rhs);
      Diff = Builder.CreateZExt(Diff, MaxLoadType);
      XorList.push_back(Diff);
    } else {
      // If there's only one load per block, we just compare the loaded values.
      Cmp = Builder.CreateICmpNE(Loads.Lhs, Loads.Rhs);
    }
  }

  auto pairWiseOr = [&](std::vector<Value *> &InList) -> std::vector<Value *> {
    std::vector<Value *> OutList;
    for (unsigned i = 0; i < InList.size() - 1; i = i + 2) {
      Value *Or = Builder.CreateOr(InList[i], InList[i + 1]);
      OutList.push_back(Or);
    }
    if (InList.size() % 2 != 0)
      OutList.push_back(InList.back());
    return OutList;
  };

  if (!Cmp) {
    // Pairwise OR the XOR results.
    OrList = pairWiseOr(XorList);

    // Pairwise OR the OR results until one result left.
    while (OrList.size() != 1) {
      OrList = pairWiseOr(OrList);
    }

    assert(Diff && "Failed to find comparison diff");
    Cmp = Builder.CreateICmpNE(OrList[0], ConstantInt::get(Diff->getType(), 0));
  }

  return Cmp;
}

void MemCmpExpansion::emitLoadCompareBlockMultipleLoads(unsigned BlockIndex,
                                                        unsigned &LoadIndex) {
  Value *Cmp = getCompareLoadPairs(BlockIndex, LoadIndex);

  BasicBlock *NextBB = (BlockIndex == (LoadCmpBlocks.size() - 1))
                           ? EndBlock
                           : LoadCmpBlocks[BlockIndex + 1];
  // Early exit branch if difference found to ResultBlock. Otherwise,
  // continue to next LoadCmpBlock or EndBlock.
  BasicBlock *BB = Builder.GetInsertBlock();
  BranchInst *CmpBr = BranchInst::Create(ResBlock.BB, NextBB, Cmp);
  Builder.Insert(CmpBr);
  if (DTU)
    DTU->applyUpdates({{DominatorTree::Insert, BB, ResBlock.BB},
                       {DominatorTree::Insert, BB, NextBB}});

  // Add a phi edge for the last LoadCmpBlock to Endblock with a value of 0
  // since early exit to ResultBlock was not taken (no difference was found in
  // any of the bytes).
  if (BlockIndex == LoadCmpBlocks.size() - 1) {
    Value *Zero = ConstantInt::get(Type::getInt32Ty(CI->getContext()), 0);
    PhiRes->addIncoming(Zero, LoadCmpBlocks[BlockIndex]);
  }
}

// This function creates the IR intructions for loading and comparing using the
// given LoadSize. It loads the number of bytes specified by LoadSize from each
// source of the memcmp parameters. It then does a subtract to see if there was
// a difference in the loaded values. If a difference is found, it branches
// with an early exit to the ResultBlock for calculating which source was
// larger. Otherwise, it falls through to the either the next LoadCmpBlock or
// the EndBlock if this is the last LoadCmpBlock. Loading 1 byte is handled with
// a special case through emitLoadCompareByteBlock. The special handling can
// simply subtract the loaded values and add it to the result phi node.
void MemCmpExpansion::emitLoadCompareBlock(unsigned BlockIndex) {
  // There is one load per block in this case, BlockIndex == LoadIndex.
  const LoadEntry &CurLoadEntry = LoadSequence[BlockIndex];

  if (CurLoadEntry.LoadSize == 1) {
    MemCmpExpansion::emitLoadCompareByteBlock(BlockIndex, CurLoadEntry.Offset);
    return;
  }

  Type *LoadSizeType =
      IntegerType::get(CI->getContext(), CurLoadEntry.LoadSize * 8);
  Type *MaxLoadType = IntegerType::get(CI->getContext(), MaxLoadSize * 8);
  assert(CurLoadEntry.LoadSize <= MaxLoadSize && "Unexpected load type");

  Builder.SetInsertPoint(LoadCmpBlocks[BlockIndex]);

  const LoadPair Loads =
      getLoadPair(LoadSizeType, /*NeedsBSwap=*/DL.isLittleEndian(), MaxLoadType,
                  CurLoadEntry.Offset);

  // Add the loaded values to the phi nodes for calculating memcmp result only
  // if result is not used in a zero equality.
  if (!IsUsedForZeroCmp) {
    ResBlock.PhiSrc1->addIncoming(Loads.Lhs, LoadCmpBlocks[BlockIndex]);
    ResBlock.PhiSrc2->addIncoming(Loads.Rhs, LoadCmpBlocks[BlockIndex]);
  }

  Value *Cmp = Builder.CreateICmp(ICmpInst::ICMP_EQ, Loads.Lhs, Loads.Rhs);
  BasicBlock *NextBB = (BlockIndex == (LoadCmpBlocks.size() - 1))
                           ? EndBlock
                           : LoadCmpBlocks[BlockIndex + 1];
  // Early exit branch if difference found to ResultBlock. Otherwise, continue
  // to next LoadCmpBlock or EndBlock.
  BasicBlock *BB = Builder.GetInsertBlock();
  BranchInst *CmpBr = BranchInst::Create(NextBB, ResBlock.BB, Cmp);
  Builder.Insert(CmpBr);
  if (DTU)
    DTU->applyUpdates({{DominatorTree::Insert, BB, NextBB},
                       {DominatorTree::Insert, BB, ResBlock.BB}});

  // Add a phi edge for the last LoadCmpBlock to Endblock with a value of 0
  // since early exit to ResultBlock was not taken (no difference was found in
  // any of the bytes).
  if (BlockIndex == LoadCmpBlocks.size() - 1) {
    Value *Zero = ConstantInt::get(Type::getInt32Ty(CI->getContext()), 0);
    PhiRes->addIncoming(Zero, LoadCmpBlocks[BlockIndex]);
  }
}

// This function populates the ResultBlock with a sequence to calculate the
// memcmp result. It compares the two loaded source values and returns -1 if
// src1 < src2 and 1 if src1 > src2.
void MemCmpExpansion::emitMemCmpResultBlock() {
  // Special case: if memcmp result is used in a zero equality, result does not
  // need to be calculated and can simply return 1.
  if (IsUsedForZeroCmp) {
    BasicBlock::iterator InsertPt = ResBlock.BB->getFirstInsertionPt();
    Builder.SetInsertPoint(ResBlock.BB, InsertPt);
    Value *Res = ConstantInt::get(Type::getInt32Ty(CI->getContext()), 1);
    PhiRes->addIncoming(Res, ResBlock.BB);
    BranchInst *NewBr = BranchInst::Create(EndBlock);
    Builder.Insert(NewBr);
    if (DTU)
      DTU->applyUpdates({{DominatorTree::Insert, ResBlock.BB, EndBlock}});
    return;
  }
  BasicBlock::iterator InsertPt = ResBlock.BB->getFirstInsertionPt();
  Builder.SetInsertPoint(ResBlock.BB, InsertPt);

  Value *Cmp = Builder.CreateICmp(ICmpInst::ICMP_ULT, ResBlock.PhiSrc1,
                                  ResBlock.PhiSrc2);

  Value *Res =
      Builder.CreateSelect(Cmp, ConstantInt::get(Builder.getInt32Ty(), -1),
                           ConstantInt::get(Builder.getInt32Ty(), 1));

  PhiRes->addIncoming(Res, ResBlock.BB);
  BranchInst *NewBr = BranchInst::Create(EndBlock);
  Builder.Insert(NewBr);
  if (DTU)
    DTU->applyUpdates({{DominatorTree::Insert, ResBlock.BB, EndBlock}});
}

void MemCmpExpansion::setupResultBlockPHINodes() {
  Type *MaxLoadType = IntegerType::get(CI->getContext(), MaxLoadSize * 8);
  Builder.SetInsertPoint(ResBlock.BB);
  // Note: this assumes one load per block.
  ResBlock.PhiSrc1 =
      Builder.CreatePHI(MaxLoadType, NumLoadsNonOneByte, "phi.src1");
  ResBlock.PhiSrc2 =
      Builder.CreatePHI(MaxLoadType, NumLoadsNonOneByte, "phi.src2");
}

void MemCmpExpansion::setupEndBlockPHINodes() {
  Builder.SetInsertPoint(&EndBlock->front());
  PhiRes = Builder.CreatePHI(Type::getInt32Ty(CI->getContext()), 2, "phi.res");
}

Value *MemCmpExpansion::getMemCmpExpansionZeroCase() {
  unsigned LoadIndex = 0;
  // This loop populates each of the LoadCmpBlocks with the IR sequence to
  // handle multiple loads per block.
  for (unsigned I = 0; I < getNumBlocks(); ++I) {
    emitLoadCompareBlockMultipleLoads(I, LoadIndex);
  }

  emitMemCmpResultBlock();
  return PhiRes;
}

/// A memcmp expansion that compares equality with 0 and only has one block of
/// load and compare can bypass the compare, branch, and phi IR that is required
/// in the general case.
Value *MemCmpExpansion::getMemCmpEqZeroOneBlock() {
  unsigned LoadIndex = 0;
  Value *Cmp = getCompareLoadPairs(0, LoadIndex);
  assert(LoadIndex == getNumLoads() && "some entries were not consumed");
  return Builder.CreateZExt(Cmp, Type::getInt32Ty(CI->getContext()));
}

/// A memcmp expansion that only has one block of load and compare can bypass
/// the compare, branch, and phi IR that is required in the general case.
Value *MemCmpExpansion::getMemCmpOneBlock() {
  Type *LoadSizeType = IntegerType::get(CI->getContext(), Size * 8);
  bool NeedsBSwap = DL.isLittleEndian() && Size != 1;

  // The i8 and i16 cases don't need compares. We zext the loaded values and
  // subtract them to get the suitable negative, zero, or positive i32 result.
  if (Size < 4) {
    const LoadPair Loads =
        getLoadPair(LoadSizeType, NeedsBSwap, Builder.getInt32Ty(),
                    /*Offset*/ 0);
    return Builder.CreateSub(Loads.Lhs, Loads.Rhs);
  }

  const LoadPair Loads = getLoadPair(LoadSizeType, NeedsBSwap, LoadSizeType,
                                     /*Offset*/ 0);
  // The result of memcmp is negative, zero, or positive, so produce that by
  // subtracting 2 extended compare bits: sub (ugt, ult).
  // If a target prefers to use selects to get -1/0/1, they should be able
  // to transform this later. The inverse transform (going from selects to math)
  // may not be possible in the DAG because the selects got converted into
  // branches before we got there.
  Value *CmpUGT = Builder.CreateICmpUGT(Loads.Lhs, Loads.Rhs);
  Value *CmpULT = Builder.CreateICmpULT(Loads.Lhs, Loads.Rhs);
  Value *ZextUGT = Builder.CreateZExt(CmpUGT, Builder.getInt32Ty());
  Value *ZextULT = Builder.CreateZExt(CmpULT, Builder.getInt32Ty());
  return Builder.CreateSub(ZextUGT, ZextULT);
}

// This function expands the memcmp call into an inline expansion and returns
// the memcmp result.
Value *MemCmpExpansion::getMemCmpExpansion() {
  // Create the basic block framework for a multi-block expansion.
  if (getNumBlocks() != 1) {
    BasicBlock *StartBlock = CI->getParent();
    EndBlock = SplitBlock(StartBlock, CI, DTU, /*LI=*/nullptr,
                          /*MSSAU=*/nullptr, "endblock");
    setupEndBlockPHINodes();
    createResultBlock();

    // If return value of memcmp is not used in a zero equality, we need to
    // calculate which source was larger. The calculation requires the
    // two loaded source values of each load compare block.
    // These will be saved in the phi nodes created by setupResultBlockPHINodes.
    if (!IsUsedForZeroCmp) setupResultBlockPHINodes();

    // Create the number of required load compare basic blocks.
    createLoadCmpBlocks();

    // Update the terminator added by SplitBlock to branch to the first
    // LoadCmpBlock.
    StartBlock->getTerminator()->setSuccessor(0, LoadCmpBlocks[0]);
    if (DTU)
      DTU->applyUpdates({{DominatorTree::Insert, StartBlock, LoadCmpBlocks[0]},
                         {DominatorTree::Delete, StartBlock, EndBlock}});
  }

  Builder.SetCurrentDebugLocation(CI->getDebugLoc());

  if (IsUsedForZeroCmp)
    return getNumBlocks() == 1 ? getMemCmpEqZeroOneBlock()
                               : getMemCmpExpansionZeroCase();

  if (getNumBlocks() == 1)
    return getMemCmpOneBlock();

  for (unsigned I = 0; I < getNumBlocks(); ++I) {
    emitLoadCompareBlock(I);
  }

  emitMemCmpResultBlock();
  return PhiRes;
}

// This function checks to see if an expansion of memcmp can be generated.
// It checks for constant compare size that is less than the max inline size.
// If an expansion cannot occur, returns false to leave as a library call.
// Otherwise, the library call is replaced with a new IR instruction sequence.
/// We want to transform:
/// %call = call signext i32 @memcmp(i8* %0, i8* %1, i64 15)
/// To:
/// loadbb:
///  %0 = bitcast i32* %buffer2 to i8*
///  %1 = bitcast i32* %buffer1 to i8*
///  %2 = bitcast i8* %1 to i64*
///  %3 = bitcast i8* %0 to i64*
///  %4 = load i64, i64* %2
///  %5 = load i64, i64* %3
///  %6 = call i64 @llvm.bswap.i64(i64 %4)
///  %7 = call i64 @llvm.bswap.i64(i64 %5)
///  %8 = sub i64 %6, %7
///  %9 = icmp ne i64 %8, 0
///  br i1 %9, label %res_block, label %loadbb1
/// res_block:                                        ; preds = %loadbb2,
/// %loadbb1, %loadbb
///  %phi.src1 = phi i64 [ %6, %loadbb ], [ %22, %loadbb1 ], [ %36, %loadbb2 ]
///  %phi.src2 = phi i64 [ %7, %loadbb ], [ %23, %loadbb1 ], [ %37, %loadbb2 ]
///  %10 = icmp ult i64 %phi.src1, %phi.src2
///  %11 = select i1 %10, i32 -1, i32 1
///  br label %endblock
/// loadbb1:                                          ; preds = %loadbb
///  %12 = bitcast i32* %buffer2 to i8*
///  %13 = bitcast i32* %buffer1 to i8*
///  %14 = bitcast i8* %13 to i32*
///  %15 = bitcast i8* %12 to i32*
///  %16 = getelementptr i32, i32* %14, i32 2
///  %17 = getelementptr i32, i32* %15, i32 2
///  %18 = load i32, i32* %16
///  %19 = load i32, i32* %17
///  %20 = call i32 @llvm.bswap.i32(i32 %18)
///  %21 = call i32 @llvm.bswap.i32(i32 %19)
///  %22 = zext i32 %20 to i64
///  %23 = zext i32 %21 to i64
///  %24 = sub i64 %22, %23
///  %25 = icmp ne i64 %24, 0
///  br i1 %25, label %res_block, label %loadbb2
/// loadbb2:                                          ; preds = %loadbb1
///  %26 = bitcast i32* %buffer2 to i8*
///  %27 = bitcast i32* %buffer1 to i8*
///  %28 = bitcast i8* %27 to i16*
///  %29 = bitcast i8* %26 to i16*
///  %30 = getelementptr i16, i16* %28, i16 6
///  %31 = getelementptr i16, i16* %29, i16 6
///  %32 = load i16, i16* %30
///  %33 = load i16, i16* %31
///  %34 = call i16 @llvm.bswap.i16(i16 %32)
///  %35 = call i16 @llvm.bswap.i16(i16 %33)
///  %36 = zext i16 %34 to i64
///  %37 = zext i16 %35 to i64
///  %38 = sub i64 %36, %37
///  %39 = icmp ne i64 %38, 0
///  br i1 %39, label %res_block, label %loadbb3
/// loadbb3:                                          ; preds = %loadbb2
///  %40 = bitcast i32* %buffer2 to i8*
///  %41 = bitcast i32* %buffer1 to i8*
///  %42 = getelementptr i8, i8* %41, i8 14
///  %43 = getelementptr i8, i8* %40, i8 14
///  %44 = load i8, i8* %42
///  %45 = load i8, i8* %43
///  %46 = zext i8 %44 to i32
///  %47 = zext i8 %45 to i32
///  %48 = sub i32 %46, %47
///  br label %endblock
/// endblock:                                         ; preds = %res_block,
/// %loadbb3
///  %phi.res = phi i32 [ %48, %loadbb3 ], [ %11, %res_block ]
///  ret i32 %phi.res
static bool expandMemCmp(CallInst *CI, const TargetTransformInfo *TTI,
                         const TargetLowering *TLI, const DataLayout *DL,
                         ProfileSummaryInfo *PSI, BlockFrequencyInfo *BFI,
                         DomTreeUpdater *DTU) {
  NumMemCmpCalls++;

  // Early exit from expansion if -Oz.
  if (CI->getFunction()->hasMinSize())
    return false;

  // Early exit from expansion if size is not a constant.
  ConstantInt *SizeCast = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!SizeCast) {
    NumMemCmpNotConstant++;
    return false;
  }
  const uint64_t SizeVal = SizeCast->getZExtValue();

  if (SizeVal == 0) {
    return false;
  }
  // TTI call to check if target would like to expand memcmp. Also, get the
  // available load sizes.
  const bool IsUsedForZeroCmp = isOnlyUsedInZeroEqualityComparison(CI);
  bool OptForSize = CI->getFunction()->hasOptSize() ||
                    llvm::shouldOptimizeForSize(CI->getParent(), PSI, BFI);
  auto Options = TTI->enableMemCmpExpansion(OptForSize,
                                            IsUsedForZeroCmp);
  if (!Options) return false;

  if (MemCmpEqZeroNumLoadsPerBlock.getNumOccurrences())
    Options.NumLoadsPerBlock = MemCmpEqZeroNumLoadsPerBlock;

  if (OptForSize &&
      MaxLoadsPerMemcmpOptSize.getNumOccurrences())
    Options.MaxNumLoads = MaxLoadsPerMemcmpOptSize;

  if (!OptForSize && MaxLoadsPerMemcmp.getNumOccurrences())
    Options.MaxNumLoads = MaxLoadsPerMemcmp;

  MemCmpExpansion Expansion(CI, SizeVal, Options, IsUsedForZeroCmp, *DL, DTU);

  // Don't expand if this will require more loads than desired by the target.
  if (Expansion.getNumLoads() == 0) {
    NumMemCmpGreaterThanMax++;
    return false;
  }

  NumMemCmpInlined++;

  Value *Res = Expansion.getMemCmpExpansion();

  // Replace call with result of expansion and erase call.
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();

  return true;
}

class ExpandMemCmpPass : public FunctionPass {
public:
  static char ID;

  ExpandMemCmpPass() : FunctionPass(ID) {
    initializeExpandMemCmpPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) return false;

    auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
    if (!TPC) {
      return false;
    }
    const TargetLowering* TL =
        TPC->getTM<TargetMachine>().getSubtargetImpl(F)->getTargetLowering();

    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    const TargetTransformInfo *TTI =
        &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
    auto *BFI = (PSI && PSI->hasProfileSummary()) ?
           &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI() :
           nullptr;
    DominatorTree *DT = nullptr;
    if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
      DT = &DTWP->getDomTree();
    auto PA = runImpl(F, TLI, TTI, TL, PSI, BFI, DT);
    return !PA.areAllPreserved();
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
    FunctionPass::getAnalysisUsage(AU);
  }

  PreservedAnalyses runImpl(Function &F, const TargetLibraryInfo *TLI,
                            const TargetTransformInfo *TTI,
                            const TargetLowering *TL, ProfileSummaryInfo *PSI,
                            BlockFrequencyInfo *BFI, DominatorTree *DT);
  // Returns true if a change was made.
  bool runOnBlock(BasicBlock &BB, const TargetLibraryInfo *TLI,
                  const TargetTransformInfo *TTI, const TargetLowering *TL,
                  const DataLayout &DL, ProfileSummaryInfo *PSI,
                  BlockFrequencyInfo *BFI, DomTreeUpdater *DTU);
};

bool ExpandMemCmpPass::runOnBlock(BasicBlock &BB, const TargetLibraryInfo *TLI,
                                  const TargetTransformInfo *TTI,
                                  const TargetLowering *TL,
                                  const DataLayout &DL, ProfileSummaryInfo *PSI,
                                  BlockFrequencyInfo *BFI,
                                  DomTreeUpdater *DTU) {
  for (Instruction& I : BB) {
    CallInst *CI = dyn_cast<CallInst>(&I);
    if (!CI) {
      continue;
    }
    LibFunc Func;
    if (TLI->getLibFunc(*CI, Func) &&
        (Func == LibFunc_memcmp || Func == LibFunc_bcmp) &&
        expandMemCmp(CI, TTI, TL, &DL, PSI, BFI, DTU)) {
      return true;
    }
  }
  return false;
}

PreservedAnalyses
ExpandMemCmpPass::runImpl(Function &F, const TargetLibraryInfo *TLI,
                          const TargetTransformInfo *TTI,
                          const TargetLowering *TL, ProfileSummaryInfo *PSI,
                          BlockFrequencyInfo *BFI, DominatorTree *DT) {
  Optional<DomTreeUpdater> DTU;
  if (DT)
    DTU.emplace(DT, DomTreeUpdater::UpdateStrategy::Lazy);

  const DataLayout& DL = F.getParent()->getDataLayout();
  bool MadeChanges = false;
  for (auto BBIt = F.begin(); BBIt != F.end();) {
    if (runOnBlock(*BBIt, TLI, TTI, TL, DL, PSI, BFI,
                   DTU.hasValue() ? DTU.getPointer() : nullptr)) {
      MadeChanges = true;
      // If changes were made, restart the function from the beginning, since
      // the structure of the function was changed.
      BBIt = F.begin();
    } else {
      ++BBIt;
    }
  }
  if (MadeChanges)
    for (BasicBlock &BB : F)
      SimplifyInstructionsInBlock(&BB);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

} // namespace

char ExpandMemCmpPass::ID = 0;
INITIALIZE_PASS_BEGIN(ExpandMemCmpPass, "expandmemcmp",
                      "Expand memcmp() to load/stores", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LazyBlockFrequencyInfoPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(ExpandMemCmpPass, "expandmemcmp",
                    "Expand memcmp() to load/stores", false, false)

FunctionPass *llvm::createExpandMemCmpPass() {
  return new ExpandMemCmpPass();
}
